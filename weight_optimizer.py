import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tqdm import tqdm
import itertools
import json
import concurrent.futures
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# 내부 모듈 임포트
from train_model import fetch_stock_list, get_financial_data_for_training_http, fetch_and_process_ticker_data
from scoring import calculate_factor_scores

from scoring import calculate_factor_scores

# --- 설정 변수 ---
VALIDATION_START_DATE = '2023-01-01'
VALIDATION_END_DATE = '2023-12-31'
TRAIN_END_DATE = '2022-12-31'
TRAIN_START_DATE = '2021-01-01'

# 가중치 탐색 후보군 정의
WEIGHT_GRID = {
    'value_score': [0.1, 0.2, 0.3],
    'quality_score': [0.1, 0.2, 0.3],
    'momentum_score': [0.1, 0.2, 0.3],
    'volatility_score': [0.1, 0.2],
    'ml_pred_proba': [0.2, 0.3, 0.4, 0.5]
}

# --- 함수 정의 ---

def prepare_full_data(start_date, end_date):
    """지정된 기간 동안의 모든 종목에 대한 피처 및 타겟이 포함된 전체 데이터프레임을 생성합니다."""
    print(f"데이터 준비 중 ({start_date} ~ {end_date})...")
    stock_list = fetch_stock_list()
    if stock_list.empty:
        raise ValueError("종목 리스트를 가져올 수 없습니다.")

    try:
        df_corp_map = pd.read_csv('corp_code_map.csv', dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError:
        raise FileNotFoundError("corp_code_map.csv 파일을 찾을 수 없습니다. make_corp_map.py를 먼저 실행해주세요.")

    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    all_fs_data = get_financial_data_for_training_http(corp_codes, start_year - 1, end_year)
    if not all_fs_data:
        raise ValueError("재무 데이터를 가져올 수 없습니다.")

    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_and_process_ticker_data, row, start_date, end_date, all_fs_data): row
            for row in target_stocks.to_dict('records')
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_stocks), desc="피처 데이터 생성"):
            result_df = future.result()
            if result_df is not None:
                all_data.append(result_df)

    if not all_data:
        raise ValueError("처리된 데이터가 없습니다.")

    final_df = pd.concat(all_data)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(inplace=True) # 계산 과정에서 생긴 결측치 제거
    
    # 날짜를 인덱스로 설정
    final_df.index = pd.to_datetime(final_df.index)
    final_df.sort_index(inplace=True)
    
    print("데이터 준비 완료.")
    return final_df

def get_model_and_data():
    """최적화를 위해 특정 기간의 모델과 데이터를 준비합니다."""
    print("1. 최적화를 위한 데이터 준비 및 모델 학습 시작...")

    # --- 1. 전체 기간 데이터 준비 (2021-01-01 ~ 2023-12-31) ---
    full_data_df = prepare_full_data(TRAIN_START_DATE, VALIDATION_END_DATE)

    # --- 2. 훈련/검증 데이터 분리 ---
    train_df = full_data_df[full_data_df.index <= TRAIN_END_DATE]
    validation_df = full_data_df[full_data_df.index >= VALIDATION_START_DATE]
    print(f"  - 훈련 데이터({TRAIN_START_DATE}~{TRAIN_END_DATE}) {len(train_df)} 행 준비 완료.")
    print(f"  - 검증 데이터({VALIDATION_START_DATE}~{VALIDATION_END_DATE}) {len(validation_df)} 행 준비 완료.")

    # --- 3. 모델 학습 ---
    features = [
        '수익률(1W)', '수익률(2W)', '수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', '거래대금_MA20', '단기 정배열', '52주_신고가_비율'
    ]
    X_train = train_df[features].astype(np.float32)
    y_train = train_df['target']

    print("  - 임시 모델 학습 중 (RandomForestClassifier)...")
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    print("  - 임시 모델 학습 완료.")

    # --- 4. 검증 데이터에 예측 및 점수 추가 ---
    print("  - 검증 데이터에 ML 예측 및 팩터 점수 추가 중...")
    validation_df['ml_pred_proba'] = model.predict_proba(validation_df[features])[:, 1]

    scored_data_list = []
    for date in tqdm(validation_df.index.unique(), desc="팩터 점수 계산"):
        daily_data = validation_df.loc[date].copy()
        daily_scored_data = calculate_factor_scores(daily_data.reset_index())
        daily_scored_data['date'] = date
        scored_data_list.append(daily_scored_data)

    if not scored_data_list:
        raise ValueError("점수 계산된 데이터가 없습니다.")

    all_scored_df = pd.concat(scored_data_list)
    all_scored_df.reset_index(inplace=True) # 'date' 인덱스를 컬럼으로 변환
    
    # 점수들을 기존 검증 데이터프레임에 병합
    score_cols_to_merge = ['종목코드', 'date'] + [col for col in all_scored_df.columns if '_score' in col]
    validation_df.reset_index(inplace=True)
    validation_df = pd.merge(validation_df, all_scored_df[score_cols_to_merge], on=['date', '종목코드'], how='left')
    
    # 백테스팅을 위해 (date, 종목코드) MultiIndex 설정
    validation_df.set_index(['date', '종목코드'], inplace=True)
    validation_df.sort_index(inplace=True)

    print("  - 모든 데이터 준비 완료.")
    return validation_df

def run_backtest_for_weights(weights, data, initial_capital=1_000_000_000, top_n=5):
    """주어진 가중치로 상세한 일별 백테스트를 실행하고 샤프 지수를 반환합니다."""
    
    # --- 1. 일별 final_score 계산 ---
    score_cols = [col for col in weights.keys() if col in data.columns]
    data['final_score'] = np.zeros(len(data))
    for col in score_cols:
        data['final_score'] += data[col].fillna(0) * weights[col]
    
    # 일별로 final_score를 0-100으로 정규화 (Min-Max Scaling)
    data['final_score'] = data.groupby(level=0)['final_score'].transform(
        lambda x: 100 * (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 50
    )

    # --- 2. 시뮬레이션 준비 ---
    cash = initial_capital
    portfolio = {}
    portfolio_history = []
    daily_dates = data.index.get_level_values('date').unique().sort_values()

    # --- 3. 일별 시뮬레이션 루프 ---
    for date in tqdm(daily_dates, desc="시뮬레이션 중", leave=False):
        # --- 3.1. 매도 로직 ---
        for ticker in list(portfolio.keys()):
            stock_info = portfolio[ticker]
            days_held = (date - stock_info['buy_date']).days
            current_price = data.loc[date, data['종목코드'] == ticker]['종가'].iloc[0]
            
            # 매도 조건 확인 (수익률은 종가 기준)
            sell_condition = (
                (current_price >= stock_info['buy_price'] * 1.05) or # 익절
                (current_price <= stock_info['buy_price'] * 0.97) or # 손절
                (days_held >= 15) # 기간 만료
            )
            
            if sell_condition:
                cash += current_price * stock_info['shares']
                del portfolio[ticker]

        # --- 3.2. 매수 로직 ---
        investment_per_stock = cash / top_n # 가용 현금을 N등분하여 투자
        daily_data = data.loc[date]
        buy_candidates = daily_data[~daily_data['종목코드'].isin(portfolio.keys())].nlargest(top_n, 'final_score')

        for i, row in buy_candidates.iterrows():
            if cash >= investment_per_stock:
                ticker = row['종목코드']
                buy_price = row['종가']
                shares = investment_per_stock // buy_price
                
                if shares > 0:
                    cash -= buy_price * shares
                    portfolio[ticker] = {
                        'buy_date': date,
                        'buy_price': buy_price,
                        'shares': shares
                    }

        # --- 3.3. 일일 포트폴리오 가치 기록 ---
        current_portfolio_value = 0
        for ticker, stock_info in portfolio.items():
            current_price = data.loc[date, data['종목코드'] == ticker]['종가'].iloc[0]
            current_portfolio_value += current_price * stock_info['shares']
        
        total_asset = cash + current_portfolio_value
        portfolio_history.append(total_asset)

    # --- 4. 최종 성과 계산 ---
    portfolio_ts = pd.Series(portfolio_history, index=daily_dates)
    daily_returns = portfolio_ts.pct_change().fillna(0)

    if daily_returns.std() == 0:
        return 0.0
        
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    return sharpe_ratio

def find_optimal_weights():
    """그리드 서치를 통해 최적의 가중치를 찾습니다."""
    
    model, data = get_model_and_data()
    
    print("2. 가중치 조합 생성 및 탐색 시작...")
    keys = list(WEIGHT_GRID.keys())
    value_lists = list(WEIGHT_GRID.values())
    
    # 모든 가능한 조합을 생성 (itertools.product)
    all_combinations = list(itertools.product(*value_lists))
    
    # 가중치의 합이 1.0에 가까운 조합만 필터링 (부동소수점 오차 감안)
    valid_combinations = [dict(zip(keys, combo)) for combo in all_combinations if np.isclose(sum(combo), 1.0)]
    
    print(f"총 {len(valid_combinations)}개의 유효한 가중치 조합을 찾았습니다.")

    best_weights = None
    best_sharpe = -np.inf
    
    # 각 조합을 테스트하는 루프
    for weights in tqdm(valid_combinations, desc="가중치 최적화 중"):
        sharpe = run_backtest_for_weights(weights, data)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

    print(f"\n3. 최적 가중치 탐색 완료!")
    print(f"  - 최적 샤프 지수: {best_sharpe:.4f}")
    print(f"  - 최적 가중치: {best_weights}")

    # 결과를 파일로 저장
    if best_weights:
        with open('optimal_weights.json', 'w') as f:
            json.dump(best_weights, f, indent=4)
        print("\n`optimal_weights.json` 파일에 최적 가중치를 저장했습니다.")
    else:
        print("\n유효한 최적 가중치를 찾지 못했습니다.")


# --- 메인 실행부 ---
if __name__ == '__main__':
    find_optimal_weights()
