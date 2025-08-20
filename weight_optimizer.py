import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import itertools
import json
import concurrent.futures
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import FinanceDataReader as fdr
import os
import requests
import time

# 내부 모듈 임포트
from scoring import calculate_factor_scores
import ensemble

# --- 설정 변수 ---
VALIDATION_START_DATE = '2023-01-01'
VALIDATION_END_DATE = '2023-12-31'
TRAIN_END_DATE = '2022-12-31'
TRAIN_START_DATE = '2020-01-01'
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522" # 본인의 키로 교체

WEIGHT_GRID = {
    'value_score': np.arange(0.0, 0.31, 0.1),
    'quality_score': np.arange(0.0, 0.31, 0.1),
    'momentum_score': np.arange(0.0, 0.41, 0.1),
    'supply_score': np.arange(0.0, 0.21, 0.1),
    'volatility_score': np.arange(0.0, 0.21, 0.1),
    'ml_pred_proba': np.arange(0.1, 0.51, 0.1),
}

def get_financial_data_for_training_http(corp_codes, start_year, end_year):
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요": return {}
    all_fs_data = {}
    for year in range(start_year, end_year + 1):
        print(f"HTTP 통신으로 {year}년 재무 데이터 수집 중 (최적화용)...")
        year_fs_data = []
        for i in tqdm(range(0, len(corp_codes), 100), desc=f"{year}년 재무 데이터"):
            corp_code_chunk = corp_codes[i:i+100]
            corp_code_str = ','.join(corp_code_chunk)
            url = "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
            params = { 'crtfc_key': DART_API_KEY, 'corp_code': corp_code_str, 'bsns_year': str(year), 'reprt_code': '11011' }
            try:
                res = requests.get(url, params=params)
                if res.status_code != 200:
                    print(f"  [오류] DART API 요청 실패 (상태 코드: {res.status_code}) 응답: {res.text}")
                    continue
                data = res.json()
                if data.get('status') != '000':
                    print(f"  [오류] DART API가 에러를 반환했습니다. (상태: {data.get('status')}, 메시지: {data.get('message')})")
                    continue
                year_fs_data.extend(data['list'])
            except Exception as e:
                print(f"  [오류] DART API 처리 중 예외 발생: {e}")
                continue
            time.sleep(0.1)
        if year_fs_data:
            df = pd.DataFrame(year_fs_data)
            df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'].str.replace(',', ''), errors='coerce')
            df_pivot = df.pivot_table(index='stock_code', columns='account_nm', values='thstrm_amount')
            all_fs_data[year] = df_pivot.to_dict('index')
    return all_fs_data

def fetch_stock_list():
    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        stock_list = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        stock_list = stock_list[~stock_list['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
        return stock_list[['종목코드', '종목명']]
    except Exception: return pd.DataFrame()

def process_single_ticker_data(stock_info, start_date, end_date, all_fs_data, df_marcap_long):
    ticker = stock_info['종목코드']
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 251 + 60: return None
        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        df = df_price[['종가', '거래량']].copy()
        df['연도'] = df.index.year
        
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy()
        if df_marcap_ticker.empty: return None
        df.sort_index(inplace=True)
        df_marcap_ticker.sort_values(by='Date', inplace=True)
        
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['Date', 'Marcap']],
                           left_index=True, right_on='Date', direction='backward')
        
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        df['거래대금'] = df['종가'] * df['거래량']
        df.dropna(subset=['시가총액'], inplace=True)
        if df.empty: return None
        
        for year, fs_year_data in all_fs_data.items():
            if ticker in fs_year_data:
                fs_data = fs_year_data[ticker]
                df.loc[df['연도'] == year, '당기순이익'] = fs_data.get('당기순이익')
                df.loc[df['연도'] == year, '자본총계'] = fs_data.get('자본총계')
        if '당기순이익' not in df.columns or '자본총계' not in df.columns: return None
        df[['당기순이익', '자본총계']] = df[['당기순이익', '자본총계']].ffill()
        if df[['당기순이익', '자본총계']].isnull().values.any(): return None
        df['PER'] = df['시가총액'] / df['당기순이익']
        df['PBR'] = df['시가총액'] / df['자본총계']
        df['ROE'] = df['당기순이익'] / df['자본총계']
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(window=20).mean()
        df['target'] = (df['종가'].shift(-15) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker
        
        df.set_index('Date', inplace=True)
        return df
    except Exception: return None

def prepare_full_data(start_date, end_date):
    print(f"데이터 준비 중 ({start_date} ~ {end_date})...")
    stock_list = fetch_stock_list()
    if stock_list.empty: raise ValueError("종목 리스트를 가져올 수 없습니다.")
    try:
        df_corp_map = pd.read_csv('corp_code_map.csv', dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError: raise FileNotFoundError("corp_code_map.csv 파일이 없습니다.")
    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()
    all_fs_data = get_financial_data_for_training_http(corp_codes, int(start_date[:4]) - 1, int(end_date[:4]))
    if not all_fs_data: raise ValueError("재무 데이터를 가져올 수 없습니다. DART API 키/서버 상태를 확인하세요.")
    try:
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y%m%d').tolist()
        marcap_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(fdr.StockListing, 'KRX-MARCAP', date): date for date in month_end_dates}
            for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(month_end_dates), desc="시가총액 데이터 수집 (최적화용)"):
                try:
                    date_str = future_to_date[future]
                    result_df = future.result()
                    if not result_df.empty:
                        result_df['Date'] = pd.to_datetime(date_str)
                        marcap_dfs.append(result_df)
                except Exception: continue
        if not marcap_dfs: raise Exception("수집된 시가총액 데이터가 없습니다.")
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'Date'], inplace=True)
    except Exception as e:
        raise ConnectionError(f"과거 시가총액 데이터 수집 실패: {e}")
    all_data = []
    stock_records = target_stocks.to_dict('records')
    for row in tqdm(stock_records, desc="피처 데이터 생성 (최적화용)"):
        result_df = process_single_ticker_data(row, start_date, end_date, all_fs_data, df_marcap_long)
        if result_df is not None:
            all_data.append(result_df)
    if not all_data: raise ValueError("처리된 데이터가 없습니다.")
    final_df = pd.concat(all_data).reset_index()
    final_df.rename(columns={'Date': 'date'}, inplace=True)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['date', '종목코드'], inplace=True)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df.sort_values(by=['date', '종목코드'], inplace=True)
    print("데이터 준비 완료.")
    return final_df

def get_model_and_data():
    print("1. 최적화를 위한 데이터 준비 및 모델 학습 시작...")
    full_data_df = prepare_full_data(TRAIN_START_DATE, VALIDATION_END_DATE)
    train_df = full_data_df[full_data_df['date'] <= pd.to_datetime(TRAIN_END_DATE)].copy()
    validation_df = full_data_df[full_data_df['date'] >= pd.to_datetime(VALIDATION_START_DATE)].copy()
    print(f"  - 훈련 데이터 {len(train_df)} 행, 검증 데이터 {len(validation_df)} 행 준비 완료.")
    features = ['수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', '거래대금_MA20']
    train_df.dropna(subset=features + ['target'], inplace=True)
    validation_df.dropna(subset=features + ['종가'], inplace=True)
    X_train = train_df[features]
    y_train = train_df['target']
    print("  - 임시 모델 학습 중 (RandomForestClassifier)...")
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    print("  - 임시 모델 학습 완료.")
    print("  - 검증 데이터에 ML 예측 및 팩터 점수 추가 중...")
    validation_df.loc[:, 'ml_pred_proba'] = model.predict_proba(validation_df[features])[:, 1]
    scored_data_list = []
    grouped = validation_df.groupby('date')
    for date, daily_data in tqdm(grouped, desc="일별 팩터 점수 계산"):
        scored_daily = calculate_factor_scores(daily_data.copy().reset_index())
        scored_daily['date'] = date
        scored_data_list.append(scored_daily)
    if not scored_data_list: raise ValueError("점수 계산된 데이터가 없습니다.")
    all_scored_df = pd.concat(scored_data_list)
    score_cols = ['종목코드', 'date', 'value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
    validation_df.reset_index(inplace=True, drop=True)
    validation_df = pd.merge(validation_df, all_scored_df[score_cols], on=['date', '종목코드'], how='left')
    validation_df.set_index(['date', '종목코드'], inplace=True)
    validation_df.sort_index(inplace=True)
    print("  - 모든 데이터 준비 완료.")
    return validation_df

def run_backtest_for_weights(weights_tuple):
    weights, data, initial_capital, top_n = weights_tuple
    backtest_data = data.copy()
    backtest_data['final_score'] = 0
    total_weight = sum(weights.values())
    if total_weight == 0: return 0.0
    for factor, weight in weights.items():
        if factor in backtest_data.columns:
            norm_col = factor + '_norm'
            backtest_data[norm_col] = backtest_data.groupby(level='date')[factor].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0.5
            )
            backtest_data['final_score'] += backtest_data[norm_col].fillna(0.5) * (weight / total_weight)
    cash = initial_capital
    portfolio = {}
    portfolio_history = []
    daily_dates = backtest_data.index.get_level_values('date').unique().sort_values()
    for date in daily_dates:
        for ticker in list(portfolio.keys()):
            if 'buy_date' in portfolio[ticker] and (date - portfolio[ticker]['buy_date']).days >= 15:
                if (date, ticker) in backtest_data.index:
                    current_price = backtest_data.loc[(date, ticker), '종가']
                    cash += current_price * portfolio[ticker]['shares']
                    del portfolio[ticker]
        investment_per_stock = cash / top_n if top_n > 0 else 0
        if date in backtest_data.index.get_level_values('date'):
            daily_candidates = backtest_data.loc[date].nlargest(top_n, 'final_score')
            for ticker, row in daily_candidates.iterrows():
                if cash >= investment_per_stock:
                    buy_price = row['종가']
                    shares = investment_per_stock // buy_price
                    if shares > 0:
                        cash -= buy_price * shares
                        portfolio[ticker] = {'shares': shares, 'buy_date': date, 'buy_price': buy_price}
        portfolio_value = sum(backtest_data.loc[(date, ticker), '종가'] * info['shares'] for ticker, info in portfolio.items() if (date, ticker) in backtest_data.index)
        total_asset = cash + portfolio_value
        portfolio_history.append(total_asset)
    portfolio_ts = pd.Series(portfolio_history, index=daily_dates)
    daily_returns = portfolio_ts.pct_change().fillna(0)
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0

def find_optimal_weights(top_n_stocks, data):
    print("2. 가중치 조합 생성 및 탐색 시작...")
    keys = list(WEIGHT_GRID.keys())
    value_lists = list(WEIGHT_GRID.values())
    all_combinations = list(itertools.product(*value_lists))
    valid_combinations = [dict(zip(keys, combo)) for combo in all_combinations if np.isclose(sum(combo), 1.0)]
    print(f"총 {len(valid_combinations)}개의 유효한 가중치 조합을 테스트합니다.")
    best_weights = None
    best_sharpe = -np.inf
    tasks = [(w, data, 1_000_000_000, top_n_stocks) for w in valid_combinations]
    results = []
    for task in tqdm(tasks, desc="가중치 최적화 중"):
        results.append(run_backtest_for_weights(task))
    for weights, sharpe in zip(valid_combinations, results):
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
            print(f"\n새로운 최적 가중치 발견! Sharpe: {best_sharpe:.4f}, Weights: {best_weights}")
    return best_sharpe, best_weights

if __name__ == '__main__':
    top_n = int(input("시뮬레이션 시 매수할 상위 종목 수를 입력하세요 (예: 5): "))
    validation_data = get_model_and_data()
    best_sharpe, best_weights = find_optimal_weights(top_n_stocks=top_n, data=validation_data)
    print(f"\n3. 최적 가중치 탐색 완료!")
    print(f"  - 최적 샤프 지수: {best_sharpe:.4f}")
    print(f"  - 최적 가중치: {best_weights}")
    if best_weights:
        with open('optimal_weights.json', 'w') as f:
            json.dump(best_weights, f, indent=4)
        print("`optimal_weights.json` 파일에 최적 가중치를 저장했습니다.")
    else:
        print("유효한 최적 가중치를 찾지 못했습니다.")
