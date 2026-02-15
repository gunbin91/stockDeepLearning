"""
가중치 최적화 스크립트
===================

이 파일은 팩터 점수와 머신러닝 예측 결과의 최적 가중치를 찾습니다.
샤프 지수를 최대화하는 가중치 조합을 탐색하여 투자 성과를 극대화합니다.

주요 기능:
- 다양한 가중치 조합 탐색
- 샤프 지수 기반 성과 평가
- 병렬 처리를 통한 빠른 최적화
- 최적 가중치 저장
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import itertools
import json
import concurrent.futures
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import os
import sys
import io
import argparse
import locale
import platform

# Windows 환경에서 로케일 설정 (FinanceDataReader 내부 오류 방지)
if platform.system() == 'Windows':
    try:
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG'] = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        # 로케일 설정 실패 시 기본값 유지
        pass

# stdout/stderr를 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 내부 모듈 임포트
import ensemble
import data_processor
from path_manager import path_manager

# --- 설정 변수 ---
VALIDATION_START_DATE = '2023-01-01'
VALIDATION_END_DATE = '2023-12-31'
TRAIN_END_DATE = '2022-12-31'
TRAIN_START_DATE = '2020-01-01'

WEIGHT_GRID = {
    'ml_pred_proba': np.arange(0.1, 1.01, 0.1),
}

def get_model_and_data():
    print("1. 최적화를 위한 데이터 준비 및 모델 학습 시작...")
    full_data_df = data_processor.get_preprocessed_data(TRAIN_START_DATE, VALIDATION_END_DATE)
    
    train_df = full_data_df[full_data_df['date'] <= pd.to_datetime(TRAIN_END_DATE)].copy()
    validation_df = full_data_df[full_data_df['date'] >= pd.to_datetime(VALIDATION_START_DATE)].copy()
    
    print(f"  - 훈련 데이터 {len(train_df)} 행, 검증 데이터 {len(validation_df)} 행 준비 완료.")
    
    # NASDAQ 버전: data_processor에서 생성되는 피처 기준으로 구성
    features = [
        'log_mktcap',
        'disparity_20',
        'disparity_120',
        'disparity_240',
        'MA20_Slope',
        'MA120_Slope',
        'MA240_Slope',
        'RSI_Signal_Oscillator',
        'ATRr_5',
        'ATRr_20',
        'HV_Volatility_5',
        'HV_Volatility_20',
        'HV_Volatility_60',
        'VWAP_Disparity_5',
        'Max_Drawdown_20',
        'Trend_Pullback_Score',
        'Position_Range_60',
        'IXIC_pct_1d',
        'IXIC_disparity_20',
        'IXIC_MA20_Slope',
        'VIX',
    ]
    
    train_df.dropna(subset=features + ['target'], inplace=True)
    validation_df.dropna(subset=['종가'], inplace=True)
    
    X_train = train_df[features]
    y_train = train_df['target']
    
    print("  - 임시 모델 학습 중 (RandomForestClassifier)...")
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    print("  - 임시 모델 학습 완료.")
    
    print("  - 검증 데이터에 ML 예측 추가 중...")
    X_val = validation_df[features].copy()
    X_val.fillna(0, inplace=True)
    validation_df.loc[:, 'ml_pred_proba'] = model.predict_proba(X_val)[:, 1]
    
    print("  - 팩터 점수는 데이터 로딩 시 이미 계산되었습니다.")
    validation_df.set_index(['date', '종목코드'], inplace=True)
    validation_df.sort_index(inplace=True)

    if not validation_df.index.is_unique:
        print("[경고] 최종 검증 데이터에 중복된 인덱스가 발견되었습니다.")

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
    parser = argparse.ArgumentParser(description='Find optimal weights for the scoring model.')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of top stocks to buy in simulation (default: 5)')
    args = parser.parse_args()

    if args.top_n <= 0:
        print("Error: top_n must be a positive number.")
    else:
        validation_data = get_model_and_data()
        best_sharpe, best_weights = find_optimal_weights(top_n_stocks=args.top_n, data=validation_data)
        print(f"\n3. 최적 가중치 탐색 완료!")
        print(f"  - 최적 샤프 지수: {best_sharpe:.4f}")
        print(f"  - 최적 가중치: {best_weights}")
        if best_weights:
            output_path = str(path_manager.get_weights_path())
            with open(output_path, 'w') as f:
                json.dump(best_weights, f, indent=4)
            print(f"`{output_path}` 파일에 최적 가중치를 저장했습니다.")
        else:
            print("유효한 최적 가중치를 찾지 못했습니다.")
