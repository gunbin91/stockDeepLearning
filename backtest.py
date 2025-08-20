import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import joblib
import finance_datareader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# 내부 모듈 임포트
from weight_optimizer import prepare_full_data
from scoring import calculate_factor_scores
import ensemble

# --- 설정 변수 ---
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
WEIGHTS_FILE = 'optimal_weights.json'
MODEL_FILE = 'stock_prediction_model_rf_upgraded.joblib'
TOP_N_STOCKS = 5

def run_detailed_backtest(data, weights, initial_capital=1_000_000_000, top_n=5):
    """상세한 백테스트를 수행하고, 수익률, MDD 등 다양한 성과 지표를 포함한 결과를 반환합니다."""
    
    # 1. 최종 점수 계산
    # calculate_final_score 함수를 사용하여 일관성 유지
    final_df_list = []
    # 그룹화 전에 인덱스 리셋
    data_reset = data.reset_index()
    for date, daily_data in tqdm(data_reset.groupby('date'), desc="일별 최종 점수 계산"):
        # ensemble 모듈에 최적 가중치를 임시로 적용하는 방식 대신, 가중치를 직접 전달
        temp_df = ensemble.calculate_final_score(daily_data.copy())
        temp_df['date'] = date
        final_df_list.append(temp_df)
    
    final_scores_df = pd.concat(final_df_list).set_index(['date', '종목코드'])
    data = data.merge(final_scores_df[['final_score']], left_index=True, right_index=True, how='left')
    data.dropna(subset=['final_score'], inplace=True)

    # 2. 시뮬레이션 변수 초기화
    cash = initial_capital
    portfolio = {}
    portfolio_history = []
    trade_log = []
    daily_dates = data.index.get_level_values('date').unique().sort_values()

    # 3. 일별 시뮬레이션
    for date in tqdm(daily_dates, desc="상세 백테스팅 중"):
        # 3.1. 매도 로직
        for ticker in list(portfolio.keys()):
            if (date, ticker) in data.index:
                stock_info = portfolio[ticker]
                current_price = data.loc[(date, ticker), '종가']
                
                sell_condition = (
                    (current_price >= stock_info['buy_price'] * 1.05) or # 익절
                    (current_price <= stock_info['buy_price'] * 0.97) or # 손절
                    ((date - stock_info['buy_date']).days >= 15) # 기간만료
                )
                
                if sell_condition:
                    sell_value = current_price * stock_info['shares']
                    cash += sell_value
                    trade_log.append({'type': 'sell', 'date': date, 'ticker': ticker, 'price': current_price, 'return': (current_price / stock_info['buy_price']) - 1})
                    del portfolio[ticker]

        # 3.2. 매수 로직
        investment_per_stock = cash / top_n if top_n > 0 else 0
        
        # date에 해당하는 데이터가 있는지 확인
        if date in data.index.get_level_values('date'):
            daily_data = data.loc[date]
            buy_candidates = daily_data[~daily_data.index.get_level_values('종목코드').isin(portfolio.keys())].nlargest(top_n, 'final_score')

            for ticker, row in buy_candidates.iterrows():
                if cash >= investment_per_stock and investment_per_stock > 0:
                    buy_price = row['종가']
                    shares = investment_per_stock // buy_price
                    if shares > 0:
                        cash -= buy_price * shares
                        portfolio[ticker] = {'buy_date': date, 'buy_price': buy_price, 'shares': shares}
                        trade_log.append({'type': 'buy', 'date': date, 'ticker': ticker, 'price': buy_price})

        # 3.3. 일일 자산 기록
        current_portfolio_value = sum(data.loc[(date, ticker), '종가'] * info['shares'] for ticker, info in portfolio.items() if (date, ticker) in data.index)
        total_asset = cash + current_portfolio_value
        portfolio_history.append(total_asset)

    # 4. 성과 지표 계산
    portfolio_ts = pd.Series(portfolio_history, index=daily_dates)
    daily_returns = portfolio_ts.pct_change().fillna(0)
    
    total_return = (portfolio_ts.iloc[-1] / portfolio_ts.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_ts)) - 1 if len(portfolio_ts) > 0 else 0
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    rolling_max = portfolio_ts.cummax()
    drawdown = (portfolio_ts - rolling_max) / rolling_max
    mdd = drawdown.min()

    return {
        "portfolio_history": portfolio_ts,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "mdd": mdd,
        "trade_log": pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    }


def create_html_report(results, output_path='backtest_report.html'):
    """백테스팅 결과를 사용하여 Plotly로 시각적인 HTML 리포트를 생성합니다."""
    
    if results["portfolio_history"].empty:
        print("백테스트 결과가 없어 리포트를 생성할 수 없습니다.")
        return

    # 1. 벤치마크(KOSPI) 데이터 가져오기
    kospi = fdr.DataReader('KS11', start=results["portfolio_history"].index.min(), end=results["portfolio_history"].index.max())
    
    # 2. 포트폴리오와 벤치마크 누적 수익률 계산
    portfolio_cumulative = (1 + results["portfolio_history"].pct_change().fillna(0)).cumprod()
    kospi_cumulative = (1 + kospi['Close'].pct_change().fillna(0)).cumprod()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3], specs=[[{"type": "scatter"}], [{"type": "table"}]])

    fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, name='포트폴리오', line=dict(color='royalblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=kospi_cumulative.index, y=kospi_cumulative, name='KOSPI', line=dict(color='grey', width=1, dash='dash')), row=1, col=1)

    metrics_df = pd.DataFrame({
        '지표': ['총수익률', '연환산 수익률', '최대 낙폭 (MDD)', '샤프 지수'],
        '값': [
            f"{results['total_return']:.2%}",
            f"{results['annual_return']:.2%}",
            f"{results['mdd']:.2%}",
            f"{results['sharpe_ratio']:.2f}"
        ]
    })
    fig.add_trace(go.Table(
        header=dict(values=list(metrics_df.columns), fill_color='paleturquoise', align='left'),
        cells=dict(values=[metrics_df.지표, metrics_df.값], fill_color='lavender', align='left')
    ), row=2, col=1)

    fig.update_layout(title_text='백테스팅 성과 분석 리포트', height=800, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title='누적 수익률')
    fig.update_xaxes(title_text='기간', row=1, col=1)
    fig.write_html(output_path)


def run_final_backtest():
    """저장된 최적 가중치를 사용하여 최종 백테스트를 수행하고 결과를 저장합니다."""
    print("1. 최종 백테스트 시작...")

    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"{WEIGHTS_FILE}을 찾을 수 없습니다. weight_optimizer.py를 먼저 실행해주세요.")
    
    with open(WEIGHTS_FILE, 'r') as f:
        optimal_weights = json.load(f)
    print(f"  - 최적 가중치를 {WEIGHTS_FILE}에서 불러왔습니다: {optimal_weights}")

    print("\n2. 테스트 데이터(2024년) 준비 중...")
    test_data = prepare_full_data(TEST_START_DATE, TEST_END_DATE)

    print("  - 정식 모델 로딩 중...")
    try:
        model_data = joblib.load(MODEL_FILE)
        model = model_data['model']
        features = model_data['features']
        # <<< 수정됨: 스케일러 불러오기
        scaler = model_data['scaler']
    except FileNotFoundError:
        raise FileNotFoundError(f"{MODEL_FILE}을 찾을 수 없습니다. train_model.py를 먼저 실행해주세요.")

    print("  - 테스트 데이터에 ML 예측 및 팩터 점수 추가 중...")
    test_data_for_pred = test_data[features].copy()
    test_data_for_pred.fillna(0, inplace=True) # 결측치 처리
    
    # <<< 수정됨: 예측 전에 스케일러 적용
    X_test_scaled = scaler.transform(test_data_for_pred)
    test_data['ml_pred_proba'] = model.predict_proba(X_test_scaled)[:, 1]

    # test_data를 (date, 종목코드) 멀티인덱스로 변환
    test_data.set_index(['date', '종목코드'], inplace=True)
    test_data.sort_index(inplace=True)
    print("  - 모든 데이터 준비 완료.")

    print("\n3. 최종 백테스팅 시뮬레이션 실행 중...")
    backtest_results = run_detailed_backtest(test_data, optimal_weights, top_n=TOP_N_STOCKS)
    
    print("\n4. HTML 리포트 생성 중...")
    create_html_report(backtest_results, output_path='backtest_report.html')
    print(f"\n✅ 백테스팅 완료. `backtest_report.html` 파일이 생성되었습니다.")

if __name__ == '__main__':
    run_final_backtest()