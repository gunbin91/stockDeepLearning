import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

# 내부 모듈 임포트
from weight_optimizer import prepare_full_data # 기존 로직 재사용
from scoring import calculate_factor_scores # 점수 계산 로직 추가
import joblib # 모델 로딩
import finance_datareader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 설정 변수 ---
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = '2024-12-31' # 연말까지 혹은 현재 날짜까지
WEIGHTS_FILE = 'optimal_weights.json'

# --- 함수 정의 ---

def run_detailed_backtest(data, weights, initial_capital=1_000_000_000, top_n=5):
    """상세한 백테스트를 수행하고, 수익률, MDD 등 다양한 성과 지표를 포함한 결과를 반환합니다."""
    
    # 1. 최종 점수 계산
    score_cols = [col for col in weights.keys() if col in data.columns]
    data['final_score'] = np.zeros(len(data))
    for col in score_cols:
        data['final_score'] += data[col].fillna(0) * weights[col]
    
    data['final_score'] = data.groupby(level=0)['final_score'].transform(
        lambda x: 100 * (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 50
    )

    # 2. 시뮬레이션 변수 초기화
    cash = initial_capital
    portfolio = {}
    portfolio_history = []
    trade_log = []
    daily_dates = data.index.get_level_values('date').unique().sort_values()

    # 3. 일별 시뮬레이션
    for date in tqdm(daily_dates, desc="상세 백테스팅 중"):
        # 3.1. 매도
        for ticker in list(portfolio.keys()):
            stock_info = portfolio[ticker]
            current_price = data.loc[(date, ticker), '종가']
            
            is_sell = False
            if current_price >= stock_info['buy_price'] * 1.05: is_sell = True # 익절
            elif current_price <= stock_info['buy_price'] * 0.97: is_sell = True # 손절
            elif (date - stock_info['buy_date']).days >= 15: is_sell = True # 기간만료
            
            if is_sell:
                sell_value = current_price * stock_info['shares']
                cash += sell_value
                trade_log.append({'type': 'sell', 'date': date, 'ticker': ticker, 'price': current_price, 'return': (current_price / stock_info['buy_price']) - 1})
                del portfolio[ticker]

        # 3.2. 매수
        investment_per_stock = cash / top_n
        buy_candidates = data.loc[date][~data.loc[date].index.get_level_values('종목코드').isin(portfolio.keys())].nlargest(top_n, 'final_score')

        for ticker, row in buy_candidates.iterrows():
            if cash >= investment_per_stock:
                buy_price = row['종가']
                shares = investment_per_stock // buy_price
                if shares > 0:
                    cash -= buy_price * shares
                    portfolio[ticker] = {'buy_date': date, 'buy_price': buy_price, 'shares': shares}
                    trade_log.append({'type': 'buy', 'date': date, 'ticker': ticker, 'price': buy_price})

        # 3.3. 일일 자산 기록
        current_portfolio_value = sum(data.loc[(date, ticker), '종가'] * stock_info['shares'] for ticker, stock_info in portfolio.items())
        total_asset = cash + current_portfolio_value
        portfolio_history.append(total_asset)

    # 4. 성과 지표 계산
    portfolio_ts = pd.Series(portfolio_history, index=daily_dates)
    daily_returns = portfolio_ts.pct_change().fillna(0)
    
    total_return = (portfolio_ts.iloc[-1] / portfolio_ts.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_ts)) - 1
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
        "trade_log": pd.DataFrame(trade_log)
    }


def run_final_backtest():
    """저장된 최적 가중치를 사용하여 최종 백테스트를 수행하고 결과를 저장합니다."""
    print("1. 최종 백테스트 시작...")

    # --- 1. 최적 가중치 불러오기 ---
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"{WEIGHTS_FILE}을 찾을 수 없습니다. weight_optimizer.py를 먼저 실행해주세요.")
    
    with open(WEIGHTS_FILE, 'r') as f:
        optimal_weights = json.load(f)
    print(f"  - 최적 가중치를 {WEIGHTS_FILE}에서 불러왔습니다.")
    print(f"  - 적용될 가중치: {optimal_weights}")

    # --- 2. 테스트 데이터 준비 (2024년) ---
    print("\n2. 테스트 데이터(2024년) 준비 중...")
    test_data = prepare_full_data(TEST_START_DATE, TEST_END_DATE)

    print("  - 정식 모델(stock_prediction_model_rf_upgraded.joblib) 로딩 중...")
    try:
        model_data = joblib.load('stock_prediction_model_rf_upgraded.joblib')
        model = model_data['model']
        features = model_data['features']
    except FileNotFoundError:
        raise FileNotFoundError("stock_prediction_model_rf_upgraded.joblib 모델 파일을 찾을 수 없습니다. train_model.py를 먼저 실행해주세요.")

    print("  - 테스트 데이터에 ML 예측 및 팩터 점수 추가 중...")
    test_data['ml_pred_proba'] = model.predict_proba(test_data[features])[:, 1]

    scored_data_list = []
    for date in tqdm(test_data.index.unique(), desc="팩터 점수 계산"):
        daily_data = test_data.loc[date].copy()
        daily_scored_data = calculate_factor_scores(daily_data.reset_index())
        daily_scored_data['date'] = date
        scored_data_list.append(daily_scored_data)

    if not scored_data_list:
        raise ValueError("점수 계산된 데이터가 없습니다.")

    all_scored_df = pd.concat(scored_data_list)
    all_scored_df.reset_index(drop=True, inplace=True)

    score_cols_to_merge = ['종목코드', 'date'] + [col for col in all_scored_df.columns if '_score' in col]
    test_data.reset_index(inplace=True)
    test_data = pd.merge(test_data, all_scored_df[score_cols_to_merge], on=['date', '종목코드'], how='left')
    test_data.set_index(['date', '종목코드'], inplace=True)
    test_data.sort_index(inplace=True)
    print("  - 모든 데이터 준비 완료.")

    # --- 3. 백테스팅 실행 ---
    print("\n3. 최종 백테스팅 시뮬레이션 실행 중...")
    backtest_results = run_detailed_backtest(test_data, optimal_weights)
    
    # --- 4. 결과 리포트 생성 ---
    print("\n4. HTML 리포트 생성 중...")
    create_html_report(backtest_results, output_path='backtest_report.html')
    print(f"\n✅ 백테스팅 완료. `backtest_report.html` 파일이 생성되었습니다.")

def create_html_report(results, output_path='backtest_report.html'):
    """백테스팅 결과를 사용하여 Plotly로 시각적인 HTML 리포트를 생성합니다."""
    
    # 1. 벤치마크(KOSPI) 데이터 가져오기
    kospi = fdr.DataReader('KS11', start=results["portfolio_history"].index.min(), end=results["portfolio_history"].index.max())
    
    # 2. 포트폴리오와 벤치마크 누적 수익률 계산
    portfolio_cumulative = (1 + results["portfolio_history"].pct_change().fillna(0)).cumprod()
    kospi_cumulative = (1 + kospi['Close'].pct_change().fillna(0)).cumprod()

    # 3. Plotly Figure 생성
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "scatter"}], [{"type": "table"}]]
    )

    # 3.1. 누적 수익률 그래프 추가
    fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, name='포트폴리오', line=dict(color='royalblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=kospi_cumulative.index, y=kospi_cumulative, name='KOSPI', line=dict(color='grey', width=1, dash='dash')), row=1, col=1)

    # 3.2. 주요 성과 지표 테이블 추가
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

    # 4. 레이아웃 업데이트
    fig.update_layout(
        title_text='백테스팅 성과 분석 리포트',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title='누적 수익률',
        xaxis_showticklabels=True, yaxis_showticklabels=True
    )
    fig.update_xaxes(title_text='기간', row=1, col=1)

    # 5. HTML 파일로 저장
    fig.write_html(output_path)


# --- 메인 실행부 ---
if __name__ == '__main__':
    run_final_backtest()
