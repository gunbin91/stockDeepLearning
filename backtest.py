import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import joblib
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 내부 모듈 임포트
from scoring import calculate_factor_scores # 이 라인은 이제 없어도 되지만, 혹시 모르니 유지
import ensemble
import data_cacher

# --- 설정 변수 ---
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
WEIGHTS_FILE = 'optimal_weights.json'
MODEL_FILE = 'stock_prediction_model_rf_upgraded.joblib'
TOP_N_STOCKS = 5

# run_detailed_backtest, create_html_report 함수는 이전과 동일하게 유지

def run_detailed_backtest(data, weights, initial_capital=1_000_000_000, top_n=5):
    final_df_list = []
    data_reset = data.reset_index()
    for date, daily_data in tqdm(data_reset.groupby('date'), desc="일별 최종 점수 계산"):
        temp_df = ensemble.calculate_final_score(daily_data.copy())
        temp_df['date'] = date
        final_df_list.append(temp_df)
    final_scores_df = pd.concat(final_df_list).set_index(['date', '종목코드'])
    data = data.merge(final_scores_df[['final_score']], left_index=True, right_index=True, how='left')
    data.dropna(subset=['final_score'], inplace=True)
    cash = initial_capital
    portfolio = {}
    portfolio_history = []
    trade_log = []
    daily_dates = data.index.get_level_values('date').unique().sort_values()
    
    score_cols_to_log = ['final_score', 'ml_pred_proba', 'value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
    # --- 수정 시작: 로그에 기록할 거시경제 지표 컬럼 목록 추가 ---
    macro_cols_to_log = ['KOSPI_pct_1d', 'KOSPI_pct_5d', 'USDKRW_pct_1d', 'USDKRW_pct_5d', 'VIX_pct_1d', 'VIX_pct_5d']
    # --- 수정 종료 ---

    for date in tqdm(daily_dates, desc="상세 백테스팅 중"):
        for ticker in list(portfolio.keys()):
            if (date, ticker) in data.index:
                stock_info = portfolio[ticker]
                current_price = data.loc[(date, ticker), '종가']
                sell_condition = ((current_price >= stock_info['buy_price'] * 1.05) or (current_price <= stock_info['buy_price'] * 0.97) or ((date - stock_info['buy_date']).days >= 15))
                if sell_condition:
                    sell_value = current_price * stock_info['shares']
                    cash += sell_value
                    
                    log_entry = {
                        'type': 'sell', 'sell_date': date, 'ticker': ticker, 
                        'sell_price': current_price, 'return': (current_price / stock_info['buy_price']) - 1,
                        'buy_date': stock_info['buy_date'], 'buy_price': stock_info['buy_price'],
                        'buy_market_cap': stock_info.get('buy_market_cap')
                    }
                    log_entry.update(stock_info['buy_scores'])
                    # --- 수정 시작: 매도 로그에 저장해둔 매수 시점의 거시경제 지표 추가 ---
                    if 'buy_macro_conditions' in stock_info:
                        log_entry.update(stock_info['buy_macro_conditions'])
                    # --- 수정 종료 ---
                    trade_log.append(log_entry)
                    
                    del portfolio[ticker]

        investment_per_stock = cash / top_n if top_n > 0 else 0
        if date in data.index.get_level_values('date'):
            daily_data = data.loc[date]
            buy_candidates = daily_data[~daily_data.index.get_level_values('종목코드').isin(portfolio.keys())].nlargest(top_n, 'final_score')
            for ticker, row in buy_candidates.iterrows():
                if cash >= investment_per_stock and investment_per_stock > 0:
                    buy_price = row['종가']
                    shares = investment_per_stock // buy_price
                    if shares > 0:
                        cash -= buy_price * shares
                        
                        buy_scores = {col: row.get(col) for col in score_cols_to_log if col in row}
                        # --- 수정 시작: 매수 시점의 거시경제 지표 저장 ---
                        buy_macro_conditions = {col: row.get(col) for col in macro_cols_to_log if col in row}
                        portfolio[ticker] = {
                            'buy_date': date, 
                            'buy_price': buy_price, 
                            'shares': shares, 
                            'buy_scores': buy_scores,
                            'buy_macro_conditions': buy_macro_conditions, # 추가
                            'buy_market_cap': row.get('시가총액')
                        }
                        # --- 수정 종료 ---
                        
        current_portfolio_value = sum(data.loc[(date, ticker), '종가'] * info['shares'] for ticker, info in portfolio.items() if (date, ticker) in data.index)
        total_asset = cash + current_portfolio_value
        portfolio_history.append(total_asset)
        
    portfolio_ts = pd.Series(portfolio_history, index=daily_dates)
    daily_returns = portfolio_ts.pct_change().fillna(0)
    total_return = (portfolio_ts.iloc[-1] / portfolio_ts.iloc[0]) - 1 if len(portfolio_ts) > 1 else 0
    annual_return = (1 + total_return) ** (252 / len(portfolio_ts)) - 1 if len(portfolio_ts) > 0 else 0
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    rolling_max = portfolio_ts.cummax()
    drawdown = (portfolio_ts - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    trade_log_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    win_rate = 0.0
    if not trade_log_df.empty and len(trade_log_df) > 0:
        winning_trades = len(trade_log_df[trade_log_df['return'] > 0])
        total_trades = len(trade_log_df)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
    return {"portfolio_history": portfolio_ts, "total_return": total_return, "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio, "mdd": mdd, "trade_log": trade_log_df, "win_rate": win_rate}

def create_html_report(results, output_path='backtest_report.html'):
    if results["portfolio_history"].empty:
        print("백테스트 결과가 없어 리포트를 생성할 수 없습니다.")
        return

    kospi = fdr.DataReader('KS11', start=results["portfolio_history"].index.min(), end=results["portfolio_history"].index.max())
    portfolio_cumulative = (1 + results["portfolio_history"].pct_change().fillna(0)).cumprod()
    kospi_cumulative = (1 + kospi['Close'].pct_change().fillna(0)).cumprod()

    sell_log = results['trade_log'].copy()
    if not sell_log.empty:
        stock_list = data_cacher.fetch_stock_list()
        sell_log = pd.merge(sell_log, stock_list, left_on='ticker', right_on='종목코드', how='left')
        
        sell_log['holding_period'] = (sell_log['sell_date'] - sell_log['buy_date']).dt.days
        
        sell_log['buy_date_str'] = sell_log['buy_date'].dt.strftime('%Y-%m-%d')
        sell_log['sell_date_str'] = sell_log['sell_date'].dt.strftime('%Y-%m-%d')
        sell_log['buy_price'] = sell_log['buy_price'].apply(lambda x: f"{x:,.0f}원")
        sell_log['sell_price'] = sell_log['sell_price'].apply(lambda x: f"{x:,.0f}원")
        sell_log['return_str'] = sell_log['return'].apply(lambda x: f"{x:+.2%}")
        
        # --- 수정 시작: 거시경제 지표 데이터 포맷팅 ---
        macro_cols_to_format = {
            'KOSPI_pct_1d': 'KOSPI(1일)', 'KOSPI_pct_5d': 'KOSPI(5일)',
            'USDKRW_pct_1d': '환율(1일)', 'USDKRW_pct_5d': '환율(5일)',
            'VIX_pct_1d': 'VIX(1일)', 'VIX_pct_5d': 'VIX(5일)'
        }
        for col, new_name in macro_cols_to_format.items():
            if col in sell_log.columns:
                sell_log[col] = sell_log[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else 'N/A')
        # --- 수정 종료 ---
        
        if 'buy_market_cap' in sell_log.columns:
            sell_log['buy_market_cap_str'] = (sell_log['buy_market_cap'] / 1_0000_0000).apply(lambda x: f"{x:,.0f}억" if pd.notna(x) else 'N/A')
        else:
            sell_log['buy_market_cap_str'] = 'N/A'

        if 'ml_pred_proba' in sell_log.columns:
            sell_log['ml_pred_proba'] = (sell_log['ml_pred_proba'] * 100).round(2)
        score_cols = ['final_score', 'ml_pred_proba', 'value_score', 'quality_score', 'momentum_score', 'supply_score', 'volatility_score']
        for col in score_cols:
            if col in sell_log.columns:
                sell_log[col] = sell_log[col].round(2)
        
        return_colors = sell_log['return'].apply(lambda x: 'rgba(255, 220, 220, 0.7)' if x > 0 else ('rgba(220, 220, 225, 0.7)' if x < 0 else 'white'))
        
        rename_map = {
            'buy_date_str': '매수일', 'sell_date_str': '매도일', 'holding_period': '보유기간(일)',
            '종목명': '종목명', 'buy_market_cap_str': '매수시점 시총', 'buy_price': '매수가', 
            'sell_price': '매도가', 'return_str': '수익률', 'final_score': '최종점수(점)', 
            'ml_pred_proba': '상승확률(%)', 'value_score': '가치(점)', 'quality_score': '퀄리티(점)', 
            'momentum_score': '모멘텀(점)', 'supply_score': '수급(점)', 'volatility_score': '변동성(점)',
            # --- 수정 시작: 거시경제 지표 컬럼 이름 추가 ---
            'KOSPI_pct_1d': 'KOSPI(1일)', 'KOSPI_pct_5d': 'KOSPI(5일)',
            'USDKRW_pct_1d': '환율(1일)', 'USDKRW_pct_5d': '환율(5일)',
            'VIX_pct_1d': 'VIX(1일)', 'VIX_pct_5d': 'VIX(5일)'
            # --- 수정 종료 ---
        }
        display_columns = list(rename_map.keys())
        sell_log = sell_log[[col for col in display_columns if col in sell_log.columns]].rename(columns=rename_map)
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.1, 0.4],
        specs=[[{"type": "scatter"}], [{"type": "table"}], [{"type": "table"}]]
    )

    fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, name='포트폴리오', line=dict(color='royalblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=kospi_cumulative.index, y=kospi_cumulative, name='KOSPI', line=dict(color='grey', width=1, dash='dash')), row=1, col=1)

    metrics_df = pd.DataFrame({
        '지표': ['총수익률', '연환산 수익률', '최대 낙폭 (MDD)', '샤프 지수', '승률'],
        '값': [f"{results['total_return']:.2%}", f"{results['annual_return']:.2%}", 
               f"{results['mdd']:.2%}", f"{results['sharpe_ratio']:.2f}", f"{results.get('win_rate', 0.0):.2%}"]
    })
    fig.add_trace(go.Table(
        header=dict(values=list(metrics_df.columns), fill_color='paleturquoise', align='left', font=dict(size=14)),
        cells=dict(values=[metrics_df.지표, metrics_df.값], fill_color='lavender', align='left', font=dict(size=14), height=30)
    ), row=2, col=1)
    
    if not sell_log.empty:
        # --- 수정 시작: 테이블 셀 속성값 업데이트 (컬럼 추가에 따른 개수 조정) ---
        num_new_cols = len([col for col in macro_cols_to_format if col in results['trade_log'].columns]) # 실제 추가된 컬럼 수 계산
        fig.add_trace(go.Table(
            header=dict(values=list(sell_log.columns), fill_color='lightskyblue', align='left', font=dict(size=12)),
            cells=dict(
                values=[sell_log[k].tolist() for k in sell_log.columns],
                fill_color=[['white'] * len(sell_log)] * 7 + [return_colors.tolist()] + [['white'] * len(sell_log)] * (7 + num_new_cols),
                align=['left', 'left', 'center', 'left', 'right', 'right', 'right', 'right'] + ['right'] * (7 + num_new_cols),
                font=dict(size=11),
                height=25
            )
        ), row=3, col=1)
        # --- 수정 종료 ---

    fig.update_layout(
        title_text='<b>백테스팅 성과 분석 리포트</b>', height=1400, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    fig.update_yaxes(title_text='누적 수익률', row=1, col=1)
    fig.add_annotation(text="<b>주요 성과 지표</b>", xref="paper", yref="paper", x=0.0, y=0.49, showarrow=False, font=dict(size=16))
    if not sell_log.empty:
        fig.add_annotation(text="<b>상세 매매 기록 (매도 완료 기준)</b>", xref="paper", yref="paper", x=0.0, y=0.38, showarrow=False, font=dict(size=16))
    
    fig.write_html(output_path)


def run_final_backtest():
    print("1. 최종 백테스트 시작...")
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"{WEIGHTS_FILE}을 찾을 수 없습니다. weight_optimizer.py를 먼저 실행해주세요.")
    with open(WEIGHTS_FILE, 'r') as f:
        optimal_weights = json.load(f)
    print(f"  - 최적 가중치를 {WEIGHTS_FILE}에서 불러왔습니다.")
    
    backtest_start_date_with_warmup = (pd.to_datetime(TEST_START_DATE) - timedelta(days=400)).strftime('%Y-%m-%d')
    test_data = data_cacher.get_preprocessed_data(backtest_start_date_with_warmup, TEST_END_DATE)
    
    print("  - 정식 모델 로딩 중...")
    try:
        model_data = joblib.load(MODEL_FILE)
        model = model_data['model']
        features = model_data['features']
        scaler = model_data['scaler']
    except FileNotFoundError:
        raise FileNotFoundError(f"{MODEL_FILE}을 찾을 수 없습니다. train_model.py를 먼저 실행해주세요.")
    
    test_data_for_pred = test_data[features].copy()
    test_data_for_pred.fillna(0, inplace=True)
    
    print("  - 테스트 데이터에 ML 예측 적용 중...")
    X_test_scaled = scaler.transform(test_data_for_pred)
    test_data['ml_pred_proba'] = model.predict_proba(X_test_scaled)[:, 1]
    
    test_data = test_data[test_data['date'] >= pd.to_datetime(TEST_START_DATE)]
    
    # <<< 팩터 점수 계산 루프가 필요 없어짐 >>>
    # data_cacher에서 이미 모든 _score 컬럼들을 계산해왔기 때문입니다.
    print("  - 팩터 점수는 데이터 로딩 시 이미 계산되었습니다.")
    
    test_data.set_index(['date', '종목코드'], inplace=True)
    test_data.sort_index(inplace=True)
    
    print("\n3. 최종 백테스팅 시뮬레이션 실행 중...")
    backtest_results = run_detailed_backtest(test_data, optimal_weights, top_n=TOP_N_STOCKS)
    
    print("\n4. HTML 리포트 생성 중...")
    create_html_report(backtest_results, output_path='backtest_report.html')
    print(f"\n✅ 백테스팅 완료. `backtest_report.html` 파일이 생성되었습니다.")

if __name__ == '__main__':
    run_final_backtest()
