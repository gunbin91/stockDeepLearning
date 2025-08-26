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

    for date in tqdm(daily_dates, desc="상세 백테스팅 중"):
        daily_trades = []
        for ticker in list(portfolio.keys()):
            stock_info = portfolio[ticker]
            is_holding_period_expired = (date - stock_info['buy_date']).days >= 15

            if (date, ticker) in data.index:
                current_price = data.loc[(date, ticker), '종가']
                
                sell_condition_price = (current_price >= stock_info['buy_price'] * 1.05) or \
                                       (current_price <= stock_info['buy_price'] * 0.97)

                if sell_condition_price or is_holding_period_expired:
                    buy_amount = stock_info['buy_price'] * stock_info['shares']
                    sell_value = current_price * stock_info['shares']
                    profit = sell_value - buy_amount
                    cash += sell_value
                    
                    log_entry = {
                        'type': 'sell', 'sell_date': date, 'ticker': ticker, 
                        'sell_price': current_price, 'return': (current_price / stock_info['buy_price']) - 1,
                        'buy_date': stock_info['buy_date'], 'buy_price': stock_info['buy_price'],
                        'buy_market_cap': stock_info.get('buy_market_cap'),
                        'buy_amount': buy_amount,
                        'profit': profit
                    }
                    log_entry.update(stock_info['buy_scores'])
                    daily_trades.append(log_entry)
                    
                    del portfolio[ticker]

        investment_per_stock = cash / top_n if top_n > 0 else 0
        if date in data.index.get_level_values('date'):
            daily_data = data.loc[date]
            daily_data_tradable = daily_data[daily_data['거래량'] > 0]
            buy_candidates = daily_data_tradable[~daily_data_tradable.index.get_level_values('종목코드').isin(portfolio.keys())].nlargest(top_n, 'final_score')
            for ticker, row in buy_candidates.iterrows():
                if cash >= investment_per_stock and investment_per_stock > 0:
                    buy_price = row['종가']
                    shares = investment_per_stock // buy_price
                    if shares > 0:
                        cash -= buy_price * shares
                        
                        buy_scores = {col: row.get(col) for col in score_cols_to_log if col in row}
                        portfolio[ticker] = {
                            'buy_date': date, 
                            'buy_price': buy_price, 
                            'shares': shares, 
                            'buy_scores': buy_scores,
                            'buy_market_cap': row.get('시가총액')
                        }
                        
        current_portfolio_value = sum(data.loc[(date, ticker), '종가'] * info['shares'] for ticker, info in portfolio.items() if (date, ticker) in data.index)
        total_asset = cash + current_portfolio_value
        portfolio_history.append(total_asset)
        
        for entry in daily_trades:
            entry['total_asset'] = total_asset
            trade_log.append(entry)
            
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
        
    final_asset = portfolio_ts.iloc[-1] if not portfolio_ts.empty else initial_capital
        
    return {"portfolio_history": portfolio_ts, "total_return": total_return, "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio, "mdd": mdd, "trade_log": trade_log_df, "win_rate": win_rate,
            "initial_capital": initial_capital, "final_asset": final_asset}

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
        
        sell_log['buy_amount_str'] = sell_log['buy_amount'].apply(lambda x: f"{x:,.0f}원")
        sell_log['profit_str'] = sell_log['profit'].apply(lambda x: f"{x:,.0f}원")
        sell_log['total_asset_str'] = sell_log['total_asset'].apply(lambda x: f"{x:,.0f}원")

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
        
        profit_numeric = results['trade_log']['profit']
        profit_colors = ['rgba(255, 220, 220, 0.7)' if p > 0 else 'rgba(220, 220, 255, 0.7)' if p < 0 else 'white' for p in profit_numeric]
        return_colors = ['rgba(255, 220, 220, 0.7)' if r > 0 else 'rgba(220, 220, 255, 0.7)' if r < 0 else 'white' for r in results['trade_log']['return']]
        
        rename_map = {
            'buy_date_str': '매수일', 'sell_date_str': '매도일', 'holding_period': '보유기간',
            '종목명': '종목명', 'buy_market_cap_str': '매수시점 시총',
            'buy_price': '매수가', 'sell_price': '매도가', 'buy_amount_str': '매수금액', 'profit_str': '실현손익',
            'return_str': '수익률', 'total_asset_str': '총자산', 'final_score': '최종점수',
            'ml_pred_proba': '상승확률', 'value_score': '가치', 'quality_score': '퀄리티',
            'momentum_score': '모멘텀', 'supply_score': '수급', 'volatility_score': '변동성'
        }
        display_columns = list(rename_map.keys())
        sell_log = sell_log[[col for col in display_columns if col in sell_log.columns]].rename(columns=rename_map)
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        row_heights=[0.45, 0.2, 0.35],
        specs=[[{"type": "scatter"}], [{"type": "table"}], [{"type": "table"}]]
    )

    fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, name='포트폴리오', line=dict(color='royalblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=kospi_cumulative.index, y=kospi_cumulative, name='KOSPI', line=dict(color='grey', width=1, dash='dash')), row=1, col=1)

    metrics_df = pd.DataFrame({
        '지표': ['초기 자본', '최종 자산', '총수익률', '연환산 수익률', '최대 낙폭 (MDD)', '샤프 지수', '승률'],
        '값': [
            f"{results.get('initial_capital', 0):,.0f}원",
            f"{results.get('final_asset', 0):,.0f}원",
            f"{results['total_return']:.2%}", 
            f"{results['annual_return']:.2%}", 
            f"{results['mdd']:.2%}", 
            f"{results['sharpe_ratio']:.2f}", 
            f"{results.get('win_rate', 0.0):.2%}"
        ]
    })
    fig.add_trace(go.Table(
        header=dict(values=list(metrics_df.columns), fill_color='paleturquoise', align='left', font=dict(size=14)),
        cells=dict(values=[metrics_df.지표, metrics_df.값], fill_color='lavender', align=['left', 'right'], font=dict(size=14))
    ), row=2, col=1)
    
    if not sell_log.empty:
        col_widths = [1.5, 1.5, 0.8, 1.8, 1.2, 1.2, 1.2, 1.5, 1.2, 1, 1.5, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.8]
        final_columns = list(sell_log.columns)
        col_widths = col_widths[:len(final_columns)]

        cell_colors = []
        for col_name in final_columns:
            if col_name == '실현손익':
                cell_colors.append(profit_colors)
            elif col_name == '수익률':
                cell_colors.append(return_colors)
            else:
                cell_colors.append(['white'] * len(sell_log))

        fig.add_trace(go.Table(
            header=dict(values=final_columns, fill_color='lightskyblue', align='left', font=dict(size=12)),
            columnwidth=col_widths,
            cells=dict(
                values=[sell_log[k].tolist() for k in final_columns],
                fill_color=cell_colors,
                align=['left', 'left', 'center', 'left'] + ['right'] * (len(final_columns) - 4),
                font=dict(size=11),
                height=25
            )
        ), row=3, col=1)

    fig.update_layout(
        title_text='<b>백테스팅 성과 분석 리포트</b>', height=1600, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50)
    )
    fig.update_yaxes(title_text='누적 수익률', row=1, col=1)
    
    fig.add_annotation(text="<b>주요 성과 지표</b>", xref="paper", yref="paper", x=0.0, y=0.54, showarrow=False, font=dict(size=16))
    if not sell_log.empty:
        fig.add_annotation(text="<b>상세 매매 기록 (매도 완료 기준)</b>", xref="paper", yref="paper", x=0.0, y=0.34, showarrow=False, font=dict(size=16))
    
    fig.write_html(output_path)

def run_final_backtest(initial_capital):
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
    
    print(f"\n3. 최종 백테스팅 시뮬레이션 실행 중... (초기 자본: {initial_capital:,.0f}원)")
    backtest_results = run_detailed_backtest(test_data, optimal_weights, initial_capital=initial_capital, top_n=TOP_N_STOCKS)
    
    print("\n4. HTML 리포트 생성 중...")
    create_html_report(backtest_results, output_path='backtest_report.html')
    print(f"\n✅ 백테스팅 완료. `backtest_report.html` 파일이 생성되었습니다.")

if __name__ == '__main__':
    while True:
        try:
            capital_input = input("초기 투자 자본금을 입력하세요 (기본값: 1,000,000,000) -> ")
            if not capital_input:
                initial_capital = 1_000_000_000
                break
            initial_capital = int(capital_input)
            if initial_capital > 0:
                break
            else:
                print("0보다 큰 값을 입력해야 합니다.")
        except ValueError:
            print("유효한 숫자를 입력하세요.")

    run_final_backtest(initial_capital)