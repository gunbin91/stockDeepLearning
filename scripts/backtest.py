import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
import joblib
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# 증권거래세율 (0.15%)
SECURITIES_TRANSACTION_TAX_RATE = 0.15

# stdout/stderr를 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 내부 모듈 임포트
import ensemble
import data_cacher

# --- 설정 변수 ---
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'optimal_weights.json')
MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stock_prediction_model_rf_upgraded.joblib')
REPORT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backtest_report.html')
TOP_N_STOCKS = 5

def run_detailed_backtest(data, weights, initial_capital, top_n, max_hold_period, take_profit_pct, stop_loss_pct, buy_universe_rank, transaction_fee_rate):
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

    take_profit_multiplier = 1 + (take_profit_pct / 100)
    stop_loss_multiplier = 1 - (stop_loss_pct / 100)

    for date in tqdm(daily_dates, desc="상세 백테스팅 중"):
        daily_trades = []
        for ticker in list(portfolio.keys()):
            stock_info = portfolio[ticker]
            is_holding_period_expired = (date - stock_info['buy_date']).days >= max_hold_period

            if (date, ticker) in data.index:
                current_price = data.loc[(date, ticker), '종가']
                
                sell_condition_price = (current_price >= stock_info['buy_price'] * take_profit_multiplier) or \
                                       (current_price <= stock_info['buy_price'] * stop_loss_multiplier)

                if sell_condition_price or is_holding_period_expired:
                    buy_amount = stock_info['actual_buy_price'] * stock_info['shares']
                    
                    # 매도 시 수수료 및 세금 적용
                    actual_sell_price = current_price * (1 - (transaction_fee_rate + SECURITIES_TRANSACTION_TAX_RATE) / 100)
                    sell_value = actual_sell_price * stock_info['shares']
                    
                    profit = sell_value - buy_amount
                    cash += sell_value
                    
                    log_entry = {
                        'type': 'sell', 'sell_date': date, 'ticker': ticker, 
                        'sell_price': current_price, 'actual_sell_price': actual_sell_price, 'return': (actual_sell_price / stock_info['actual_buy_price']) - 1,
                        'buy_date': stock_info['buy_date'], 'buy_price': stock_info['buy_price'], # 수수료 적용 전 가격
                        'actual_buy_price': stock_info['actual_buy_price'], # 수수료 적용 후 가격
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
            
            # 1. 전체 거래 가능 종목 중에서 '최종 점수' 기준으로 상위 buy_universe_rank에 드는 종목들만 매수 고려 대상이 됩니다.
            overall_top_universe = daily_data_tradable.nlargest(buy_universe_rank, 'final_score')
            
            # 2. 이렇게 1차로 걸러진 종목들 중에서 현재 보유하고 있는 종목들을 제외합니다.
            # 3. 남은 종목들 중에서 '최종 점수'가 높은 순서대로 top_n (매수 종목 수)개만큼 매수합니다.
            #    만약 남은 종목 수가 top_n보다 적다면, 그만큼만 매수하고 나머지는 현금으로 보유합니다.
            buy_candidates = overall_top_universe[~overall_top_universe.index.get_level_values('종목코드').isin(portfolio.keys())].nlargest(top_n, 'final_score')

            for ticker, row in buy_candidates.iterrows():
                if cash >= investment_per_stock and investment_per_stock > 0:
                    buy_price = row['종가']
                    
                    # 매수 시 수수료 적용
                    actual_buy_price = buy_price * (1 + transaction_fee_rate / 100)
                    shares = investment_per_stock // actual_buy_price
                    
                    if shares > 0:
                        buy_amount_with_fee = actual_buy_price * shares
                        cash -= buy_amount_with_fee
                        
                        buy_scores = {col: row.get(col) for col in score_cols_to_log if col in row}
                        portfolio[ticker] = {
                            'buy_date': date, 
                            'buy_price': buy_price, # 수수료 적용 전 가격
                            'actual_buy_price': actual_buy_price, # 수수료 적용 후 가격
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


def create_html_report(results, output_path=REPORT_FILE):
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

    metrics_data = {
        '지표': [
            '초기 자본', '최종 자산', '총수익률', '연환산 수익률',
            '최대 낙폭 (MDD)', '샤프 지수', '승률', '거래 수수료율',
            '증권거래세율', '최대 보유 기간', '익절 목표 (%)', '손절 라인 (%)',
            '매수 종목 수 (N)', '매수 대상 범위'
        ],
        '값': [
            f"{results.get('initial_capital', 0):,.0f}원",
            f"{results.get('final_asset', 0):,.0f}원",
            f"{results['total_return']:.2%}", 
            f"{results['annual_return']:.2%}", 
            f"{results['mdd']:.2%}", 
            f"{results['sharpe_ratio']:.2f}", 
            f"{results.get('win_rate', 0.0):.2%}",
            f"{results.get('transaction_fee_rate', 0.0):.3f}%",
            f"{results.get('securities_transaction_tax_rate', 0.0):.2f}%",
            f"{results.get('max_hold_period', 0)}일",
            f"{results.get('take_profit_pct', 0.0):.2f}%",
            f"{results.get('stop_loss_pct', 0.0):.2f}%",
            f"{results.get('top_n', 0)}개",
            f"{results.get('buy_universe_rank', 0)}위"
        ]
    }
    metrics_df_temp = pd.DataFrame(metrics_data)

    # 2열씩 묶어서 4열 DataFrame 생성
    metrics_df = pd.DataFrame({
        '지표': metrics_df_temp['지표'].iloc[::2].reset_index(drop=True),
        '값': metrics_df_temp['값'].iloc[::2].reset_index(drop=True),
        '지표 ': metrics_df_temp['지표'].iloc[1::2].reset_index(drop=True),
        '값 ': metrics_df_temp['값'].iloc[1::2].reset_index(drop=True)
    })

    fig.add_trace(go.Table(
        header=dict(values=list(metrics_df.columns), fill_color='paleturquoise', align='left', font=dict(size=14)),
        cells=dict(values=[metrics_df['지표'], metrics_df['값'], metrics_df['지표 '], metrics_df['값 ']], fill_color='lavender', align=['left', 'right', 'left', 'right'], font=dict(size=14))
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
    
    # Streamlit과의 호환성을 위해 full_html=False, include_plotlyjs='cdn' 사용
    html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def run_final_backtest(initial_capital, max_hold_period, take_profit_pct, stop_loss_pct, top_n, buy_universe_rank, transaction_fee_rate):
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
    backtest_results = run_detailed_backtest(
        test_data, 
        optimal_weights, 
        initial_capital=initial_capital, 
        top_n=top_n,
        max_hold_period=max_hold_period,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        buy_universe_rank=buy_universe_rank,
        transaction_fee_rate=transaction_fee_rate
    )
    backtest_results['transaction_fee_rate'] = transaction_fee_rate # 결과에 수수료율 추가
    backtest_results['initial_capital'] = initial_capital
    backtest_results['max_hold_period'] = max_hold_period
    backtest_results['take_profit_pct'] = take_profit_pct
    backtest_results['stop_loss_pct'] = stop_loss_pct
    backtest_results['top_n'] = top_n
    backtest_results['buy_universe_rank'] = buy_universe_rank
    backtest_results['securities_transaction_tax_rate'] = SECURITIES_TRANSACTION_TAX_RATE # 증권거래세율 추가
    
    print("\n4. HTML 리포트 생성 중...")
    create_html_report(backtest_results, output_path=REPORT_FILE)
    print(f"\n✅ 백테스팅 완료. `{REPORT_FILE}` 파일이 생성되었습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock trading strategy backtest.')
    parser.add_argument('--capital', type=int, default=10000000, help='Initial capital')
    parser.add_argument('--max-hold', type=int, default=15, help='Maximum holding period in days')
    parser.add_argument('--take-profit', type=float, default=5.0, help='Take profit percentage')
    parser.add_argument('--stop-loss', type=float, default=3.0, help='Stop loss percentage')
    parser.add_argument('--top-n', type=int, default=5, help='Number of stocks to buy')
    parser.add_argument('--buy-universe', type=int, default=20, help='Rank universe to consider for buying')
    parser.add_argument('--fee', type=float, default=0.015, help='Transaction fee rate (e.g., 0.015 for 0.015%)')
    args = parser.parse_args()

    if args.capital <= 0:
        print("Error: Capital must be a positive number.")
    else:
        run_final_backtest(
            initial_capital=args.capital, 
            max_hold_period=args.max_hold, 
            take_profit_pct=args.take_profit, 
            stop_loss_pct=args.stop_loss, 
            top_n=args.top_n, 
            buy_universe_rank=args.buy_universe,
            transaction_fee_rate=args.fee
        )

