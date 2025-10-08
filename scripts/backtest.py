"""
백테스팅 실행 스크립트
====================

이 파일은 주식 투자 전략의 성과를 검증하기 위한 백테스팅을 수행합니다.
과거 데이터를 사용하여 투자 전략의 수익률과 위험을 분석합니다.

주요 기능:
- 과거 데이터 기반 전략 검증
- 수익률 및 위험 지표 계산
- 거래 비용 및 세금 고려
- 시각화된 결과 보고서 생성
"""

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
import traceback
import threading
import time
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

# 증권거래세율 (0.15%)
SECURITIES_TRANSACTION_TAX_RATE = 0.15

# stdout/stderr를 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 내부 모듈 임포트
import ensemble
import data_processor
from logger import (log_info, log_error, log_critical, log_warning, shutdown_logger,
                   start_analysis_report, log_data_collection_status, log_processing_status, 
                   log_final_results, log_performance_info, log_saved_files, complete_analysis_report,
                   log_progress)
from path_manager import path_manager

# --- 설정 변수 (통일된 경로 사용) ---
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
WEIGHTS_FILE = str(path_manager.get_weights_path())
MODEL_FILE = str(path_manager.get_model_path())
REPORT_FILE = str(path_manager.get_backtest_report_path())
TOP_N_STOCKS = 5

def run_detailed_backtest(data, weights, initial_capital, top_n, max_hold_period, take_profit_pct, stop_loss_pct, buy_universe_rank, transaction_fee_rate):
    """상세 백테스팅 실행 - 강화된 에러 처리"""
    try:
        log_info("백테스팅 시작", context={
            "initial_capital": initial_capital,
            "top_n": top_n,
            "max_hold_period": max_hold_period,
            "data_rows": len(data)
        })
        
        final_df_list = []
        data_reset = data.reset_index()
        
        # 일별 최종 점수 계산 (에러 처리 강화)
        total_dates = len(data_reset.groupby('date'))
        for i, (date, daily_data) in enumerate(tqdm(data_reset.groupby('date'), desc="일별 최종 점수 계산", total=total_dates)):
            # 진행률 로그 메시지 처리
            if i % 10 == 0 or i == total_dates - 1:  # 10개마다 또는 마지막에 로그
                log_progress("일별 최종 점수 계산", i + 1, total_dates)
            try:
                temp_df = ensemble.calculate_final_score(daily_data.copy())
                temp_df['date'] = date
                final_df_list.append(temp_df)
            except Exception as e:
                log_error(f"일별 점수 계산 중 에러 발생", exception=e, context={
                    "date": str(date),
                    "daily_data_rows": len(daily_data)
                })
                # 에러가 발생한 날짜는 건너뛰고 계속 진행
                continue
        
        if not final_df_list:
            raise ValueError("계산된 점수 데이터가 없습니다")
            
        final_scores_df = pd.concat(final_df_list).set_index(['date', '종목코드'])
        data = data.merge(final_scores_df[['final_score']], left_index=True, right_index=True, how='left')
        data.dropna(subset=['final_score'], inplace=True)
        
        log_info("최종 점수 계산 완료", context={
            "final_scores_count": len(final_scores_df),
            "merged_data_rows": len(data)
        })
        
    except Exception as e:
        log_critical("백테스팅 초기화 중 치명적 에러", exception=e)
        raise
    cash = initial_capital
    portfolio = {}
    portfolio_history = []
    trade_log = []
    daily_dates = data.index.get_level_values('date').unique().sort_values()
    
    # 로그: 백테스팅 날짜 범위 확인
    print(f"🔍 백테스팅 날짜 범위: {daily_dates.min()} ~ {daily_dates.max()}")
    print(f"🔍 총 백테스팅 날짜 수: {len(daily_dates)}개")
    
    score_cols_to_log = ['final_score', 'ml_pred_proba', 'volatility_score']

    take_profit_multiplier = 1 + (take_profit_pct / 100)
    stop_loss_multiplier = 1 - (stop_loss_pct / 100)

    total_dates = len(daily_dates)
    for i, date in enumerate(tqdm(daily_dates, desc="상세 백테스팅 중")):
        # 진행률 로그 메시지 처리
        if i % 50 == 0 or i == total_dates - 1:  # 50개마다 또는 마지막에 로그
            log_progress("상세 백테스팅", i + 1, total_dates)
        try:
            # 로그: 현재 처리 중인 날짜 (9월 17일 이후만)
            if date >= pd.to_datetime('2025-09-17'):
                print(f"🔍 백테스팅 처리 중: {date.strftime('%Y-%m-%d')}")
            daily_trades = []
            
            # 포트폴리오 매도 처리 (에러 처리 강화)
            for ticker in list(portfolio.keys()):
                try:
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
                except Exception as e:
                    log_error(f"포트폴리오 매도 처리 중 에러", exception=e, context={
                        "date": str(date),
                        "ticker": ticker,
                        "portfolio_keys": list(portfolio.keys())
                    })
                    # 에러가 발생한 종목은 포트폴리오에서 제거하지 않고 계속 진행
                    continue

            # 매수 처리 (에러 처리 강화)
            try:
                investment_per_stock = cash / top_n if top_n > 0 else 0
                if date in data.index.get_level_values('date'):
                    daily_data = data.loc[date]
                    daily_data_tradable = daily_data[daily_data['거래량'] > 0]
                    
                    # 로그: 9월 17일 이후 데이터 확인
                    if date >= pd.to_datetime('2025-09-17'):
                        print(f"🔍 {date.strftime('%Y-%m-%d')} 데이터: 전체 {len(daily_data)}개, 거래가능 {len(daily_data_tradable)}개")
                    
                    # 1. 전체 거래 가능 종목 중에서 '최종 점수' 기준으로 상위 buy_universe_rank에 드는 종목들만 매수 고려 대상이 됩니다.
                    overall_top_universe = daily_data_tradable.nlargest(buy_universe_rank, 'final_score')
                    
                    # 2. 이렇게 1차로 걸러진 종목들 중에서 현재 보유하고 있는 종목들을 제외합니다.
                    # 3. 남은 종목들 중에서 '최종 점수'가 높은 순서대로 top_n (매수 종목 수)개만큼 매수합니다.
                    #    만약 남은 종목 수가 top_n보다 적다면, 그만큼만 매수하고 나머지는 현금으로 보유합니다.
                    buy_candidates = overall_top_universe[~overall_top_universe.index.get_level_values('종목코드').isin(portfolio.keys())].nlargest(top_n, 'final_score')

                    for ticker, row in buy_candidates.iterrows():
                        try:
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
                        except Exception as e:
                            log_error(f"개별 종목 매수 처리 중 에러", exception=e, context={
                                "date": str(date),
                                "ticker": ticker,
                                "cash": cash,
                                "investment_per_stock": investment_per_stock
                            })
                            continue
            except Exception as e:
                log_error(f"매수 처리 중 에러", exception=e, context={
                    "date": str(date),
                    "cash": cash,
                    "top_n": top_n
                })
                # 매수 처리 실패 시 해당 날짜는 건너뛰고 계속 진행
                        
            # 포트폴리오 가치 계산 (에러 처리 강화)
            try:
                current_portfolio_value = 0
                for ticker, info in portfolio.items():
                    try:
                        if (date, ticker) in data.index:
                            # 현재 날짜에 데이터가 있으면 현재 가격 사용
                            current_price = data.loc[(date, ticker), '종가']
                        else:
                            # 데이터가 없으면 이전 가격 사용 (최대 5일 전까지)
                            current_price = None
                            for days_back in range(1, 6):
                                prev_date = date - timedelta(days=days_back)
                                if (prev_date, ticker) in data.index:
                                    current_price = data.loc[(prev_date, ticker), '종가']
                                    break
                            
                            # 5일 내에 데이터가 없으면 매수가로 대체 (최악의 경우)
                            if current_price is None:
                                current_price = info['buy_price']
                                log_warning(f"종목 가격 데이터 없음", context={
                                    "ticker": ticker,
                                    "date": str(date),
                                    "buy_date": str(info['buy_date']),
                                    "buy_price": info['buy_price'],
                                    "holding_days": (date - info['buy_date']).days
                                })
                        
                        current_portfolio_value += current_price * info['shares']
                    except Exception as e:
                        log_error(f"개별 종목 가치 계산 중 에러", exception=e, context={
                            "date": str(date),
                            "ticker": ticker,
                            "portfolio_info": info
                        })
                        # 에러가 발생한 종목은 매수가로 대체
                        current_portfolio_value += info['buy_price'] * info['shares']
                        continue
                
                total_asset = cash + current_portfolio_value
                portfolio_history.append(total_asset)
                
                for entry in daily_trades:
                    entry['total_asset'] = total_asset
                    trade_log.append(entry)
                    
            except Exception as e:
                log_error(f"포트폴리오 가치 계산 중 에러", exception=e, context={
                    "date": str(date),
                    "cash": cash,
                    "portfolio_count": len(portfolio)
                })
                # 포트폴리오 가치 계산 실패 시 이전 가치 유지
                if portfolio_history:
                    portfolio_history.append(portfolio_history[-1])
                else:
                    portfolio_history.append(initial_capital)
        except Exception as e:
            log_critical(f"백테스팅 일별 처리 중 치명적 에러", exception=e, context={
                "date": str(date),
                "cash": cash,
                "portfolio_count": len(portfolio)
            })
            # 치명적 에러 발생 시 해당 날짜는 건너뛰고 계속 진행
            continue
            
        # 로그: 9월 17일 이후 포트폴리오 상태
        if date >= pd.to_datetime('2025-09-17'):
            print(f"🔍 {date.strftime('%Y-%m-%d')} 포트폴리오: 보유종목 {len(portfolio)}개, 현금 {cash:,.0f}원, 총자산 {total_asset:,.0f}원")
            
    portfolio_ts = pd.Series(portfolio_history, index=daily_dates)
    
    # 로그: 백테스팅 완료 후 최종 상태
    print(f"🔍 백테스팅 완료: 총 거래일 {len(portfolio_ts)}개, 최종 자산 {portfolio_ts.iloc[-1]:,.0f}원")
    print(f"🔍 마지막 거래일: {portfolio_ts.index[-1].strftime('%Y-%m-%d')}")
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
        # 백테스팅용 주식 목록 (캐시 없이 최신 데이터 사용)
        stock_list = data_processor.fetch_stock_list()
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
        score_cols = ['final_score', 'ml_pred_proba', 'volatility_score']
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
            'ml_pred_proba': '상승확률', 'volatility_score': '변동성'
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
    
    html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def run_final_backtest(initial_capital, max_hold_period, take_profit_pct, stop_loss_pct, top_n, buy_universe_rank, transaction_fee_rate):
    """최종 백테스팅 실행 - 강화된 에러 처리"""
    start_time = time.time()
    
    try:
        # 백테스팅 시작 보고서 (중복 제거)
        start_analysis_report(f"백테스팅 ({TEST_START_DATE} ~ {TEST_END_DATE})")
        
        log_info("💰 투자 설정")
        log_info(f"   └─ 초기 자본: {initial_capital:,}원")
        log_info(f"   └─ 매수 종목 수: {top_n}개")
        log_info(f"   └─ 최대 보유 기간: {max_hold_period}일")
        log_info(f"   └─ 익절 기준: +{take_profit_pct}%")
        log_info(f"   └─ 손절 기준: -{stop_loss_pct}%")
        log_info(f"   └─ 거래 수수료: {transaction_fee_rate}%")
        
        log_info("1. 최종 백테스트 시작...")
        
        # 가중치 파일 확인 및 로딩
        if not os.path.exists(WEIGHTS_FILE):
            error_msg = f"{WEIGHTS_FILE}을 찾을 수 없습니다. weight_optimizer.py를 먼저 실행해주세요."
            log_critical("가중치 파일 없음", context={"weights_file": WEIGHTS_FILE})
            raise FileNotFoundError(error_msg)
        
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                optimal_weights = json.load(f)
            log_info("최적 가중치 로딩 완료", context={"weights_file": WEIGHTS_FILE})
            log_info(f"  - 최적 가중치를 {WEIGHTS_FILE}에서 불러왔습니다.")
        except Exception as e:
            log_critical("가중치 파일 로딩 실패", exception=e, context={"weights_file": WEIGHTS_FILE})
            raise
        
        # 데이터 로딩 (강화된 에러 처리)
        try:
            backtest_start_date_with_warmup = (pd.to_datetime(TEST_START_DATE) - timedelta(days=400)).strftime('%Y-%m-%d')
            log_info("백테스팅 데이터 로딩 시작", context={
                "start_date": backtest_start_date_with_warmup,
                "end_date": TEST_END_DATE
            })
            
            test_data = data_processor.get_preprocessed_data(backtest_start_date_with_warmup, TEST_END_DATE)
            
            if test_data is None or test_data.empty:
                log_critical("백테스팅 데이터가 비어있습니다", context={
                    "start_date": backtest_start_date_with_warmup,
                    "end_date": TEST_END_DATE
                })
                raise ValueError("백테스팅 데이터가 비어있습니다")
            
            # 데이터 품질 검증 추가
            total_rows = len(test_data)
            valid_rows = len(test_data.dropna(subset=['종목코드', 'date']))
            data_quality = valid_rows / total_rows * 100 if total_rows > 0 else 0
            
            log_info(f"📊 백테스팅 데이터 품질: {valid_rows:,}/{total_rows:,}개 유효 행 ({data_quality:.1f}%)")
            
            if data_quality < 50:
                log_critical("백테스팅 데이터 품질이 너무 낮습니다", context={
                    "data_quality": data_quality,
                    "valid_rows": valid_rows,
                    "total_rows": total_rows
                })
                raise ValueError(f"백테스팅 데이터 품질이 너무 낮습니다: {data_quality:.1f}%")
            elif data_quality < 80:
                log_warning(f"⚠️ 백테스팅 데이터 품질이 낮습니다: {data_quality:.1f}% (권장: 90% 이상)")
                
            log_info("백테스팅 데이터 로딩 완료", context={
                "data_rows": len(test_data),
                "data_columns": list(test_data.columns)
            })
            
        except Exception as e:
            log_critical("백테스팅 데이터 로딩 실패", exception=e, context={
                "start_date": backtest_start_date_with_warmup,
                "end_date": TEST_END_DATE
            })
            raise
    
        # 모델 로딩 (강화된 에러 처리)
        try:
            print("  - 정식 모델 로딩 중...")
            log_info("ML 모델 로딩 시작", context={"model_file": MODEL_FILE})
            
            model_data = joblib.load(MODEL_FILE)
            model = model_data['model']
            features = model_data['features']
            scaler = model_data['scaler']
            
            log_info("ML 모델 로딩 완료", context={
                "model_type": type(model).__name__,
                "features_count": len(features),
                "scaler_type": type(scaler).__name__
            })
            
        except FileNotFoundError as e:
            error_msg = f"{MODEL_FILE}을 찾을 수 없습니다. train_model.py를 먼저 실행해주세요."
            log_critical("ML 모델 파일 없음", exception=e, context={"model_file": MODEL_FILE})
            raise FileNotFoundError(error_msg)
        except Exception as e:
            log_critical("ML 모델 로딩 실패", exception=e, context={"model_file": MODEL_FILE})
            raise
        
        # ML 예측 적용 (강화된 에러 처리)
        try:
            print("  - 테스트 데이터에 ML 예측 적용 중...")
            log_info("ML 예측 적용 시작", context={
                "data_rows": len(test_data),
                "features": features
            })
            
            test_data_for_pred = test_data[features].copy()
            test_data_for_pred.fillna(0, inplace=True)
            
            X_test_scaled = scaler.transform(test_data_for_pred)
            test_data['ml_pred_proba'] = model.predict_proba(X_test_scaled)[:, 1]
            
            log_info("ML 예측 적용 완료", context={
                "predictions_count": len(test_data['ml_pred_proba']),
                "prediction_range": f"{test_data['ml_pred_proba'].min():.3f} ~ {test_data['ml_pred_proba'].max():.3f}"
            })
            
        except Exception as e:
            log_critical("ML 예측 적용 실패", exception=e, context={
                "features": features,
                "data_shape": test_data.shape
            })
            raise
    
        # 데이터 전처리 (강화된 에러 처리)
        try:
            test_data = test_data[test_data['date'] >= pd.to_datetime(TEST_START_DATE)]
            
            # 로그: 백테스팅용 데이터 상태 확인
            log_info(f"🔍 백테스팅용 데이터: {len(test_data):,}개 행")
            log_info(f"🔍 데이터 날짜 범위: {test_data['date'].min()} ~ {test_data['date'].max()}")
            
            log_info("백테스팅 데이터 전처리 완료", context={
                "filtered_rows": len(test_data),
                "date_range": f"{test_data['date'].min()} ~ {test_data['date'].max()}"
            })
            
            # <<< 팩터 점수 계산 루프가 필요 없어짐 >>>
            log_info("  - 팩터 점수는 데이터 로딩 시 이미 계산되었습니다.")
            
            test_data.set_index(['date', '종목코드'], inplace=True)
            test_data.sort_index(inplace=True)
            
            # 로그: 인덱스 설정 후 상태 확인
            log_info(f"🔍 인덱스 설정 후: {len(test_data):,}개 행")
            log_info(f"🔍 인덱스 날짜 범위: {test_data.index.get_level_values('date').min()} ~ {test_data.index.get_level_values('date').max()}")
            log_info(f"🔍 인덱스 날짜 수: {len(test_data.index.get_level_values('date').unique())}개")
            
            log_info("데이터 인덱스 설정 완료", context={
                "indexed_rows": len(test_data),
                "unique_dates": len(test_data.index.get_level_values('date').unique()),
                "unique_tickers": len(test_data.index.get_level_values('종목코드').unique())
            })
            
        except Exception as e:
            log_critical("데이터 전처리 실패", exception=e, context={
                "data_shape": test_data.shape if 'test_data' in locals() else "unknown"
            })
            raise
        
        # 백테스팅 실행 (강화된 에러 처리)
        try:
            log_info(f"\n3. 최종 백테스팅 시뮬레이션 실행 중... (초기 자본: {initial_capital:,.0f}원)")
            log_info("백테스팅 시뮬레이션 시작", context={
                "initial_capital": initial_capital,
                "data_rows": len(test_data)
            })
            
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
            
            # 결과에 메타데이터 추가
            backtest_results['transaction_fee_rate'] = transaction_fee_rate
            backtest_results['initial_capital'] = initial_capital
            backtest_results['max_hold_period'] = max_hold_period
            backtest_results['take_profit_pct'] = take_profit_pct
            backtest_results['stop_loss_pct'] = stop_loss_pct
            backtest_results['top_n'] = top_n
            backtest_results['buy_universe_rank'] = buy_universe_rank
            backtest_results['securities_transaction_tax_rate'] = SECURITIES_TRANSACTION_TAX_RATE
            
            log_info("백테스팅 시뮬레이션 완료", context={
                "final_asset": backtest_results.get('final_asset', 0),
                "total_return": backtest_results.get('total_return', 0),
                "trade_count": len(backtest_results.get('trade_log', []))
            })
            
        except Exception as e:
            log_critical("백테스팅 시뮬레이션 실패", exception=e, context={
                "initial_capital": initial_capital,
                "data_rows": len(test_data)
            })
            raise
        
        # HTML 리포트 생성 (강화된 에러 처리)
        try:
            log_info("\n4. HTML 리포트 생성 중...")
            log_info("HTML 리포트 생성 시작", context={"report_file": REPORT_FILE})
            
            create_html_report(backtest_results, output_path=REPORT_FILE)
            
            log_info("HTML 리포트 생성 완료", context={"report_file": REPORT_FILE})
            log_info(f"\n✅ 백테스팅 완료. `{REPORT_FILE}` 파일이 생성되었습니다.")
            
        except Exception as e:
            log_critical("HTML 리포트 생성 실패", exception=e, context={"report_file": REPORT_FILE})
            # 리포트 생성 실패해도 백테스팅 결과는 반환
            log_info(f"\n⚠️ 백테스팅은 완료되었지만 리포트 생성에 실패했습니다: {e}")
        
        # 로거 종료
        try:
            shutdown_logger()
        except Exception:
            pass
            
    except Exception as e:
        log_critical("최종 백테스팅 실행 중 치명적 에러", exception=e)
        # 로거 종료
        try:
            shutdown_logger()
        except Exception:
            pass
        raise

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

