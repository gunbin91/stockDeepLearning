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
TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
# 기본은 최근 1년 (웹 팝업 기본값과 동일)
TEST_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
# 250영업일(최대 롤링/룩백) 커버 목적의 안전 완충(캘린더 일수)
WARMUP_DAYS = 400
WEIGHTS_FILE = str(path_manager.get_weights_path())
MODEL_FILE = str(path_manager.get_model_path())
REPORT_FILE = str(path_manager.get_backtest_report_path())
TOP_N_STOCKS = 5
JSON_REPORT_FILE = str(path_manager.data_dir / 'backtest_report.json')

def _apply_model_prediction_to_backtest_df(
    df: pd.DataFrame,
    *,
    model_file: str,
    proba_col: str,
    model_label: str,
    require_feature_names_dataframe: bool = False
) -> pd.DataFrame:
    """
    백테스팅 데이터프레임에 모델 예측 확률 컬럼을 추가합니다.
    - 학습 시 저장된 features / scaler / imputation_values를 그대로 사용 (학습-추론 일관성)
    - 결측치는 0 채우기 대신 imputation_values로 대체 (누수 방지 + 일관성)
    """
    if df is None or df.empty:
        return df

    if not os.path.exists(model_file):
        log_warning(f"{model_label} 모델 파일이 없어 예측을 건너뜁니다", context={"model_file": model_file})
        df[proba_col] = np.nan
        return df

    try:
        model_data = joblib.load(model_file)
        model = model_data.get('model')
        features = model_data.get('features', [])
        scaler = model_data.get('scaler', None)
        imputation_values = model_data.get('imputation_values', None)
    except Exception as e:
        log_warning(f"{model_label} 모델 로딩 실패로 예측을 건너뜁니다", exception=e, context={"model_file": model_file})
        df[proba_col] = np.nan
        return df

    if model is None or not features:
        log_warning(f"{model_label} 모델/피처 정보가 비정상이라 예측을 건너뜁니다", context={"model_file": model_file})
        df[proba_col] = np.nan
        return df

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        log_warning(f"{model_label} 예측 불가: 필요한 피처 부족", context={"missing_features_count": len(missing_features)})
        df[proba_col] = np.nan
        return df

    try:
        X_pred = df[features].copy()
        if imputation_values is not None:
            X_pred.fillna(imputation_values, inplace=True)
        else:
            # 과거 모델/파일 호환: imputation_values가 없으면 최소한 0으로 대체
            X_pred.fillna(0, inplace=True)

        if scaler is not None:
            X_scaled = scaler.transform(X_pred)
            if require_feature_names_dataframe:
                # LightGBM feature-name warning 방지
                X_scaled = pd.DataFrame(X_scaled, columns=features)
        else:
            X_scaled = X_pred

        df[proba_col] = model.predict_proba(X_scaled)[:, 1]
        log_info(f"{model_label} 예측 적용 완료", context={
            "rows": len(df),
            "proba_col": proba_col,
            "min": float(np.nanmin(df[proba_col])) if df[proba_col].notna().any() else None,
            "max": float(np.nanmax(df[proba_col])) if df[proba_col].notna().any() else None
        })
        return df
    except Exception as e:
        log_warning(f"{model_label} 예측 실패", exception=e, context={"model_file": model_file})
        df[proba_col] = np.nan
        return df

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
        # 인덱스 정렬 보장 (pct_change 계산 및 일별 slicing 안정화)
        data.sort_index(inplace=True)

        # ============================================================
        # (안전장치) 당일 등락율(전일 종가 대비) 계산
        # - 상한가 근접(+29% 이상) 종목 매수 금지
        # - 상한가 근접(+29% 이상) 보유 종목은 해당일 매도 유예 (B안)
        # ============================================================
        if 'daily_change_pct' not in data.columns:
            try:
                data['daily_change_pct'] = data.groupby(level='종목코드')['종가'].pct_change() * 100
            except Exception as e:
                # 등락율 계산 실패 시에도 백테스트 전체를 멈추지 않도록 방어
                log_warning("당일 등락율 계산 실패 (필터 비활성화)", exception=e)
                data['daily_change_pct'] = np.nan
        
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
    
    score_cols_to_log = ['final_score', 'ml_pred_proba', 'lgb_pred_proba']

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
                        # (B안) 상한가 근접(+29% 이상) 날에는 익절/손절/기간만료 포함 '해당일 매도 금지'
                        try:
                            daily_change_pct = data.loc[(date, ticker), 'daily_change_pct'] if 'daily_change_pct' in data.columns else np.nan
                            if pd.notna(daily_change_pct) and float(daily_change_pct) >= 29.0:
                                continue
                        except Exception:
                            # 등락율 확인 실패 시 매도 로직은 기존대로 진행
                            pass
                        
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
                                'type': 'sell', 'trade_date': date, 'sell_date': date, 'ticker': ticker, 
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
                    # (안전장치) 상한가 근접(+29% 이상) 종목은 매수 후보에서 제외
                    if 'daily_change_pct' in daily_data_tradable.columns:
                        daily_data_tradable = daily_data_tradable[
                            daily_data_tradable['daily_change_pct'].isna() | (daily_data_tradable['daily_change_pct'] < 29.0)
                        ]
                    
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


def create_json_report(results, output_path=None):
    """JSON 리포트 생성 함수"""
    if output_path is None:
        output_path = str(path_manager.data_dir / 'backtest_report.json')
    
    if results["portfolio_history"].empty:
        print("백테스트 결과가 없어 리포트를 생성할 수 없습니다.")
        return

    # 포트폴리오 히스토리: 실제 총자산 값(원 단위) 저장
    portfolio_dates = [d.strftime('%Y-%m-%d') for d in results["portfolio_history"].index]
    portfolio_values = [float(v) for v in results["portfolio_history"].values]
    
    # KOSPI 데이터 가져오기 (비교용 누적 수익률)
    try:
        kospi = fdr.DataReader('KS11', start=results["portfolio_history"].index.min(), end=results["portfolio_history"].index.max())
        kospi_cumulative = (1 + kospi['Close'].pct_change().fillna(0)).cumprod()
        
        # KOSPI 초기값 기준으로 정규화 (포트폴리오와 비교 가능하도록)
        initial_capital = float(results.get('initial_capital', 0))
        kospi_values = [float(v * initial_capital) for v in kospi_cumulative.values]
        kospi_dates = [d.strftime('%Y-%m-%d') for d in kospi_cumulative.index]
    except Exception as e:
        log_warning(f"KOSPI 데이터 로딩 실패: {e}")
        kospi_dates = []
        kospi_values = []
    
    # 거래 로그 처리
    trade_log_all = results['trade_log'].copy()
    trade_log_records = []
    
    if not trade_log_all.empty:
        # trade_date 컬럼 보정 (없을 경우 sell_date 사용)
        if 'trade_date' not in trade_log_all.columns and 'sell_date' in trade_log_all.columns:
            trade_log_all['trade_date'] = trade_log_all['sell_date']

        # 백테스팅용 주식 목록 (캐시 없이 최신 데이터 사용)
        stock_list = data_processor.fetch_stock_list()
        if stock_list is not None and not stock_list.empty:
            trade_log_all = pd.merge(trade_log_all, stock_list, left_on='ticker', right_on='종목코드', how='left')
        else:
            # 주식 목록 수집 실패 시 종목명 컬럼만 추가 (NaN으로)
            log_warning("주식 목록 수집 실패: 종목명 없이 리포트 생성 (종목코드만 표시)")
            if '종목명' not in trade_log_all.columns:
                trade_log_all['종목명'] = None
        
        # 날짜순 정렬
        if 'trade_date' in trade_log_all.columns:
            trade_log_all = trade_log_all.sort_values('trade_date').reset_index(drop=True)
        
        # 누적 실현손익 계산
        cumulative_profit = 0.0
        if 'profit' in trade_log_all.columns:
            # 매도 거래만 누적 실현손익 계산
            trade_log_all['cumulative_profit'] = 0.0
            for idx, row in trade_log_all.iterrows():
                if row['type'] == 'sell' and pd.notna(row.get('profit')):
                    cumulative_profit += row['profit']
                    trade_log_all.at[idx, 'cumulative_profit'] = cumulative_profit
                else:
                    # 매수 거래는 이전 누적값 유지
                    trade_log_all.at[idx, 'cumulative_profit'] = cumulative_profit
        
        # 거래 로그를 레코드로 변환
        for _, row in trade_log_all.iterrows():
            record = {
                'type': row['type'],
                'trade_date': row['trade_date'].strftime('%Y-%m-%d') if pd.notna(row['trade_date']) else None,
                'ticker': row['ticker'],
                'stock_name': row.get('종목명', 'N/A'),
                'buy_date': row['buy_date'].strftime('%Y-%m-%d') if pd.notna(row['buy_date']) else None,
                'sell_date': row['sell_date'].strftime('%Y-%m-%d') if pd.notna(row['sell_date']) and 'sell_date' in row else None,
                'holding_period': int((row['sell_date'] - row['buy_date']).days) if 'sell_date' in row and pd.notna(row['sell_date']) and pd.notna(row['buy_date']) else None,
                'buy_price': float(row['buy_price']) if pd.notna(row['buy_price']) else None,
                'actual_buy_price': float(row['actual_buy_price']) if pd.notna(row['actual_buy_price']) else None,
                'sell_price': float(row['sell_price']) if 'sell_price' in row and pd.notna(row['sell_price']) else None,
                'actual_sell_price': float(row['actual_sell_price']) if 'actual_sell_price' in row and pd.notna(row['actual_sell_price']) else None,
                'shares': int(row['shares']) if 'shares' in row and pd.notna(row['shares']) else None,
                'buy_amount': float(row['buy_amount']) if pd.notna(row['buy_amount']) else None,
                'profit': float(row['profit']) if 'profit' in row and pd.notna(row['profit']) else None,
                'return': float(row['return']) if 'return' in row and pd.notna(row['return']) else None,
                'buy_market_cap': float(row['buy_market_cap']) if 'buy_market_cap' in row and pd.notna(row['buy_market_cap']) else None,
                'total_asset': float(row['total_asset']) if 'total_asset' in row and pd.notna(row['total_asset']) else None,
                'cumulative_profit': float(row['cumulative_profit']) if 'cumulative_profit' in row and pd.notna(row['cumulative_profit']) else None,
                'final_score': float(row['final_score']) if 'final_score' in row and pd.notna(row['final_score']) else None,
                'ml_pred_proba': float(row['ml_pred_proba']) if 'ml_pred_proba' in row and pd.notna(row['ml_pred_proba']) else None,
                # 현재 프로젝트 표준 키: lgb_pred_proba (flask/api/weights 및 test_data 컬럼명과 정합)
                'lgb_pred_proba': float(row['lgb_pred_proba']) if 'lgb_pred_proba' in row and pd.notna(row['lgb_pred_proba']) else None,
                'volatility_score': float(row['volatility_score']) if 'volatility_score' in row and pd.notna(row['volatility_score']) else None
            }
            trade_log_records.append(record)
    
    # 리포트 데이터 구성
    report_data = {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_period': {
                'start_date': results["portfolio_history"].index.min().strftime('%Y-%m-%d'),
                'end_date': results["portfolio_history"].index.max().strftime('%Y-%m-%d'),
                'total_days': len(results["portfolio_history"])
            }
        },
        'performance_metrics': {
            'initial_capital': float(results.get('initial_capital', 0)),
            'final_asset': float(results.get('final_asset', 0)),
            'total_return': float(results.get('total_return', 0)),
            'annual_return': float(results.get('annual_return', 0)),
            'sharpe_ratio': float(results.get('sharpe_ratio', 0)),
            'mdd': float(results.get('mdd', 0)),
            'win_rate': float(results.get('win_rate', 0))
        },
        'strategy_parameters': {
            'transaction_fee_rate': float(results.get('transaction_fee_rate', 0)),
            'securities_transaction_tax_rate': float(results.get('securities_transaction_tax_rate', SECURITIES_TRANSACTION_TAX_RATE)),
            'max_hold_period': int(results.get('max_hold_period', 0)),
            'take_profit_pct': float(results.get('take_profit_pct', 0)),
            'stop_loss_pct': float(results.get('stop_loss_pct', 0)),
            'top_n': int(results.get('top_n', 0)),
            'buy_universe_rank': int(results.get('buy_universe_rank', 0))
        },
        'portfolio_history': {
            'dates': portfolio_dates,
            'values': portfolio_values
        },
        'kospi_history': {
            'dates': kospi_dates,
            'values': kospi_values
        },
        'trade_log': trade_log_records
    }
    
    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    log_info(f"JSON 리포트 생성 완료: {output_path}")
    return report_data

def run_final_backtest(
    initial_capital,
    max_hold_period,
    take_profit_pct,
    stop_loss_pct,
    top_n,
    buy_universe_rank,
    transaction_fee_rate,
    start_date: str = None,
    end_date: str = None,
    warmup_days: int = None
):
    """최종 백테스팅 실행 - 강화된 에러 처리"""
    start_time = time.time()
    
    try:
        # 기간 기본값/검증
        start_date = start_date or TEST_START_DATE
        end_date = end_date or TEST_END_DATE
        warmup_days = WARMUP_DAYS if warmup_days is None else int(warmup_days)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt > end_dt:
            raise ValueError(f"백테스팅 기간 오류: start_date({start_date}) > end_date({end_date})")

        # 백테스팅 시작 보고서 (중복 제거)
        start_analysis_report(f"백테스팅 ({start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')})")
        
        log_info("💰 투자 설정")
        log_info(f"   └─ 초기 자본: {initial_capital:,}원")
        log_info(f"   └─ 매수 종목 수: {top_n}개")
        log_info(f"   └─ 최대 보유 기간: {max_hold_period}일")
        log_info(f"   └─ 익절 기준: +{take_profit_pct}%")
        log_info(f"   └─ 손절 기준: -{stop_loss_pct}%")
        log_info(f"   └─ 거래 수수료: {transaction_fee_rate}%")
        log_info(f"   └─ 백테스팅 기간: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} (웜업: {warmup_days}일)")
        
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
            backtest_start_date_with_warmup = (start_dt - timedelta(days=warmup_days)).strftime('%Y-%m-%d')
            log_info("백테스팅 데이터 로딩 시작", context={
                "start_date": backtest_start_date_with_warmup,
                "end_date": end_dt.strftime('%Y-%m-%d')
            })
            
            test_data = data_processor.get_preprocessed_data(backtest_start_date_with_warmup, end_dt.strftime('%Y-%m-%d'))
            
            if test_data is None or test_data.empty:
                log_critical("백테스팅 데이터가 비어있습니다", context={
                    "start_date": backtest_start_date_with_warmup,
                    "end_date": end_dt.strftime('%Y-%m-%d')
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
                "end_date": end_dt.strftime('%Y-%m-%d')
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
            rf_imputation_values = model_data.get('imputation_values', None)
            
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
        
        # ML 예측 적용 (가중치가 0이면 예측 로직 자체를 패스)
        if float(optimal_weights.get('ml_pred_proba', 0.5)) <= 0:
            log_info("ML 가중치가 0이라 RandomForest 예측을 건너뜁니다.")
            test_data['ml_pred_proba'] = np.nan
        else:
            try:
                print("  - 테스트 데이터에 ML 예측 적용 중...")
                log_info("ML 예측 적용 시작", context={
                    "data_rows": len(test_data),
                    "features": features
                })
                
                test_data_for_pred = test_data[features].copy()
                # ✅ 학습/추론 일관성: 모델에 저장된 결측치 대체값 사용 (없으면 0 fallback)
                if rf_imputation_values is not None:
                    test_data_for_pred.fillna(rf_imputation_values, inplace=True)
                else:
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

        # LightGBM 예측 적용 (가중치가 0이면 예측 로직 자체를 패스)
        if float(optimal_weights.get('lgb_pred_proba', 0.5)) <= 0:
            log_info("LGB 가중치가 0이라 LightGBM 예측을 건너뜁니다.")
            test_data['lgb_pred_proba'] = np.nan
        else:
            try:
                lgb_model_file = str(path_manager.get_lgb_model_path())
                test_data = _apply_model_prediction_to_backtest_df(
                    test_data,
                    model_file=lgb_model_file,
                    proba_col='lgb_pred_proba',
                    model_label='LightGBM',
                    require_feature_names_dataframe=True
                )
            except Exception as e:
                # 절대 백테스팅 전체를 멈추지 않도록 방어
                log_warning("LightGBM 예측 적용 중 예외(무시하고 계속 진행)", exception=e)
                if 'lgb_pred_proba' not in test_data.columns:
                    test_data['lgb_pred_proba'] = np.nan
    
        # 데이터 전처리 (강화된 에러 처리)
        try:
            test_data = test_data[test_data['date'] >= start_dt]
            
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
        
        # JSON 리포트 생성 (강화된 에러 처리)
        try:
            log_info("\n4. JSON 리포트 생성 중...")
            json_report_path = str(path_manager.data_dir / 'backtest_report.json')
            log_info("JSON 리포트 생성 시작", context={"report_file": json_report_path})
            
            create_json_report(backtest_results, output_path=json_report_path)
            
            log_info("JSON 리포트 생성 완료", context={"report_file": json_report_path})
            
        except Exception as e:
            log_critical("JSON 리포트 생성 실패", exception=e, context={"report_file": json_report_path if 'json_report_path' in locals() else "Unknown"})
            # 리포트 생성 실패해도 백테스팅 결과는 반환
            log_warning(f"\n⚠️ JSON 리포트 생성에 실패했습니다: {e}")
        
        log_info(f"\n✅ 백테스팅 완료. JSON 리포트 파일이 생성되었습니다.")
        
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
    parser.add_argument('--start-date', type=str, default=TEST_START_DATE, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=TEST_END_DATE, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--warmup-days', type=int, default=WARMUP_DAYS, help='Warmup days before start-date (calendar days)')
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
            transaction_fee_rate=args.fee,
            start_date=args.start_date,
            end_date=args.end_date,
            warmup_days=args.warmup_days
        )

