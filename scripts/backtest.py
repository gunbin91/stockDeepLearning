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
import ml_model
from logger import (log_info, log_error, log_critical, log_warning, shutdown_logger,
                   start_analysis_report, log_data_collection_status, log_processing_status, 
                   log_final_results, log_performance_info, log_saved_files, complete_analysis_report,
                   log_progress)
from path_manager import path_manager

# --- 설정 변수 (통일된 경로 사용) ---
# 기본값 설정 (파라미터로 제공되지 않을 경우 사용)
DEFAULT_TEST_START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
DEFAULT_TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
WEIGHTS_FILE = str(path_manager.get_weights_path())
# cuML 모델 경로 (우선 사용)
CUML_MODEL_FILE = str(path_manager.data_dir / 'cuml_ensemble_model.joblib')
# 기존 모델 경로 (fallback)
MODEL_FILE = str(path_manager.get_model_path())
JSON_REPORT_FILE = str(path_manager.data_dir / 'backtest_report.json')
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
                daily_data_copy = daily_data.copy()

                # 가중치 기반 스킵: LGBM 가중치가 0이면 예측을 수행하지 않음 (호환성: 컬럼은 NaN으로 유지)
                lgbm_weight = 0.0
                try:
                    if isinstance(weights, dict):
                        if 'lgbm_pred_proba' in weights:
                            lgbm_weight = float(weights.get('lgbm_pred_proba', 0.0))
                        elif 'lgb_pred_proba' in weights:
                            lgbm_weight = float(weights.get('lgb_pred_proba', 0.0))
                except Exception:
                    lgbm_weight = 0.0

                # LGBM 예측 추가 (lgbm_pred_proba 컬럼이 없는 경우)
                if 'lgbm_pred_proba' not in daily_data_copy.columns:
                    if lgbm_weight > 0:
                        try:
                            lgbm_predicted_df = ml_model.predict_with_lgbm_model(daily_data_copy)
                            if not lgbm_predicted_df.empty and 'lgbm_pred_proba' in lgbm_predicted_df.columns:
                                # LGBM 예측 결과 병합
                                daily_data_copy = pd.merge(
                                    daily_data_copy,
                                    lgbm_predicted_df[['종목코드', 'lgbm_pred_proba']],
                                    on='종목코드',
                                    how='left'
                                )
                        except Exception as e:
                            # LGBM 예측 실패 시 경고만 출력하고 계속 진행
                            log_warning(f"   ⚠️ [LGBM] 예측 실패 (날짜: {date}): {e}")
                            daily_data_copy['lgbm_pred_proba'] = np.nan
                    else:
                        # 스킵 시에도 컬럼은 유지
                        daily_data_copy['lgbm_pred_proba'] = np.nan
                
                temp_df = ensemble.calculate_final_score(daily_data_copy)
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
        
        # 병합할 컬럼 목록 (final_score는 필수, 나머지는 final_scores_df에 있으면 포함)
        merge_cols = ['final_score']
        optional_cols = ['ml_pred_proba', 'lgbm_pred_proba', 'volatility_score']
        for col in optional_cols:
            if col in final_scores_df.columns:
                merge_cols.append(col)
        
        # final_scores_df의 모든 관련 컬럼을 한 번에 병합
        # 이렇게 하면 RF와 LGBM 예측이 모두 포함됨
        data = data.merge(final_scores_df[merge_cols], left_index=True, right_index=True, how='left', suffixes=('', '_new'))
        
        # _new 접미사가 붙은 컬럼 처리 (final_scores_df의 값이 우선)
        for col in optional_cols:
            new_col = col + '_new'
            if new_col in data.columns:
                # final_scores_df의 값이 있으면 사용, 없으면 기존 data의 값 유지
                mask = data[new_col].notna()
                if mask.any():
                    data.loc[mask, col] = data.loc[mask, new_col]
                data.drop(columns=[new_col], inplace=True)
        
        data.dropna(subset=['final_score'], inplace=True)

        # ============================================================
        # 당일 등락율(전일 종가 대비) 계산
        # - 요구사항: 당일 등락율 +29% 이상 종목은
        #   (1) 당일 신규 매수 금지
        #   (2) 보유 중이면 당일 매도 조건이 충족되어도 매도 제외
        #
        # - 계산: (당일 종가 / 전일 종가 - 1) * 100
        # - 전일 종가가 없으면 NaN (필터 미적용)
        # ============================================================
        if 'daily_return_pct' not in data.columns:
            try:
                # 날짜 순 정렬이 중요
                data = data.sort_index()
                daily_return = None

                # 전날종가가 있으면 그 값을 우선 사용
                if '전날종가' in data.columns:
                    close = pd.to_numeric(data['종가'], errors='coerce')
                    prev = pd.to_numeric(data['전날종가'], errors='coerce')
                    daily_return = (close / prev - 1) * 100.0
                    daily_return = daily_return.where(prev > 0)

                # 전날종가가 없거나 계산 실패 시 종가 pct_change로 대체
                if daily_return is None:
                    closes = pd.to_numeric(data['종가'], errors='coerce')
                    daily_return = closes.groupby(level='종목코드').pct_change() * 100.0

                data['daily_return_pct'] = daily_return
            except Exception as e:
                # 계산 실패해도 백테스트는 계속 진행 (필터는 미적용)
                log_warning("⚠️ 당일 등락율 계산 실패: 필터를 적용하지 않고 진행합니다.", exception=e)
                data['daily_return_pct'] = np.nan
        
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
    trade_seq = 0
    daily_dates = data.index.get_level_values('date').unique().sort_values()
    
    # 로그: 백테스팅 날짜 범위 확인
    print(f"🔍 백테스팅 날짜 범위: {daily_dates.min()} ~ {daily_dates.max()}")
    print(f"🔍 총 백테스팅 날짜 수: {len(daily_dates)}개")
    
    score_cols_to_log = ['final_score', 'ml_pred_proba', 'lgbm_pred_proba', 'volatility_score']

    take_profit_multiplier = 1 + (take_profit_pct / 100)
    stop_loss_multiplier = 1 - (stop_loss_pct / 100)

    total_dates = len(daily_dates)
    for i, date in enumerate(tqdm(daily_dates, desc="상세 백테스팅 중")):
        # 진행률 로그 메시지 처리
        if i % 50 == 0 or i == total_dates - 1:  # 50개마다 또는 마지막에 로그
            log_progress("상세 백테스팅", i + 1, total_dates)
        try:
            # 디버깅: 매일 시작 시 현금 및 포트폴리오 상태 로깅 (샘플링)
            if i % 100 == 0 or i == 0 or i == total_dates - 1:
                portfolio_value_before = sum(info['buy_price'] * info['shares'] for info in portfolio.values())
                log_info(f"백테스팅 진행 상황", context={
                    "date": date.strftime('%Y-%m-%d'),
                    "cash": f"{cash:,.0f}원",
                    "portfolio_count": len(portfolio),
                    "portfolio_value_estimate": f"{portfolio_value_before:,.0f}원",
                    "total_asset_estimate": f"{cash + portfolio_value_before:,.0f}원"
                })
        except:
            pass  # 디버깅 로그 실패는 무시
        
        try:
            # 로그: 현재 처리 중인 날짜 (9월 17일 이후만)
            if date >= pd.to_datetime('2025-09-17'):
                print(f"🔍 백테스팅 처리 중: {date.strftime('%Y-%m-%d')}")
            daily_trades = []
            
            # 포트폴리오 매도 처리
            for ticker in list(portfolio.keys()):
                stock_info = portfolio[ticker]
                is_holding_period_expired = (date - stock_info['buy_date']).days >= max_hold_period

                # 현재 가격 조회 (데이터가 없으면 이전 가격 사용)
                current_price = None
                if (date, ticker) in data.index:
                    current_price = data.loc[(date, ticker), '종가']
                else:
                    # 이전 가격 사용 (최대 5일 전까지)
                    for days_back in range(1, 6):
                        prev_date = date - timedelta(days=days_back)
                        if (prev_date, ticker) in data.index:
                            current_price = data.loc[(prev_date, ticker), '종가']
                            break
                    
                    # 5일 내에 데이터가 없으면 매수가로 대체 (보수적 평가)
                    if current_price is None:
                        current_price = stock_info['buy_price']
                
                # ============================================================
                # 상한가(급등) 예외: 당일 등락율 +29% 이상이면 당일 매도 금지
                # - 데이터가 없는 날에는 등락율을 판단할 수 없으므로 예외 미적용
                # ============================================================
                try:
                    if (date, ticker) in data.index and 'daily_return_pct' in data.columns:
                        pct_val = data.loc[(date, ticker), 'daily_return_pct']
                        if pd.notna(pct_val) and float(pct_val) >= 29.0:
                            # 당일 매도 조건이 충족되어도 매도에서 제외 (보유 유지)
                            continue
                except Exception:
                    # 등락율 조회 실패 시에는 기존 로직 유지
                    pass

                # 매도 조건 확인 (current_price는 이미 위에서 None이 아닌 값으로 설정됨)
                sell_condition_price = (current_price >= stock_info['buy_price'] * take_profit_multiplier) or \
                                       (current_price <= stock_info['buy_price'] * stop_loss_multiplier)

                # 매도 실행 (익절/손절 또는 보유 기간 만료)
                if sell_condition_price or is_holding_period_expired:
                    buy_amount = stock_info['actual_buy_price'] * stock_info['shares']
                    
                    # 매도 시 수수료 및 세금 적용
                    actual_sell_price = current_price * (1 - (transaction_fee_rate + SECURITIES_TRANSACTION_TAX_RATE) / 100)
                    sell_value = actual_sell_price * stock_info['shares']
                    
                    profit = sell_value - buy_amount
                    cash += sell_value
                    
                    log_entry = {
                        'type': 'sell', 'trade_date': date, 'sell_date': date, 'ticker': ticker, 
                        'sell_price': current_price, 'actual_sell_price': actual_sell_price, 
                        'return': (actual_sell_price / stock_info['actual_buy_price']) - 1,
                        'buy_date': stock_info['buy_date'], 'buy_price': stock_info['buy_price'], # 수수료 적용 전 가격
                        'actual_buy_price': stock_info['actual_buy_price'], # 수수료 적용 후 가격
                        'buy_market_cap': stock_info.get('buy_market_cap'),
                        'buy_amount': buy_amount,
                        'profit': profit
                    }
                    log_entry.update(stock_info['buy_scores'])
                    daily_trades.append(log_entry)
                    
                    # 포트폴리오에서 제거
                    del portfolio[ticker]

            # 매수 처리
            investment_per_stock = cash / top_n if top_n > 0 else 0
            if date in data.index.get_level_values('date'):
                daily_data = data.loc[date]
                daily_data_tradable = daily_data[daily_data['거래량'] > 0]

                # ============================================================
                # 상한가(급등) 필터: 당일 등락율 +29% 이상 종목은 매수 금지
                # - 등락율이 NaN이면(전일 데이터 없음 등) 필터 미적용
                # ============================================================
                if 'daily_return_pct' in daily_data_tradable.columns:
                    try:
                        daily_data_tradable = daily_data_tradable[
                            (daily_data_tradable['daily_return_pct'].isna()) |
                            (daily_data_tradable['daily_return_pct'] < 29.0)
                        ]
                    except Exception:
                        # 필터 실패 시 기존 로직 유지
                        pass
                
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
                    if cash >= investment_per_stock and investment_per_stock > 0:
                        buy_price = row['종가']
                        
                        # 매수 시 수수료 적용
                        actual_buy_price = buy_price * (1 + transaction_fee_rate / 100)
                        shares = investment_per_stock // actual_buy_price
                        
                        if shares > 0:
                            buy_amount_with_fee = actual_buy_price * shares
                            cash -= buy_amount_with_fee
                            
                            # buy_scores 생성 (lgbm_pred_proba 포함)
                            buy_scores = {}
                            for col in score_cols_to_log:
                                if col in row.index:
                                    val = row[col]
                                    # NaN 체크
                                    if pd.notna(val):
                                        buy_scores[col] = float(val) if isinstance(val, (int, float, np.number)) else val
                            
                            portfolio[ticker] = {
                                'buy_date': date, 
                                'buy_price': buy_price, # 수수료 적용 전 가격
                                'actual_buy_price': actual_buy_price, # 수수료 적용 후 가격
                                'shares': shares, 
                                'buy_scores': buy_scores,
                                'buy_market_cap': row.get('시가총액') if '시가총액' in row.index else None
                            }
                            
                            # 매수 이력 추가
                            buy_amount = actual_buy_price * shares
                            buy_log_entry = {
                                'type': 'buy', 'trade_date': date, 'buy_date': date, 'ticker': ticker,
                                'buy_price': buy_price, 'actual_buy_price': actual_buy_price,
                                'buy_amount': buy_amount, 'shares': shares,
                                'buy_market_cap': row.get('시가총액') if '시가총액' in row.index else None
                            }
                            buy_log_entry.update(buy_scores)
                            daily_trades.append(buy_log_entry)
            
            # 포트폴리오 가치 계산
            current_portfolio_value = 0
            for ticker, info in portfolio.items():
                # 현재 가격 조회 (데이터가 없으면 이전 가격 사용)
                current_price = None
                if (date, ticker) in data.index:
                    current_price = data.loc[(date, ticker), '종가']
                    # NaN 체크
                    if pd.isna(current_price):
                        current_price = None
                else:
                    # 이전 가격 사용 (최대 5일 전까지)
                    for days_back in range(1, 6):
                        prev_date = date - timedelta(days=days_back)
                        if (prev_date, ticker) in data.index:
                            current_price = data.loc[(prev_date, ticker), '종가']
                            # NaN 체크
                            if not pd.isna(current_price):
                                break
                            else:
                                current_price = None
                    
                    # 5일 내에 데이터가 없으면 매수가로 대체 (보수적 평가)
                    if current_price is None or pd.isna(current_price):
                        current_price = info['buy_price']
                
                # 가격과 주식 수가 유효한 경우에만 계산
                if current_price is not None and not pd.isna(current_price) and info['shares'] > 0:
                    current_portfolio_value += current_price * info['shares']
            
            # 총자산 계산: 현금 + 포트폴리오 가치
            total_asset = cash + current_portfolio_value
            
            # 디버깅: 거래가 발생한 날짜에 상세 로깅
            if len(daily_trades) > 0:
                log_info(f"거래 발생일 상세 정보", context={
                    "date": date.strftime('%Y-%m-%d'),
                    "cash": f"{cash:,.0f}원",
                    "portfolio_value": f"{current_portfolio_value:,.0f}원",
                    "total_asset": f"{total_asset:,.0f}원",
                    "trade_count": len(daily_trades),
                    "portfolio_count": len(portfolio)
                })
            
            portfolio_history.append(total_asset)
            
            # 모든 거래 이력에 총자산 추가
            for entry in daily_trades:
                entry['total_asset'] = total_asset
                entry['trade_seq'] = trade_seq
                trade_seq += 1
                trade_log.append(entry)
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
    
    # 정합성 검증
    log_info("백테스팅 정합성 검증 시작")
    validation_errors = []
    
    # 1. 초기 자본 검증 (첫 거래일의 수수료를 고려)
    if len(portfolio_ts) > 0:
        first_asset = portfolio_ts.iloc[0]
        first_date = portfolio_ts.index[0]
        
        # 첫 거래일의 거래 로그 확인하여 수수료 계산
        first_date_trades = [t for t in trade_log if t.get('trade_date') == first_date]
        first_date_buy_trades = [t for t in first_date_trades if t.get('type') == 'buy']
        
        # 첫 거래일에 발생한 수수료 계산
        # buy_amount = actual_buy_price * shares (수수료 포함)
        # buy_price = 실제 주가 (수수료 미포함)
        # 수수료 = (actual_buy_price - buy_price) * shares = buy_amount - (buy_price * shares)
        estimated_fee = 0.0
        if first_date_buy_trades:
            for trade in first_date_buy_trades:
                buy_amount = trade.get('buy_amount', 0)
                buy_price = trade.get('buy_price', 0)
                shares = trade.get('shares', 0)
                if buy_amount > 0 and buy_price > 0 and shares > 0:
                    # 수수료 = buy_amount - (buy_price * shares)
                    trade_fee = buy_amount - (buy_price * shares)
                    estimated_fee += trade_fee
        
        # 예상 첫 자산 = 초기 자본 - 첫 거래일 수수료
        # (포트폴리오 가치는 매수가 기준이므로 수수료 차감 후 현금 + 포트폴리오 가치)
        expected_first_asset = initial_capital - estimated_fee
        
        # 허용 오차: 수수료 계산의 부동소수점 오차 및 반올림 오차 고려 (1% 또는 최소 100원)
        tolerance = max(estimated_fee * 0.01, 100.0) if estimated_fee > 0 else 100.0
        
        if abs(first_asset - expected_first_asset) > tolerance:
            validation_errors.append(
                f"초기 자본 불일치: 예상 {expected_first_asset:,.0f}원 (초기자본 {initial_capital:,.0f}원 - 수수료 {estimated_fee:,.0f}원), "
                f"실제 {first_asset:,.0f}원 (차이: {abs(first_asset - expected_first_asset):,.0f}원)"
            )
    
    # 2. 거래 로그 정합성 검증 (trade_log_df 생성 전에 임시로 생성)
    trade_log_df_temp = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    if not trade_log_df_temp.empty:
        # 매수/매도 거래 검증
        buy_trades = trade_log_df_temp[trade_log_df_temp['type'] == 'buy']
        sell_trades = trade_log_df_temp[trade_log_df_temp['type'] == 'sell']
        
        # 매수 금액 합계 검증
        if not buy_trades.empty and 'buy_amount' in buy_trades.columns:
            total_buy_amount = buy_trades['buy_amount'].sum()
            log_info(f"총 매수 금액: {total_buy_amount:,.0f}원")
        
        # 매도 금액 합계 검증
        if not sell_trades.empty and 'profit' in sell_trades.columns:
            total_profit = sell_trades['profit'].sum()
            log_info(f"총 실현 손익: {total_profit:,.0f}원")
        
        # 매수/매도 쌍 검증 (매수한 종목이 모두 매도되었는지)
        buy_tickers = set(buy_trades['ticker'].unique())
        sell_tickers = set(sell_trades['ticker'].unique())
        unmatched_buys = buy_tickers - sell_tickers
        if unmatched_buys:
            log_warning(f"매도되지 않은 매수 종목 {len(unmatched_buys)}개: {list(unmatched_buys)[:5]}...")
        
        # 최종 포트폴리오 상태 검증
        final_cash_estimate = initial_capital
        for _, trade in trade_log_df_temp.iterrows():
            if trade['type'] == 'buy':
                final_cash_estimate -= trade.get('buy_amount', 0)
            elif trade['type'] == 'sell':
                final_cash_estimate += trade.get('buy_amount', 0) + trade.get('profit', 0)
        
        log_info(f"거래 로그 기반 추정 현금: {final_cash_estimate:,.0f}원")
        log_info(f"실제 최종 자산: {portfolio_ts.iloc[-1]:,.0f}원")
    
    # 3. 포트폴리오 히스토리 연속성 검증
    if len(portfolio_ts) > 1:
        negative_values = portfolio_ts[portfolio_ts < 0]
        if len(negative_values) > 0:
            validation_errors.append(f"음수 자산 값 발견: {len(negative_values)}개")
        
        # 급격한 변화 검증 (일일 100% 이상 변화는 의심)
        pct_changes = portfolio_ts.pct_change().fillna(0)
        extreme_changes = pct_changes[abs(pct_changes) > 1.0]
        if len(extreme_changes) > 0:
            log_warning(f"급격한 자산 변화 발견: {len(extreme_changes)}일 (100% 이상 변화)")
            for date, change in extreme_changes.head(5).items():
                log_warning(f"  {date.strftime('%Y-%m-%d')}: {change:.2%} 변화")
    
    if validation_errors:
        log_error("정합성 검증 실패", context={"errors": validation_errors})
        for error in validation_errors:
            print(f"❌ {error}")
    else:
        log_info("정합성 검증 완료: 이상 없음")
        print("✅ 정합성 검증 완료: 이상 없음")
    
    daily_returns = portfolio_ts.pct_change().fillna(0)
    total_return = (portfolio_ts.iloc[-1] / portfolio_ts.iloc[0]) - 1 if len(portfolio_ts) > 1 else 0
    annual_return = (1 + total_return) ** (252 / len(portfolio_ts)) - 1 if len(portfolio_ts) > 0 else 0
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    rolling_max = portfolio_ts.cummax()
    drawdown = (portfolio_ts - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    trade_log_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    
    # 거래 이력을 날짜순으로 정렬
    if not trade_log_df.empty and 'trade_date' in trade_log_df.columns:
        trade_log_df = trade_log_df.sort_values('trade_date').reset_index(drop=True)
    
    win_rate = 0.0
    if not trade_log_df.empty and len(trade_log_df) > 0:
        # 매도 거래만으로 승률 계산
        sell_trades = trade_log_df[trade_log_df['type'] == 'sell']
        if not sell_trades.empty and 'return' in sell_trades.columns:
            winning_trades = len(sell_trades[sell_trades['return'] > 0])
            total_trades = len(sell_trades)
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
            stock_name = row.get('종목명', 'N/A')
            if pd.isna(stock_name):
                stock_name = None
            record = {
                'type': row['type'],
                'trade_date': row['trade_date'].strftime('%Y-%m-%d') if pd.notna(row['trade_date']) else None,
                'ticker': row['ticker'],
                'stock_name': stock_name,
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
                'lgbm_pred_proba': float(row['lgbm_pred_proba']) if 'lgbm_pred_proba' in row and pd.notna(row['lgbm_pred_proba']) else None,
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
    
    # JSON 안전 변환 (NaN/Infinity 제거)
    def _sanitize_json_value(value):
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return None
        if isinstance(value, dict):
            return {k: _sanitize_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize_json_value(v) for v in value]
        return value

    report_data = _sanitize_json_value(report_data)

    # JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    log_info(f"JSON 리포트 생성 완료: {output_path}")
    return report_data


def run_final_backtest(initial_capital, max_hold_period, take_profit_pct, stop_loss_pct, top_n, buy_universe_rank, transaction_fee_rate, start_date=None, end_date=None):
    """최종 백테스팅 실행 - 강화된 에러 처리
    
    Args:
        initial_capital: 초기 자본
        max_hold_period: 최대 보유 기간
        take_profit_pct: 익절 비율
        stop_loss_pct: 손절 비율
        top_n: 매수 종목 수
        buy_universe_rank: 매수 대상 범위
        transaction_fee_rate: 거래 수수료율
        start_date: 테스트 시작일 (YYYY-MM-DD 형식, None이면 기본값 사용)
        end_date: 테스트 종료일 (YYYY-MM-DD 형식, None이면 기본값 사용)
    """
    start_time = time.time()
    
    # 날짜 파라미터 기본값 설정
    if start_date is None:
        start_date = DEFAULT_TEST_START_DATE
    if end_date is None:
        end_date = DEFAULT_TEST_END_DATE
    
    try:
        # 백테스팅 시작 보고서 (중복 제거)
        start_analysis_report(f"백테스팅 ({start_date} ~ {end_date})")
        
        log_info("💰 투자 설정")
        log_info(f"   └─ 초기 자본: {initial_capital:,}원")
        log_info(f"   └─ 매수 종목 수: {top_n}개")
        log_info(f"   └─ 최대 보유 기간: {max_hold_period}일")
        log_info(f"   └─ 익절 기준: +{take_profit_pct}%")
        log_info(f"   └─ 손절 기준: -{stop_loss_pct}%")
        log_info(f"   └─ 거래 수수료: {transaction_fee_rate}%")
        log_info(f"   └─ 테스트 기간: {start_date} ~ {end_date}")
        
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

        # 가중치 기반 스킵 플래그
        def _w(key: str, default: float = 0.0) -> float:
            try:
                if isinstance(optimal_weights, dict):
                    if key in optimal_weights:
                        return float(optimal_weights.get(key, default))
                    # 키 호환
                    if key == 'lgbm_pred_proba' and 'lgb_pred_proba' in optimal_weights:
                        return float(optimal_weights.get('lgb_pred_proba', default))
            except Exception:
                pass
            return float(default)

        do_rf = _w('ml_pred_proba', 0.5) > 0
        do_lgbm = _w('lgbm_pred_proba', 0.5) > 0
        do_volatility = _w('volatility_score', 0.10) > 0
        log_info("⚙️ 가중치 기반 계산 스킵 설정", context={
            "volatility_score": _w('volatility_score', 0.10),
            "ml_pred_proba": _w('ml_pred_proba', 0.5),
            "lgbm_pred_proba": _w('lgbm_pred_proba', 0.5),
            "do_volatility": do_volatility,
            "do_rf": do_rf,
            "do_lgbm": do_lgbm
        })
        
        # 데이터 로딩 (강화된 에러 처리)
        # Warmup 기간 400일 유지
        try:
            backtest_start_date_with_warmup = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
            log_info("백테스팅 데이터 로딩 시작", context={
                "test_start_date": start_date,
                "test_end_date": end_date,
                "data_start_date": backtest_start_date_with_warmup,
                "data_end_date": end_date,
                "warmup_days": 400
            })
            log_info(f"   📅 테스트 기간: {start_date} ~ {end_date}")
            log_info(f"   📅 데이터 수집 기간: {backtest_start_date_with_warmup} ~ {end_date} (Warmup 400일 포함)")
            
            # volatility_score 가중치가 0이면, 일별 팩터 점수 계산(현재는 변동성 점수) 자체를 스킵
            test_data = data_processor.get_preprocessed_data(
                backtest_start_date_with_warmup,
                end_date,
                skip_factor_scores=(not do_volatility)
            )
            
            # 즉시 빈 데이터 검증 (초기 수집/병합 실패 조기 발견)
            if test_data is None or test_data.empty:
                log_critical("백테스팅 데이터가 비어있습니다", context={
                    "start_date": backtest_start_date_with_warmup,
                    "end_date": end_date,
                    "hint": "가격/재무 수집 실패 또는 날짜 구간에 데이터 부재"
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
                "end_date": end_date
            })
            raise
    
        # 모델 로딩 (강화된 에러 처리)
        try:
            print("  - 정식 모델 로딩 중...")
            
            # cuML 모델 파일 우선 확인, 없으면 기존 모델 파일 확인
            model_file_path = None
            is_cuml_model = False
            
            if os.path.exists(CUML_MODEL_FILE):
                model_file_path = CUML_MODEL_FILE
                is_cuml_model = True
                log_info("ML 모델 로딩 시작 (cuML 앙상블)", context={"model_file": model_file_path})
            elif os.path.exists(MODEL_FILE):
                model_file_path = MODEL_FILE
                is_cuml_model = False
                log_info("ML 모델 로딩 시작 (기존 모델)", context={"model_file": model_file_path})
            else:
                error_msg = f"모델 파일을 찾을 수 없습니다. (cuML: {CUML_MODEL_FILE}, 기존: {MODEL_FILE})"
                log_critical("ML 모델 파일 없음", context={"cuml_file": CUML_MODEL_FILE, "legacy_file": MODEL_FILE})
                raise FileNotFoundError(error_msg)
            
            model_data = joblib.load(model_file_path)
            
            # cuML 앙상블 모델인지 확인
            if is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'mini_batch_ensemble':
                # cuML 앙상블 모델 처리
                from ml_model_wrapper import EnsembleModelWrapper
                models = model_data['models']
                scaler = model_data['scaler']
                features = model_data['features']
                
                # 앙상블 모델 래퍼 생성
                model = EnsembleModelWrapper(models, scaler)
                
                log_info("ML 모델 로딩 완료 (cuML 앙상블)", context={
                    "model_type": f"cuML 앙상블 ({len(models)}개 모델)",
                    "features_count": len(features),
                    "scaler_type": type(scaler).__name__ if scaler else "None"
                })
            elif is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'single_model':
                # cuML 단일 모델 처리
                model = model_data['model']
                features = model_data['features']
                scaler = model_data['scaler']
                # imputation_values는 나중에 로드 (필요할 때)
                
                log_info("ML 모델 로딩 완료 (cuML 단일 모델)", context={
                    "model_type": "cuML 단일 모델",
                    "features_count": len(features),
                    "scaler_type": type(scaler).__name__ if scaler else "None"
                })
            else:
                # 기존 모델 구조 (sklearn) 또는 model_type이 없는 경우
                model = model_data['model']
                features = model_data['features']
                scaler = model_data['scaler']
                
                # cuML 모델인지 확인 (타입으로 판단)
                if 'cuml' in str(type(model)).lower() or hasattr(model, 'predict_proba') and 'cuml' in str(type(model).__module__).lower():
                    is_cuml_model = True
                    log_info("ML 모델 로딩 완료 (cuML 모델 - 타입 감지)", context={
                        "model_type": type(model).__name__,
                        "features_count": len(features),
                        "scaler_type": type(scaler).__name__ if scaler else "None"
                    })
                else:
                    is_cuml_model = False
                    log_info("ML 모델 로딩 완료 (sklearn 모델)", context={
                        "model_type": type(model).__name__,
                        "features_count": len(features),
                        "scaler_type": type(scaler).__name__ if scaler else "None"
                    })
            
        except FileNotFoundError as e:
            error_msg = f"모델 파일을 찾을 수 없습니다. train_gpu_main.py를 먼저 실행해주세요."
            log_critical("ML 모델 파일 없음", exception=e, context={"model_file": model_file_path if 'model_file_path' in locals() else "Unknown"})
            raise FileNotFoundError(error_msg)
        except Exception as e:
            log_critical("ML 모델 로딩 실패", exception=e, context={"model_file": model_file_path if 'model_file_path' in locals() else "Unknown"})
            raise
        
        # ML 예측 적용 (강화된 에러 처리) - 가중치 0이면 스킵
        try:
            if not do_rf:
                log_info("⏭️ RF(ml_pred_proba) 가중치가 0이라 예측을 건너뜁니다.")
                test_data['ml_pred_proba'] = np.nan
            else:
                print("  - 테스트 데이터에 ML 예측 적용 중...")
                log_info("ML 예측 적용 시작", context={
                    "data_rows": len(test_data),
                    "features": features
                })
            
            # 모델이 기대하는 피처만 선택 (데이터에 있는 피처만 사용)
            # 중요: 모델이 저장된 피처 순서를 정확히 따라야 함
            available_features = [f for f in features if f in test_data.columns]
            missing_features = [f for f in features if f not in test_data.columns]
            
            if missing_features:
                log_warning(f"   ⚠️ 필요한 피처 부족: {len(missing_features)}개 - {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
                if len(available_features) == 0:
                    log_error("   ❌ 사용 가능한 피처가 없습니다. 예측을 수행할 수 없습니다.")
                    raise ValueError("모델이 기대하는 피처가 데이터에 없습니다.")
                log_info(f"   ℹ️ 사용 가능한 피처 {len(available_features)}개로 예측 진행")
            
            # 모델의 피처 수 확인 (cuML 모델은 n_features_in_ 속성 확인)
            # 주의: cuML 모델은 내부적으로 학습 시 사용된 피처 수를 저장하므로 정확히 일치해야 함
            model_expected_features = None
            try:
                if hasattr(model, 'n_features_in_'):
                    model_expected_features = model.n_features_in_
                elif hasattr(model, 'n_features_'):
                    model_expected_features = model.n_features_
                # cuML 모델의 경우 내부 모델에서 확인
                elif hasattr(model, 'cuml_model') and hasattr(model.cuml_model, 'n_features_in_'):
                    model_expected_features = model.cuml_model.n_features_in_
                # EnsembleModelWrapper의 경우 내부 모델들 확인
                elif hasattr(model, 'models') and len(model.models) > 0:
                    first_model = model.models[0]
                    if hasattr(first_model, 'n_features_in_'):
                        model_expected_features = first_model.n_features_in_
            except Exception:
                pass  # 모델 피처 수 확인 실패는 무시
            
            # 가장 먼저 확인: 모델이 저장된 features 리스트와 실제 모델의 피처 수가 일치하는지 확인
            # 이게 불일치하면 근본적인 문제이므로 먼저 체크
            if model_expected_features is not None and model_expected_features != len(features):
                log_error(f"   ❌ 심각한 불일치: 모델 내부는 {model_expected_features}개 피처를 기대하지만, 저장된 features 리스트는 {len(features)}개입니다.")
                log_error(f"   ❌ 저장된 features: {features}")
                log_error(f"   ❌ 이 모델은 피처 수가 불일치합니다. 모델을 다시 학습해야 합니다.")
                error_msg = f"모델 내부 피처 수({model_expected_features})와 저장된 features 리스트({len(features)})가 일치하지 않습니다. 모델을 다시 학습해야 합니다."
                log_critical(error_msg)
                raise ValueError(error_msg)
            
            # 모델이 기대하는 피처 수와 사용 가능한 피처 수가 일치하는지 확인
            if model_expected_features is not None and model_expected_features != len(available_features):
                log_error(f"   ❌ 모델 피처 수 불일치: 모델은 {model_expected_features}개를 기대하지만 데이터는 {len(available_features)}개입니다.")
                log_error(f"   ❌ 모델이 저장된 features 리스트: {len(features)}개 - {features}")
                log_error(f"   ❌ 사용 가능한 features: {len(available_features)}개 - {available_features}")
                if model_expected_features > len(available_features):
                    error_msg = f"모델은 {model_expected_features}개 피처를 기대하지만 {len(available_features)}개만 제공되었습니다. 모델을 다시 학습해야 합니다."
                    log_critical(error_msg)
                    raise ValueError(error_msg)
                else:
                    log_warning(f"   ⚠️ 모델이 기대하는 피처 수({model_expected_features})가 제공된 피처 수({len(available_features)})보다 적습니다. 예측을 시도하지만 오류가 발생할 수 있습니다.")
            
            # 사용 가능한 피처만 선택 (모델이 저장된 순서대로)
            test_data_for_pred = test_data[available_features].copy()
            
            # imputation_values 로드 (모델 데이터에서)
            imputation_values = None
            if 'imputation_values' in model_data and model_data['imputation_values']:
                imputation_values = model_data['imputation_values']
            
            # imputation_values가 있으면 사용, 없으면 0으로 채움
            if imputation_values:
                available_imputation = {k: v for k, v in imputation_values.items() if k in available_features}
                test_data_for_pred.fillna(available_imputation, inplace=True)
            else:
                test_data_for_pred.fillna(0, inplace=True)
            
            # 스케일러의 피처 수 확인 (cuML scaler는 n_features_in_ 속성 사용)
            scaler_expected_features = None
            if scaler and hasattr(scaler, 'n_features_in_'):
                scaler_expected_features = scaler.n_features_in_
            elif scaler and hasattr(scaler, 'mean_'):
                # cuML StandardScaler는 mean_ 속성의 길이로 피처 수 확인 가능
                scaler_expected_features = len(scaler.mean_) if hasattr(scaler.mean_, '__len__') else None
            
            # 현재 데이터의 피처 수
            current_features_count = len(available_features)
            
            # 스케일러와 데이터의 피처 수가 일치하는지 확인
            use_scaler = scaler is not None
            if scaler_expected_features is not None and scaler_expected_features != current_features_count:
                log_warning(f"스케일러 피처 수 불일치: 스케일러는 {scaler_expected_features}개를 기대하지만 데이터는 {current_features_count}개입니다. 스케일링을 건너뜁니다.")
                use_scaler = False
            
            # cuML 모델인 경우 cuDF DataFrame으로 변환 필요
            if is_cuml_model:
                import cudf
                # cuML scaler인지 확인
                if use_scaler and hasattr(scaler, 'transform'):
                    # cuML scaler는 cuDF DataFrame을 받아야 함
                    try:
                        # 모델이 저장된 피처 순서대로 컬럼 정렬 (중요: 순서가 정확히 일치해야 함)
                        test_data_ordered = test_data_for_pred[available_features]
                        test_data_cudf = cudf.from_pandas(test_data_ordered)
                        X_test_scaled_cudf = scaler.transform(test_data_cudf)
                        # cuML 모델은 cuDF DataFrame을 받을 수 있음
                        X_test_scaled = X_test_scaled_cudf
                    except ImportError:
                        log_warning("cuDF를 사용할 수 없습니다. 원본 데이터 사용")
                        test_data_ordered = test_data_for_pred[available_features]
                        X_test_scaled = cudf.from_pandas(test_data_ordered)
                    except Exception as e:
                        log_warning(f"cuML 스케일링 오류: {e}, 원본 데이터 사용")
                        test_data_ordered = test_data_for_pred[available_features]
                        X_test_scaled = cudf.from_pandas(test_data_ordered)
                else:
                    # 스케일러를 사용하지 않음 (피처 수 불일치 또는 스케일러 없음)
                    # cuML 모델은 cuDF DataFrame을 기대하므로 변환
                    # 모델이 저장된 피처 순서대로 컬럼 정렬 (중요: 순서가 정확히 일치해야 함)
                    test_data_ordered = test_data_for_pred[available_features]
                    X_test_scaled = cudf.from_pandas(test_data_ordered)
            elif use_scaler:
                # sklearn scaler (sklearn 모델)
                try:
                    X_test_scaled = scaler.transform(test_data_for_pred)
                except Exception as e:
                    log_warning(f"sklearn 스케일링 오류: {e}, 원본 데이터 사용")
                    X_test_scaled = test_data_for_pred.values
            else:
                # 스케일러를 사용하지 않음 (sklearn 모델)
                X_test_scaled = test_data_for_pred.values
            
            # 예측 수행 (큰 데이터셋의 경우 배치 처리)
            # cuDF DataFrame인 경우 len() 사용, numpy 배열인 경우 shape[0] 사용
            if hasattr(X_test_scaled, '__len__'):
                test_size = len(X_test_scaled)
            elif hasattr(X_test_scaled, 'shape'):
                test_size = X_test_scaled.shape[0]
            else:
                test_size = test_data_for_pred.shape[0]
            
            if test_size > 1000000 and is_cuml_model:
                log_info(f"   [PRED] 큰 데이터셋 ({test_size:,}행) - 배치 처리로 예측")
                batch_size = 500000  # 배치 크기
                num_batches = (test_size + batch_size - 1) // batch_size
                
                pred_proba_list = []
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, test_size)
                    
                    # cuDF DataFrame인 경우 iloc 사용
                    if hasattr(X_test_scaled, 'iloc'):
                        X_batch = X_test_scaled.iloc[start_idx:end_idx]
                    else:
                        # numpy 배열인 경우
                        X_batch = X_test_scaled[start_idx:end_idx]
                    
                    try:
                        batch_proba = model.predict_proba(X_batch)
                        # 결과 처리
                        if isinstance(batch_proba, np.ndarray):
                            if batch_proba.ndim == 2:
                                pred_proba_list.append(batch_proba[:, 1])
                            else:
                                pred_proba_list.append(batch_proba)
                        else:
                            # cuDF Series나 다른 형태
                            if hasattr(batch_proba, 'iloc'):
                                pred_proba_list.append(batch_proba.iloc[:, 1].to_pandas().values if hasattr(batch_proba.iloc[:, 1], 'to_pandas') else batch_proba.iloc[:, 1].values)
                            else:
                                pred_proba_list.append(batch_proba[:, 1] if hasattr(batch_proba, '__getitem__') else batch_proba)
                        
                        del X_batch, batch_proba
                        # 배치 간 메모리 정리
                        if batch_idx < num_batches - 1:
                            enhanced_gpu_memory_cleanup(force_defrag=False) if 'enhanced_gpu_memory_cleanup' in globals() else None
                    except Exception as e:
                        log_error(f"   ❌ 배치 {batch_idx+1}/{num_batches} 예측 실패: {e}")
                        # 실패한 배치는 중립값(0.5)으로 채움
                        batch_len = end_idx - start_idx
                        pred_proba_list.append(np.full(batch_len, 0.5))
                        del X_batch
                        enhanced_gpu_memory_cleanup(force_defrag=False) if 'enhanced_gpu_memory_cleanup' in globals() else None
                
                # 배치 결과 합치기
                pred_proba = np.concatenate(pred_proba_list)
                del pred_proba_list
            else:
                # 작은 데이터셋은 한 번에 처리
                pred_proba = model.predict_proba(X_test_scaled)
            
            # 반환 형태에 따라 처리 (cuML 모델은 cuDF DataFrame 반환)
            if isinstance(pred_proba, np.ndarray):
                if pred_proba.ndim == 2:
                    test_data['ml_pred_proba'] = pred_proba[:, 1]
                else:
                    test_data['ml_pred_proba'] = pred_proba
            elif hasattr(pred_proba, 'iloc'):
                # cuDF DataFrame인 경우
                if hasattr(pred_proba.iloc[:, 1], 'to_pandas'):
                    test_data['ml_pred_proba'] = pred_proba.iloc[:, 1].to_pandas().values
                elif hasattr(pred_proba.iloc[:, 1], 'to_numpy'):
                    test_data['ml_pred_proba'] = pred_proba.iloc[:, 1].to_numpy()
                else:
                    test_data['ml_pred_proba'] = pred_proba.iloc[:, 1].values
            else:
                # 기타 형태는 그대로 사용
                test_data['ml_pred_proba'] = pred_proba[:, 1] if hasattr(pred_proba, '__getitem__') else pred_proba
            
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

        # LGBM은 일자별 최종 점수 계산 단계에서 필요할 때만 붙이도록 run_detailed_backtest에서 처리합니다.
        # (가중치 0이면 run_detailed_backtest에서 스킵)
    
        # 데이터 전처리 (강화된 에러 처리)
        try:
            # 테스트 기간에 맞게 데이터 필터링
            test_data = test_data[(test_data['date'] >= pd.to_datetime(start_date)) & 
                                  (test_data['date'] <= pd.to_datetime(end_date))]
            
            # 로그: 백테스팅용 데이터 상태 확인
            log_info(f"🔍 백테스팅용 데이터: {len(test_data):,}개 행")
            log_info(f"🔍 데이터 날짜 범위: {test_data['date'].min()} ~ {test_data['date'].max()}")
            
            log_info("백테스팅 데이터 전처리 완료", context={
                "filtered_rows": len(test_data),
                "date_range": f"{test_data['date'].min()} ~ {test_data['date'].max()}",
                "test_period": f"{start_date} ~ {end_date}"
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
    parser.add_argument('--start-date', type=str, default=None, help='Test start date (YYYY-MM-DD format, default: 1 year ago)')
    parser.add_argument('--end-date', type=str, default=None, help='Test end date (YYYY-MM-DD format, default: today)')
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
            end_date=args.end_date
        )

