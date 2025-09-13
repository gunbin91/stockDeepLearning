# data_fetcher.py

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from pykrx import stock
import pandas_ta as ta
import concurrent.futures
from tqdm import tqdm
import os
import sys
from datetime import datetime, timedelta
import time
import config
import gc
from logger import log_info, log_warning, log_error, log_critical
from exceptions import DataFetchError, DataValidationError
from smart_cache import get_cache, cached

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FINANCIAL_DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'financial_data_pykrx_pit.parquet')

def get_fs_data_from_pit(stock_list, selected_analysis_date):
    try:
        log_info(f"재무 지표 데이터베이스 로딩 시작: {FINANCIAL_DB_PATH}")
        funda_df = pd.read_parquet(FINANCIAL_DB_PATH)
        log_info("재무 지표 데이터베이스 로딩 완료")
    except FileNotFoundError:
        error_msg = f"재무 지표 데이터베이스 파일({FINANCIAL_DB_PATH})을 찾을 수 없습니다."
        log_critical(error_msg)
        log_info("먼저 `scripts/build_db_pykrx.py`를 실행하여 데이터베이스를 생성해주세요.")
        raise DataFetchError(error_msg, source="pykrx_database")
    except Exception as e:
        import traceback
        error_msg = f"재무 지표 데이터베이스 로딩 중 오류 발생: {e}"
        log_error(error_msg)
        print(f"❌ [ERROR] {error_msg}")
        print(f"❌ [ERROR] 상세 오류 정보:")
        print(traceback.format_exc())
        raise DataFetchError(error_msg, source="pykrx_database")
    
    log_info(f"분석 기준일({selected_analysis_date.strftime('%Y-%m-%d')}): pykrx DB에서 최신 재무 지표 조회.")
    analysis_date_ts = pd.to_datetime(selected_analysis_date)
    
    available_funda = funda_df[funda_df['date'] <= analysis_date_ts].copy()
    if available_funda.empty:
        warning_msg = f"{analysis_date_ts.strftime('%Y-%m-%d')} 이전에 집계된 재무 지표 데이터가 없습니다."
        log_warning(warning_msg)
        return pd.DataFrame()
    
    latest_funda = available_funda.sort_values('date').drop_duplicates(subset=['종목코드'], keep='last')
    
    result_df = pd.merge(stock_list[['종목코드']], latest_funda, on='종목코드', how='left')
    log_info(f"✅ 사용 가능한 재무 지표 처리 완료: {len(result_df.dropna())}개 기업")
    return result_df

@cached("stock_list", ttl_seconds=86400)  # 24시간 캐시
def fetch_stock_list():
    log_info("FinanceDataReader를 통해 KOSPI 및 KOSDAQ 전 종목 시가총액 정보 수집 (KRX-MARCAP)...")
    try:
        df_marcap = fdr.StockListing('KRX-MARCAP')
        df_marcap = df_marcap[~df_marcap['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list = df_marcap[['Code', 'Name', 'Stocks']].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명', 'Stocks': '상장주식수'}, inplace=True)
        stock_list = stock_list[stock_list['상장주식수'] > 0]
        log_info(f"총 {len(stock_list)}개 종목을 찾았습니다.")
        return stock_list
    except Exception as e:
        error_msg = f"FinanceDataReader API 통신 실패 (KRX-MARCAP): {e}"
        log_error(error_msg)
        raise DataFetchError(error_msg, source="FinanceDataReader")

@cached("macro_data", ttl_seconds=3600)  # 1시간 캐시
def _fetch_macro_data(start_date, end_date):
    log_info(f"거시 경제 지표 데이터 수집 중 ({start_date} ~ {end_date})...")
    try:
        kospi = fdr.DataReader('KS11', start_date, end_date)
        usdkrw = fdr.DataReader('USD/KRW', start_date, end_date)
        vix = fdr.DataReader('^VIX', start_date, end_date)
        macro_df = pd.concat([kospi['Close'].rename('KOSPI'), usdkrw['Close'].rename('USDKRW'), vix['Close'].rename('VIX')], axis=1).ffill()
        for col in ['KOSPI', 'USDKRW', 'VIX']:
            if col in macro_df.columns: 
                macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
                macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        log_info("✅ 거시 경제 지표 수집 완료.")
        return macro_df
    except Exception as e:
        error_msg = f"거시 경제 지표 수집 실패: {e}"
        log_error(error_msg)
        raise DataFetchError(error_msg, source="macro_economic_data")

def fetch_and_process_ticker_data(stock_info, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df):
    ticker = stock_info['종목코드']; shares = stock_info['상장주식수']
    try:
        fetch_start = (pd.to_datetime(start_date_for_fetch) - timedelta(days=60)).strftime('%Y-%m-%d')
        df_price_full = fdr.DataReader(ticker, fetch_start, end_date_for_fetch)
        if df_price_full.empty or len(df_price_full) < 251 + 60: return None, None
        df_price_full.rename(columns={'Close':'종가', 'High':'고가', 'Low':'저가', 'Volume':'거래량'}, inplace=True)

        selected_analysis_date_ts = pd.Timestamp(selected_analysis_date)
        df_temp = df_price_full[df_price_full.index <= selected_analysis_date_ts]
        if df_temp.empty: return None, None
        actual_analysis_date = df_temp.index.max()
        reference_date_price = df_temp.loc[actual_analysis_date]['종가']
        latest_current_price = df_price_full.iloc[-1]['종가']
        
        df_for_indicators = df_price_full[df_price_full.index <= actual_analysis_date].copy()
        
        fs_data = latest_fs_df[latest_fs_df['종목코드'] == ticker]
        if fs_data.empty or fs_data[['PER', 'PBR']].isnull().values.any(): return None, None
        
        latest_data = {} 

        df_for_indicators['거래대금'] = df_for_indicators['종가'] * df_for_indicators['거래량']
        latest_data['log_mktcap'] = np.log(reference_date_price * shares) if (reference_date_price * shares) > 0 else np.nan
        latest_data['이익수익률'] = 1 / fs_data['PER'].values[0] if fs_data['PER'].values[0] != 0 else np.nan

        latest_data['수익률(1M)'] = df_for_indicators['종가'].pct_change(20).iloc[-1]
        latest_data['수익률(3M)'] = df_for_indicators['종가'].pct_change(60).iloc[-1]
        latest_data['변동성(1W)'] = (df_for_indicators['종가'].rolling(5).std() / df_for_indicators['종가'].rolling(5).mean()).iloc[-1]
        latest_data['변동성(1M)'] = (df_for_indicators['종가'].rolling(20).std() / df_for_indicators['종가'].rolling(20).mean()).iloc[-1]
        latest_data['변동성(3M)'] = (df_for_indicators['종가'].rolling(60).std() / df_for_indicators['종가'].rolling(60).mean()).iloc[-1]
        latest_data['거래대금_MA5'] = df_for_indicators['거래대금'].rolling(5).mean().iloc[-1]
        latest_data['거래대금_MA20'] = df_for_indicators['거래대금'].rolling(20).mean().iloc[-1]
        
        df_for_indicators.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
        df_for_indicators.ta.obv(close='종가', volume='거래량', append=True)
        df_for_indicators.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
        
        bbands = df_for_indicators.ta.bbands(close='종가', length=20, std=2)
        # <<< ✨ 핵심 수정: pandas-ta 버전업에 따른 볼린저밴드 컬럼명 변경 대응 ✨ >>>
        if bbands is not None and all(col in bbands.columns for col in ['BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBM_20_2.0_2.0']):
             latest_data['BBW_20_2'] = ((bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / bbands['BBM_20_2.0_2.0']).iloc[-1]
             # BB_Position 계산: (현재가 - 하단밴드) / (상단밴드 - 하단밴드)
             current_price = df_for_indicators['종가'].iloc[-1]
             bb_lower = bbands['BBL_20_2.0_2.0'].iloc[-1]
             bb_upper = bbands['BBU_20_2.0_2.0'].iloc[-1]
             if bb_upper != bb_lower:
                 latest_data['BB_Position'] = (current_price - bb_lower) / (bb_upper - bb_lower)
             else:
                 latest_data['BB_Position'] = 0.5  # 중앙값
        else:
             latest_data['BBW_20_2'] = np.nan
             latest_data['BB_Position'] = np.nan

        for p in [120, 240]:
            ma = df_for_indicators['종가'].rolling(window=p).mean()
            latest_data[f'disparity_{p}'] = ((df_for_indicators['종가'] / ma) * 100).iloc[-1]

        latest_data['52주_신고가_비율'] = (df_for_indicators['종가'] / df_for_indicators['종가'].rolling(250).max()).iloc[-1]

        technical_features_to_add = ['ATRr_14', 'OBV', 'ADX_14']
        for feature in technical_features_to_add:
            if feature in df_for_indicators.columns:
                 latest_data[feature] = df_for_indicators[feature].iloc[-1]

        latest_data.update(fs_data.iloc[0].to_dict())
        latest_data['종목명'] = stock_info['종목명']
        latest_data['현재가'] = latest_current_price
        latest_data['기준일가'] = reference_date_price
        if '시가총액_기준일' in stock_info and pd.notna(stock_info['시가총액_기준일']):
            latest_data['시가총액'] = stock_info['시가총액_기준일'] / 1_0000_0000
        else:
            latest_data['시가총액'] = (reference_date_price * shares) / 1_0000_0000
        latest_data['종목코드'] = ticker # Add ticker code to the features

        return latest_data, actual_analysis_date
    except Exception as e:
        return None, None

def fetch_all_data(stock_list, selected_analysis_date):
    today = datetime.now()
    end_date_for_fetch = today.strftime('%Y-%m-%d')
    start_date_for_fetch = (today - timedelta(days=450)).strftime('%Y-%m-%d')

    latest_fs_df = get_fs_data_from_pit(stock_list, selected_analysis_date)
    
    if latest_fs_df.empty or latest_fs_df.dropna().empty:
        print("사용 가능한 재무 데이터가 없어 분석을 중단합니다."); return pd.DataFrame(), None

    if selected_analysis_date.date() < today.date():
        print(f"과거 분석(기준일={selected_analysis_date.strftime('%Y-%m-%d')}): 기준일의 시가총액 데이터를 수집합니다.")
        try:
            df_marcap_past = fdr.StockListing('KRX-MARCAP', selected_analysis_date.strftime('%Y%m%d'))
            df_marcap_past.rename(columns={'Code': '종목코드'}, inplace=True)
            
            stock_list = pd.merge(
                stock_list[['종목코드', '종목명']],
                df_marcap_past[['종목코드', 'Marcap', 'Stocks']],
                on='종목코드',
                how='inner'
            )
            stock_list.rename(columns={'Stocks': '상장주식수', 'Marcap': '시가총액_기준일'}, inplace=True)
            print(f"✅ 기준일({selected_analysis_date.strftime('%Y-%m-%d')})에 존재했던 {len(stock_list)}개 종목으로 필터링되었습니다.")
        except Exception as e:
            print(f"경고: 기준일({selected_analysis_date.strftime('%Y-%m-%d')})의 시가총액 데이터를 가져오는 데 실패했습니다: {e}")
            print("최신 상장주식수 정보를 사용하여 분석을 계속합니다.")
            
    macro_df = _fetch_macro_data(start_date_for_fetch, end_date_for_fetch)
    if macro_df.empty:
        print("거시 경제 데이터 수집에 실패하여 분석을 중단합니다."); return pd.DataFrame(), None
    all_feature_data, all_actual_dates = [], []
    stock_records = stock_list.to_dict('records')
    
    # 배치 단위로 처리하여 메모리 효율성 향상
    batch_size = 100
    total_batches = (len(stock_records) + batch_size - 1) // batch_size
    total_stocks = len(stock_records)
    
    log_info("📈 주식 분석을 위한 종목 데이터 수집을 시작합니다...")
    log_info(f"   📊 총 {total_stocks:,}개 종목의 거래 데이터를 수집합니다")
    log_info(f"   🔄 {total_batches}개 그룹으로 나누어 안전하게 처리합니다")
    log_info("   ⏱️ 예상 소요 시간: 약 5-10분 (API 응답 속도에 따라 달라질 수 있습니다)")
    print()
    
    for i in range(0, len(stock_records), batch_size):
        batch = stock_records[i:i + batch_size]
        current_batch = i // batch_size + 1
        batch_start = i + 1
        batch_end = min(i + batch_size, total_stocks)
        
        log_info(f"📊 종목 그룹 {current_batch}/{total_batches} 처리 중... ({batch_start:,}~{batch_end:,}번째 종목)")
        log_info(f"   🔍 각 종목의 가격, 거래량, 기술적 지표를 계산하고 있습니다...")
        log_info(f"   📈 수집 데이터: 주가, 거래량, RSI, MACD, 볼린저밴드, 이동평균선 등")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 워커 수 감소
            future_to_stock = {executor.submit(fetch_and_process_ticker_data, r, start_date_for_fetch, end_date_for_fetch, selected_analysis_date, latest_fs_df): r for r in batch}
            
            # 전체 진행률을 포함한 진행률 표시
            total_processed = (current_batch - 1) * batch_size
            total_remaining = total_stocks - total_processed
            
            # 간단한 진행률 표시 (개행 문제 해결)
            completed_count = 0
            total_count = len(batch)
            
            print(f"   └─ 그룹 {current_batch}/{total_batches} - 종목 분석 진행률: 0% (0/{total_count})", end='', flush=True)
            
            for future in concurrent.futures.as_completed(future_to_stock):
                try:
                    result, analysis_date = future.result()
                    if result and analysis_date: 
                        all_feature_data.append(result)
                        all_actual_dates.append(analysis_date)
                    
                    completed_count += 1
                    progress_percent = (completed_count / total_count) * 100
                    
                    # 같은 줄에서 진행률 업데이트
                    print(f"\r   └─ 그룹 {current_batch}/{total_batches} - 종목 분석 진행률: {progress_percent:.0f}% ({completed_count}/{total_count})", end='', flush=True)
                    
                except Exception: 
                    completed_count += 1
                    progress_percent = (completed_count / total_count) * 100
                    print(f"\r   └─ 그룹 {current_batch}/{total_batches} - 종목 분석 진행률: {progress_percent:.0f}% ({completed_count}/{total_count})", end='', flush=True)
                    continue
            
            # 완료 후 개행
            print()  # 개행 추가
        
        # 배치 간 메모리 정리
        gc.collect()
        time.sleep(0.1)  # API 부하 방지
        
        success_count = len(all_feature_data)
        progress_percent = (current_batch / total_batches) * 100
        log_info(f"   ✅ 그룹 {current_batch}/{total_batches} 완료! ({progress_percent:.1f}% 진행)")
        log_info(f"   📊 현재까지 {success_count:,}개 종목의 분석 데이터를 수집했습니다")
        log_info(f"   💾 메모리 정리 및 API 부하 방지를 위해 잠시 대기 중...")
        print()
    
    log_info("🎉 종목 데이터 수집이 모두 완료되었습니다!")
    log_info(f"   📊 총 {len(all_feature_data):,}개 종목의 분석 데이터를 준비했습니다")
    log_info("   🔄 데이터 정제 및 거시경제 지표 병합을 시작합니다...")
    print()
    
    if not all_feature_data: return pd.DataFrame(), None
    final_df = pd.DataFrame(all_feature_data)

    final_df['date'] = pd.to_datetime(all_actual_dates)
    final_df = final_df.sort_values('date')
    macro_df = macro_df.sort_index()
    
    # 인덱스를 datetime으로 확실히 변환
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        macro_df.index = pd.to_datetime(macro_df.index)
    
    log_info("   🔗 거시경제 지표(KOSPI, USD/KRW, VIX)를 종목 데이터와 병합 중...")
    
    # merge_asof 대신 더 안전한 방법 사용
    try:
        final_df = pd.merge_asof(final_df, macro_df, left_on='date', right_index=True, direction='backward')
    except Exception as e:
        log_warning(f"   ⚠️ merge_asof 실패, 일반 merge로 시도: {e}")
        # 일반 merge로 대체
        macro_df_reset = macro_df.reset_index()
        macro_df_reset.rename(columns={'index': 'date'}, inplace=True)
        final_df = pd.merge(final_df, macro_df_reset, on='date', how='left')
    
    log_info("   🧹 무한대 값 및 결측값 정제 중...")
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['종목코드', '종목명', '현재가'], inplace=True)
    
    if final_df.empty:
        log_error("피처 생성 후 유효한 데이터가 없습니다.")
        return pd.DataFrame(), None
        
    actual_analysis_date_final = pd.to_datetime(final_df['date'].mode()[0])
    
    log_info("✅ 모든 피처 데이터 생성 완료!")
    log_info(f"   📊 최종 데이터: {len(final_df):,}개 종목, {len(final_df.columns)}개 피처")
    return final_df, actual_analysis_date_final