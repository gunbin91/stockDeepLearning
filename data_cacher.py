import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from pykrx import stock
import time
from datetime import datetime, timedelta
import pandas_ta as ta
import concurrent.futures
from tqdm import tqdm
import os
import gc

from scoring import calculate_factor_scores
from smart_cache import get_cache, cached

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
FINANCIAL_DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'financial_data_pykrx_pit.parquet')
CACHE_END_DATE = datetime(datetime.now().year, 12, 31).strftime('%Y-%m-%d')
CACHE_FILENAME = f"historical_data_up_to_{CACHE_END_DATE.replace('-', '')}.parquet"
CACHE_FILE_PATH = os.path.join(CACHE_DIR, CACHE_FILENAME)

from logger import log_info, log_critical, log_error, log_warning

try:
    funda_df = pd.read_parquet(FINANCIAL_DB_PATH)
    funda_df['date'] = pd.to_datetime(funda_df['date'])
    funda_df.sort_values('date', inplace=True)
    log_info(f"pykrx 시점(Point-in-Time) 재무 지표 데이터베이스 로드 완료", 
             context={'file_path': FINANCIAL_DB_PATH, 'records': len(funda_df)})
except FileNotFoundError:
    error_msg = f"재무 지표 데이터베이스 파일({FINANCIAL_DB_PATH})을 찾을 수 없습니다"
    log_critical(error_msg, context={'file_path': FINANCIAL_DB_PATH, 'action': 'build_db_pykrx_required'})
    print("먼저 `scripts/build_db_pykrx.py`를 실행하여 데이터베이스를 생성해주세요.")
    funda_df = pd.DataFrame()
except Exception as e:
    log_error("재무 지표 데이터베이스 로딩 중 오류 발생", exception=e, 
              context={'file_path': FINANCIAL_DB_PATH})
    funda_df = pd.DataFrame()

def auto_update_financial_db():
    """재무 데이터베이스 자동 업데이트"""
    if not os.path.exists(FINANCIAL_DB_PATH):
        log_warning("재무 데이터베이스 파일이 없습니다. 초기 생성이 필요합니다.", 
                   context={'file_path': FINANCIAL_DB_PATH})
        return False
    
    try:
        # 현재 최신 날짜 확인
        funda_df = pd.read_parquet(FINANCIAL_DB_PATH)
        funda_df['date'] = pd.to_datetime(funda_df['date'])
        latest_date = funda_df['date'].max().date()
        yesterday = datetime.now().date() - timedelta(days=1)
        
        log_info("재무 데이터베이스 상태 확인", 
                context={'latest_date': str(latest_date), 'yesterday': str(yesterday)})
        
        # 어제까지 데이터가 없으면 수집
        if latest_date < yesterday:
            log_info("재무 데이터베이스 업데이트 필요", 
                    context={'from_date': str(latest_date), 'to_date': str(yesterday)})
            
            missing_start = (latest_date + timedelta(days=1)).strftime('%Y%m%d')
            missing_end = yesterday.strftime('%Y%m%d')
            
            log_info("누락 기간 데이터 수집 시작", 
                    context={'period': f"{missing_start} ~ {missing_end}"})
            
            # 새 데이터 수집 (build_db_pykrx.py 로직 활용)
            new_data = _collect_financial_data_period(missing_start, missing_end)
            
            if not new_data.empty:
                # 기존 데이터와 병합하여 파일 업데이트
                updated_df = pd.concat([funda_df, new_data], ignore_index=True)
                updated_df.sort_values(by=['date', '종목코드'], inplace=True)
                updated_df.to_parquet(FINANCIAL_DB_PATH, index=False)
                log_info("재무 데이터베이스 업데이트 완료", 
                        context={'new_records': len(new_data), 'total_records': len(updated_df)})
                return True
            else:
                log_warning("재무 데이터 수집 실패")
                return False
        else:
            log_info("재무 데이터베이스가 최신 상태입니다")
            return True
            
    except Exception as e:
        log_error("재무 데이터베이스 업데이트 중 오류", exception=e, 
                 context={'file_path': FINANCIAL_DB_PATH})
        return False

def _collect_financial_data_period(start_date, end_date):
    """지정된 기간의 재무 데이터 수집 (build_db_pykrx.py 로직 활용)"""
    try:
        # 거래일 목록 조회
        trading_days = pd.to_datetime(stock.get_market_ohlcv(start_date, end_date, "005930").index).strftime('%Y%m%d').tolist()
        
        all_daily_data = []
        for day in tqdm(trading_days, desc="재무 데이터 수집"):
            try:
                df_fundamental = stock.get_market_fundamental(day, market="ALL")
                
                if df_fundamental.empty:
                    time.sleep(0.1)
                    continue
                
                if 'PBR' in df_fundamental.columns:
                    df_fundamental = df_fundamental[df_fundamental['PBR'] > 0]
                    
                    if not df_fundamental.empty:
                        df_fundamental.reset_index(inplace=True)
                        df_fundamental.rename(columns={'티커': '종목코드'}, inplace=True)
                        df_fundamental['date'] = pd.to_datetime(day, format='%Y%m%d')
                        all_daily_data.append(df_fundamental)
                
                time.sleep(0.1)
            except Exception:
                continue
        
        if all_daily_data:
            final_df = pd.concat(all_daily_data, ignore_index=True)
            required_cols = ['date', '종목코드', 'PBR', 'PER', 'EPS', 'BPS', 'DIV', 'DPS']
            final_df = final_df[[col for col in required_cols if col in final_df.columns]]
            
            if 'PBR' in final_df.columns and 'PER' in final_df.columns:
                final_df['ROE'] = np.where(final_df['PER'] != 0, final_df['PBR'] / final_df['PER'], np.nan)
            
            final_df.sort_values(by=['date', '종목코드'], inplace=True)
            return final_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ 재무 데이터 수집 중 오류: {e}")
        return pd.DataFrame()

def auto_update_stock_cache():
    """주식 데이터 캐시 자동 업데이트"""
    if not os.path.exists(CACHE_FILE_PATH):
        print("⚠️ 주식 데이터 캐시 파일이 없습니다. 초기 생성이 필요합니다.")
        return False
    
    try:
        # 현재 최신 날짜 확인
        df = pd.read_parquet(CACHE_FILE_PATH)
        df['date'] = pd.to_datetime(df['date'])
        latest_date = df['date'].max().date()
        yesterday = datetime.now().date() - timedelta(days=1)
        
        print(f"📊 주식 데이터 캐시 최신 날짜: {latest_date}")
        print(f"📅 어제 날짜: {yesterday}")
        
        # 어제까지 데이터가 없으면 수집
        if latest_date < yesterday:
            print(f"🔄 주식 데이터 캐시 업데이트 필요: {latest_date} → {yesterday}")
            
            missing_start = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            missing_end = yesterday.strftime('%Y-%m-%d')
            
            print(f"📅 누락 기간: {missing_start} ~ {missing_end}")
            print("⏳ 주식 데이터 수집 중...")
            
            # 새 데이터 수집 (_fetch_and_prepare_data 로직 활용)
            new_data = _fetch_and_prepare_data(missing_start, missing_end)
            
            if not new_data.empty:
                # 기존 데이터와 병합하여 파일 업데이트
                updated_df = pd.concat([df, new_data], ignore_index=True)
                updated_df = updated_df.drop_duplicates(subset=['date', '종목코드'], keep='last')
                updated_df.sort_values(by=['date', '종목코드'], inplace=True)
                updated_df.to_parquet(CACHE_FILE_PATH, index=False)
                print(f"✅ 주식 데이터 캐시 업데이트 완료: {len(new_data):,}개 새 레코드 추가")
                return True
            else:
                print("❌ 주식 데이터 수집 실패")
                return False
        else:
            print("✅ 주식 데이터 캐시가 최신 상태입니다.")
            return True
            
    except Exception as e:
        print(f"❌ 주식 데이터 캐시 업데이트 중 오류: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def auto_update_macro_cache():
    """거시경제 데이터 캐시 자동 업데이트"""
    try:
        # 거시경제 데이터는 스마트 캐시 시스템에서 TTL 기반으로 관리됨
        # 여기서는 캐시 상태만 확인하고 필요시 새로 수집하도록 함
        print("📊 거시경제 데이터 캐시 상태 확인 중...")
        
        # 스마트 캐시에서 거시경제 데이터 확인
        cache = get_cache()
        cache_params = {
            'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'end_date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            'function': 'macro_data'
        }
        
        cached_macro = cache.get('macro_data', cache_params, ttl_seconds=3600)
        if cached_macro is not None:
            print("✅ 거시경제 데이터 캐시가 최신 상태입니다.")
            return True
        else:
            print("⚠️ 거시경제 데이터 캐시가 만료되었습니다. 다음 사용 시 새로 수집됩니다.")
            return True
            
    except Exception as e:
        print(f"❌ 거시경제 데이터 캐시 확인 중 오류: {e}")
        return False

def auto_update_all_caches():
    """모든 캐시 파일 자동 업데이트"""
    print("🔄 캐시 파일 자동 업데이트 시작...")
    
    success_count = 0
    total_count = 3
    
    # 1. 재무 데이터베이스 업데이트
    print("\n1️⃣ 재무 데이터베이스 업데이트 중...")
    if auto_update_financial_db():
        success_count += 1
    
    # 2. 주식 데이터 캐시 업데이트
    print("\n2️⃣ 주식 데이터 캐시 업데이트 중...")
    if auto_update_stock_cache():
        success_count += 1
    
    # 3. 거시경제 데이터 캐시 업데이트
    print("\n3️⃣ 거시경제 데이터 캐시 업데이트 중...")
    if auto_update_macro_cache():
        success_count += 1
    
    print(f"\n✅ 캐시 파일 업데이트 완료: {success_count}/{total_count} 성공")
    return success_count == total_count

def create_initial_financial_db(start_date, end_date):
    """지정된 기간의 재무 데이터베이스 초기 생성"""
    print(f"🏗️ 재무 데이터베이스 초기 생성 시작: {start_date} ~ {end_date}")
    
    start_date_str = start_date.replace('-', '')
    end_date_str = end_date.replace('-', '')
    
    try:
        # 거래일 목록 조회
        trading_days = pd.to_datetime(stock.get_market_ohlcv(start_date_str, end_date_str, "005930").index).strftime('%Y%m%d').tolist()
        
        all_daily_data = []
        for day in tqdm(trading_days, desc="재무 데이터 수집"):
            try:
                df_fundamental = stock.get_market_fundamental(day, market="ALL")
                
                if df_fundamental.empty:
                    time.sleep(0.1)
                    continue
                
                if 'PBR' in df_fundamental.columns:
                    df_fundamental = df_fundamental[df_fundamental['PBR'] > 0]
                    
                    if not df_fundamental.empty:
                        df_fundamental.reset_index(inplace=True)
                        df_fundamental.rename(columns={'티커': '종목코드'}, inplace=True)
                        df_fundamental['date'] = pd.to_datetime(day, format='%Y%m%d')
                        all_daily_data.append(df_fundamental)
                
                time.sleep(0.1)
            except Exception:
                continue
        
        if all_daily_data:
            final_df = pd.concat(all_daily_data, ignore_index=True)
            required_cols = ['date', '종목코드', 'PBR', 'PER', 'EPS', 'BPS', 'DIV', 'DPS']
            final_df = final_df[[col for col in required_cols if col in final_df.columns]]
            
            if 'PBR' in final_df.columns and 'PER' in final_df.columns:
                final_df['ROE'] = np.where(final_df['PER'] != 0, final_df['PBR'] / final_df['PER'], np.nan)
            
            final_df.sort_values(by=['date', '종목코드'], inplace=True)
            final_df.to_parquet(FINANCIAL_DB_PATH, index=False)
            print(f"✅ 재무 데이터베이스 초기 생성 완료: {len(final_df):,}개 레코드")
            return True
        else:
            print("❌ 재무 데이터 수집 실패")
            return False
            
    except Exception as e:
        print(f"❌ 재무 데이터베이스 초기 생성 중 오류: {e}")
        return False

def create_initial_stock_cache(start_date, end_date):
    """지정된 기간의 주식 데이터 캐시 초기 생성"""
    print(f"🏗️ 주식 데이터 캐시 초기 생성 시작: {start_date} ~ {end_date}")
    
    try:
        # _fetch_and_prepare_data 로직 활용
        historical_df = _fetch_and_prepare_data(start_date, end_date)
        
        if not historical_df.empty:
            historical_df.to_parquet(CACHE_FILE_PATH)
            print(f"✅ 주식 데이터 캐시 초기 생성 완료: {len(historical_df):,}개 레코드")
            return True
        else:
            print("❌ 주식 데이터 수집 실패")
            return False
            
    except Exception as e:
        print(f"❌ 주식 데이터 캐시 초기 생성 중 오류: {e}")
        return False

def create_initial_cache_files(start_date, end_date):
    """지정된 기간의 모든 캐싱 파일 초기 생성"""
    print(f"🏗️ 초기 캐싱 파일 생성 시작: {start_date} ~ {end_date}")
    
    success_count = 0
    total_count = 2
    
    # 1. 재무 데이터베이스 생성
    if not os.path.exists(FINANCIAL_DB_PATH):
        print("\n1️⃣ 재무 데이터베이스 초기 생성 중...")
        if create_initial_financial_db(start_date, end_date):
            success_count += 1
    else:
        print("✅ 재무 데이터베이스가 이미 존재합니다.")
        success_count += 1
    
    # 2. 주식 데이터 캐시 생성
    if not os.path.exists(CACHE_FILE_PATH):
        print("\n2️⃣ 주식 데이터 캐시 초기 생성 중...")
        if create_initial_stock_cache(start_date, end_date):
            success_count += 1
    else:
        print("✅ 주식 데이터 캐시가 이미 존재합니다.")
        success_count += 1
    
    print(f"\n✅ 초기 캐싱 파일 생성 완료: {success_count}/{total_count} 성공")
    return success_count == total_count

def fetch_stock_list():
    """통일된 주식 목록 가져오기 (data_fetcher 모듈 사용)"""
    try:
        # data_fetcher의 통일된 함수 사용
        import data_fetcher
        stock_list = data_fetcher.fetch_stock_list()
        # 백테스팅용으로 종목코드, 종목명만 반환
        return stock_list[['종목코드', '종목명']]
    except Exception as e:
        print(f"경고: 통일된 주식 목록 가져오기 실패, 백업 방식 사용: {e}")
        # 백업 방식 (기존 로직)
        try:
            df_kospi = fdr.StockListing('KOSPI')
            df_kosdaq = fdr.StockListing('KOSDAQ')
            stock_list = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
            stock_list = stock_list[~stock_list['Name'].str.contains('스팩|리츠', na=False)].copy()
            stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
            return stock_list[['종목코드', '종목명']]
        except Exception:
            return pd.DataFrame()

def _fetch_macro_data(start_date, end_date):
    print("거시 경제 지표 데이터 수집 중 (KOSPI, USD/KRW, VIX)...")
    try:
        kospi = fdr.DataReader('KS11', start_date, end_date)[['Close']].rename(columns={'Close': 'KOSPI'})
        usdkrw = fdr.DataReader('USD/KRW', start_date, end_date)[['Close']].rename(columns={'Close': 'USDKRW'})
        vix = fdr.DataReader('^VIX', start_date, end_date)[['Close']].rename(columns={'Close': 'VIX'})
        macro_df = pd.merge(kospi, usdkrw, left_index=True, right_index=True, how='outer')
        macro_df = pd.merge(macro_df, vix, left_index=True, right_index=True, how='outer')
        macro_df.sort_index(inplace=True); macro_df.ffill(inplace=True); macro_df.bfill(inplace=True)
        for col in macro_df.columns:
            macro_df[f'{col}_pct_1d'] = macro_df[col].pct_change(1)
            macro_df[f'{col}_pct_5d'] = macro_df[col].pct_change(5)
        macro_df.reset_index(inplace=True); macro_df.rename(columns={'index': 'date'}, inplace=True)
        print("✅ 거시 경제 지표 수집 완료.")
        return macro_df
    except Exception as e:
        print(f"!!!!!!!! [치명적 오류] 거시 경제 지표 수집 실패: {e} !!!!!!!!")
        raise e



def process_single_ticker_data(stock_info, start_date, end_date, df_marcap_long, pbar_lock):
    ticker = stock_info['종목코드']
    try:
        # 하이브리드 방식으로 주가 데이터 수집 (Yahoo Finance → KRX → NAVER)
        df_price = None
        try:
            df_price = fdr.DataReader(ticker, start_date, end_date)
        except:
            try:
                df_price = fdr.DataReader(f'KRX:{ticker}', start_date, end_date)
            except:
                try:
                    df_price = fdr.DataReader(f'NAVER:{ticker}', start_date, end_date)
                except:
                    df_price = None
        
        if df_price is None or df_price.empty or len(df_price) < 251 + 60: return None
        df_price.rename(columns={'Open':'시가', 'Close':'종가', 'High': '고가', 'Low': '저가', 'Volume':'거래량'}, inplace=True)
        df = df_price[['시가', '종가', '고가', '저가', '거래량']].copy(); df.sort_index(inplace=True)
        
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy()
        if df_marcap_ticker.empty: 
            # 시가총액 데이터가 없으면 이전달 데이터로 대체 시도
            print(f"⚠️ {ticker} 종목의 시가총액 데이터가 없어 이전달 데이터로 대체를 시도합니다.")
            return None
        df_marcap_ticker.sort_values(by='Date', inplace=True)
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['Date', 'Marcap']], left_index=True, right_on='Date', direction='backward')
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        
        # 시가총액 데이터가 None인 경우 이전달 데이터로 대체
        if df['시가총액'].isnull().any():
            print(f"⚠️ {ticker} 종목의 일부 시가총액 데이터가 None입니다. 이전달 데이터로 대체합니다.")
            # 이전달 시가총액 데이터로 대체
            df['시가총액'] = df['시가총액'].fillna(method='ffill')
            # 여전히 None이 있으면 이전 데이터로 대체
            if df['시가총액'].isnull().any():
                df['시가총액'] = df['시가총액'].fillna(method='bfill')
        
        df.dropna(subset=['시가총액'], inplace=True)
        if df.empty: return None
        
        if not funda_df.empty:
            ticker_funda = funda_df[funda_df['종목코드'] == ticker]
            if not ticker_funda.empty:
                df = pd.merge_asof(left=df, right=ticker_funda[['date', 'PER', 'PBR', 'ROE', 'EPS', 'BPS']], 
                                   left_index=True, right_on='date', direction='backward')

        if 'PER' not in df.columns or df['PER'].isnull().all() or df['PBR'].isnull().all():
            return None

        df['거래대금'] = df['종가'] * df['거래량']
        df['log_mktcap'] = np.log(df['시가총액'])
        df['이익수익률'] = 1 / df['PER']
        df['수익률(1M)'] = df['종가'].pct_change(20); df['수익률(3M)'] = df['종가'].pct_change(60)
        df['변동성(1W)'] = df['종가'].rolling(5).std() / df['종가'].rolling(5).mean()
        df['변동성(1M)'] = df['종가'].rolling(20).std() / df['종가'].rolling(20).mean()
        df['변동성(3M)'] = df['종가'].rolling(60).std() / df['종가'].rolling(60).mean()
        df['거래대금_MA5'] = df['거래대금'].rolling(5).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(20).mean()
        
        df.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
        df.ta.obv(close='종가', volume='거래량', append=True)
        df.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
        
        bbands = df.ta.bbands(close='종가', length=20, std=2)
        # <<< ✨ 핵심 수정: pandas-ta 버전업에 따른 볼린저밴드 컬럼명 변경 대응 ✨ >>>
        if bbands is not None and all(col in bbands.columns for col in ['BBL_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBM_20_2.0_2.0']):
             df['BBW_20_2'] = (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / bbands['BBM_20_2.0_2.0']
             # BB_Position 계산: (현재가 - 하단밴드) / (상단밴드 - 하단밴드)
             df['BB_Position'] = (df['종가'] - bbands['BBL_20_2.0_2.0']) / (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0'])
             # 0~1 범위로 제한
             df['BB_Position'] = df['BB_Position'].clip(0, 1)
        else:
             df['BBW_20_2'] = np.nan
             df['BB_Position'] = np.nan

        for p in [120, 240]:
            ma = df['종가'].rolling(window=p).mean()
            df[f'disparity_{p}'] = (df['종가'] / ma) * 100

        df['52주_최고가'] = df['종가'].rolling(250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']

        df['target'] = (df['종가'].shift(-15) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker
        df.set_index('Date', inplace=True)
        df.drop(columns=['date'], inplace=True, errors='ignore')
        return df
    except Exception as e:
        with pbar_lock: tqdm.write(f"⚠️ {stock_info['종목명']}({ticker}) 데이터 처리 중 오류: {e} (건너뜀)"); return None

def get_marcap_dates(start_date, end_date):
    """년도/월 비교 기반 시가총액 데이터 수집 날짜 생성"""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    marcap_dates = []
    
    # 현재 년도/월인지 확인
    if (start_dt.year == current_year and start_dt.month == current_month) or \
       (end_dt.year == current_year and end_dt.month == current_month):
        # 현재 월: 일별 수집 (거래일만)
        daily_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        for day in daily_range:
            if day.weekday() < 5:  # 월-금 거래일만
                marcap_dates.append(day.strftime('%Y%m%d'))
    else:
        # 과거 월: 월별 수집
        monthly_dates = pd.date_range(start=start_dt, end=end_dt, freq='M')
        marcap_dates = [date.strftime('%Y%m%d') for date in monthly_dates]
    
    return marcap_dates

def _fetch_and_prepare_data(start_date, end_date):
    print(f"데이터 준비 중 ({start_date} ~ {end_date})...")
    stock_list = fetch_stock_list()
    if stock_list.empty: raise ValueError("종목 리스트를 가져올 수 없습니다.")
    try:
        # 개선된 시가총액 데이터 수집 날짜 생성
        marcap_dates = get_marcap_dates(start_date, end_date)
        
        if not marcap_dates:
            raise Exception("수집할 시가총액 데이터 날짜가 없습니다.")
        
        marcap_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(fdr.StockListing, 'KRX-MARCAP', date): date for date in marcap_dates}
            for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(marcap_dates), desc="시가총액 데이터 수집"):
                try:
                    date_str = future_to_date[future]; result_df = future.result()
                    if not result_df.empty: result_df['Date'] = pd.to_datetime(date_str); marcap_dfs.append(result_df)
                except Exception: continue
        
        if not marcap_dfs: raise Exception("수집된 시가총액 데이터가 없습니다.")
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'Date'], inplace=True)
        
        # 시가총액 데이터 수집 완료 로그
        log_info(f"✅ 시가총액 데이터 수집 완료: {len(marcap_dfs)}개 날짜, {len(df_marcap_long)}개 레코드")
    except Exception as e:
        raise ConnectionError(f"과거 시가총액 데이터 수집 실패: {e}")
    
    all_data = []; stock_records = stock_list.to_dict('records')
    
    # 개별 종목 피처 데이터 생성 시작 로그
    log_info(f"📊 개별 종목 피처 데이터 생성 시작: {len(stock_records)}개 종목")
    
    with tqdm(total=len(stock_records), desc="개별 종목 피처 데이터 생성") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_stock = {executor.submit(process_single_ticker_data, row, start_date, end_date, df_marcap_long, pbar.get_lock()): row for row in stock_records}
            for future in concurrent.futures.as_completed(future_to_stock):
                try:
                    result_df = future.result()
                    if result_df is not None: all_data.append(result_df)
                except Exception: pass
                pbar.update(1)

    if not all_data: raise ValueError("처리된 데이터가 없습니다.")
    
    # 개별 종목 피처 데이터 생성 완료 로그
    log_info(f"✅ 개별 종목 피처 데이터 생성 완료: {len(all_data)}개 종목 처리됨")
    
    raw_feature_df = pd.concat(all_data).reset_index()
    raw_feature_df.rename(columns={'Date': 'date'}, inplace=True)
    raw_feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    raw_feature_df.dropna(subset=['date', '종목코드'], inplace=True)
    raw_feature_df['date'] = pd.to_datetime(raw_feature_df['date'])
    raw_feature_df.drop_duplicates(subset=['date', '종목코드'], keep='first', inplace=True)
    
    
    
    macro_df = _fetch_macro_data(start_date, end_date)
    if not macro_df.empty:
        raw_feature_df = pd.merge(raw_feature_df, macro_df, on='date', how='left')
    
    raw_feature_df.sort_values(by=['date', '종목코드'], inplace=True)

    print("일별 팩터 점수 계산 중...")
    final_df = raw_feature_df.groupby('date', group_keys=False).apply(calculate_factor_scores).reset_index(drop=True)
    
    return final_df

def get_preprocessed_data(start_date, end_date, use_cache=True, auto_update=True):
    """메모리 최적화된 데이터 전처리 함수"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = get_cache()
    
    # 캐시 사용 시 자동 업데이트 실행
    if use_cache and auto_update:
        print("🔄 캐시 파일 자동 업데이트 확인 중...")
        
        # 1. 초기 캐싱 파일이 없으면 생성
        if not os.path.exists(FINANCIAL_DB_PATH) or not os.path.exists(CACHE_FILE_PATH):
            print("⚠️ 초기 캐싱 파일이 없습니다. 초기 생성합니다...")
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            create_initial_cache_files(start_date, yesterday)
        
        # 2. 기존 자동 업데이트 실행
        auto_update_all_caches()
    
    # 실제 거래일을 확인하여 오늘 날짜 분석인지 판단
    today = datetime.now().date()
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    # 공통 함수를 사용하여 실제 거래일 확인
    from data_fetcher import get_actual_trading_date
    actual_trading_date = get_actual_trading_date(pd.to_datetime(end_date))
    
    is_today_analysis = actual_trading_date == today
    
    # 캐시 사용 여부 확인
    if not use_cache:
        print(f"🔄 주식추천 페이지: 정합성 있는 실시간 데이터 수집을 위해 캐시를 우회합니다 ({start_date} ~ {end_date})")
    elif not is_today_analysis:
        # 캐시 키 생성
        cache_params = {
            'start_date': start_date,
            'end_date': end_date,
            'function': 'get_preprocessed_data'
        }
        cached_data = cache.get('preprocessed_data', cache_params, ttl_seconds=3600)
        if cached_data is not None:
            print(f"✅ 캐시된 데이터 로딩: {start_date} ~ {end_date}")
            return cached_data
    else:
        print(f"🔄 오늘 날짜 분석: 실시간 데이터 수집을 위해 캐시를 우회합니다 ({start_date} ~ {end_date})")
    
    print(f"⚠️ 캐시 미스. 데이터를 새로 생성합니다: {start_date} ~ {end_date}")
    
    # 청크 기반으로 데이터 처리
    historical_df = _get_historical_data_chunked(start_date, end_date)
    fresh_df = _get_fresh_data_chunked(start_date, end_date, is_today_analysis)
    
    # 메모리 효율적인 병합
    if not historical_df.empty and not fresh_df.empty:
        combined_df = pd.concat([historical_df, fresh_df], ignore_index=True)
        # 중복 제거
        combined_df = combined_df.drop_duplicates(subset=['date', '종목코드'], keep='last')
    elif not historical_df.empty:
        combined_df = historical_df
    elif not fresh_df.empty:
        combined_df = fresh_df
    else:
        combined_df = pd.DataFrame()
    
    # 날짜 범위 필터링
    if not combined_df.empty:
        final_df = combined_df[
            (combined_df['date'] >= pd.to_datetime(start_date)) & 
            (combined_df['date'] <= pd.to_datetime(end_date))
        ].copy()
    else:
        final_df = pd.DataFrame()
    
    # 결과 캐싱
    if not final_df.empty:
        cache.set('preprocessed_data', cache_params, final_df, ttl_seconds=3600)
    
    # 메모리 정리
    del historical_df, fresh_df, combined_df
    gc.collect()
    
    print('✅ 모든 데이터 준비 완료.')
    return final_df

def _get_historical_data_chunked(start_date, end_date):
    """과거 데이터를 청크 단위로 처리"""
    if os.path.exists(CACHE_FILE_PATH):
        print(f"📂 과거 주식 데이터를 불러오는 중...")
        print(f"   📅 분석 기간: {start_date} ~ {end_date}")
        
        # parquet 파일에서 청크 단위로 읽기 (메모리 최적화)
        print(f"   🔍 데이터베이스에서 분석 기간({start_date} ~ {end_date}) 데이터를 확인하고 있습니다...")
        
        try:
            import pyarrow.parquet as pq
            
            # parquet 파일의 메타데이터 확인
            parquet_file = pq.ParquetFile(CACHE_FILE_PATH)
            total_rows = parquet_file.metadata.num_rows
            print(f"   📊 총 {total_rows:,}개 레코드를 확인했습니다")
            
            # 청크 단위로 읽기
            chunk_size = 100000
            chunks = []
            chunk_count = 0
            
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_count += 1
                chunk = batch.to_pandas()
                print(f"   🔍 데이터베이스에서 {chunk_count}번째 데이터 블록을 확인하고 있습니다... ({len(chunk):,}개 레코드)")
                
                # 날짜 필터링
                chunk['date'] = pd.to_datetime(chunk['date'])
                mask = (chunk['date'] >= pd.to_datetime(start_date)) & (chunk['date'] <= pd.to_datetime(end_date))
                filtered_chunk = chunk[mask]
                
                if not filtered_chunk.empty:
                    chunks.append(filtered_chunk)
                    print(f"   ✅ 분석 기간에 해당하는 {len(filtered_chunk):,}개 데이터를 찾았습니다")
            
            if chunks:
                result_df = pd.concat(chunks, ignore_index=True)
                print(f"✅ 과거 데이터 로딩 완료! 총 {len(result_df):,}개의 주식 데이터를 준비했습니다")
                return result_df
            else:
                print("⚠️ 해당 기간의 데이터가 없습니다.")
                return pd.DataFrame()
                
        except ImportError:
            # pyarrow가 없는 경우 전체 데이터를 읽어서 필터링
            print("   ⚠️ pyarrow가 없어서 전체 데이터를 로드합니다...")
            df = pd.read_parquet(CACHE_FILE_PATH)
            print(f"   📊 총 {len(df):,}개 레코드를 로드했습니다")
            
            # 날짜 필터링
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            filtered_df = df[mask]
            
            if not filtered_df.empty:
                print(f"   ✅ 분석 기간에 해당하는 {len(filtered_df):,}개 데이터를 찾았습니다")
                print(f"✅ 과거 데이터 로딩 완료! 총 {len(filtered_df):,}개의 주식 데이터를 준비했습니다")
                return filtered_df
            else:
                print("⚠️ 해당 기간의 데이터가 없습니다.")
                return pd.DataFrame()
    else:
        print(f"⚠️ 주식 데이터베이스가 없습니다. 처음부터 데이터를 수집합니다...")
        print(f"   📅 수집 기간: {start_date} ~ {CACHE_END_DATE}")
        print("   ⏳ 이 작업은 시간이 오래 걸릴 수 있습니다. 잠시만 기다려주세요...")
        historical_df = _fetch_and_prepare_data(start_date, CACHE_END_DATE)
        historical_df.to_parquet(CACHE_FILE_PATH)
        print(f"✅ 주식 데이터베이스 구축 완료! ({len(historical_df):,}개 데이터 저장)")
        return historical_df

def _get_fresh_data_chunked(start_date, end_date, is_today_analysis=False):
    """최신 데이터를 청크 단위로 처리"""
    fresh_start_date = (datetime.strptime(CACHE_END_DATE, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # 오늘 날짜 분석이거나 캐시 종료일 이후인 경우 실시간 데이터 수집
    if is_today_analysis or datetime.strptime(end_date, '%Y-%m-%d').date() > datetime.strptime(CACHE_END_DATE, '%Y-%m-%d').date():
        if is_today_analysis:
            print(f"\n🔄 오늘 날짜 분석: 실시간 주식 데이터를 수집하는 중...")
        else:
            print(f"\n🔄 최신 주식 데이터를 수집하는 중...")
        print(f"   📅 수집 기간: {fresh_start_date} ~ {end_date}")
        print("   🌐 인터넷에서 최신 거래 정보를 가져오고 있습니다...")
        
        fresh_start_date_with_warmup = (pd.to_datetime(fresh_start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
        fresh_df = _fetch_and_prepare_data(fresh_start_date_with_warmup, end_date)
        
        if not fresh_df.empty:
            fresh_df = fresh_df[fresh_df['date'] >= pd.to_datetime(fresh_start_date)].copy()
            print(f"✅ 최신 데이터 수집 완료! {len(fresh_df):,}개의 최신 주식 데이터를 가져왔습니다")
            return fresh_df
        else:
            print("⚠️ 최신 데이터를 가져올 수 없습니다.")
    else:
        print("ℹ️ 최신 데이터 수집이 필요하지 않습니다. (기존 데이터로 충분)")
    
    return pd.DataFrame()