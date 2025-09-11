# scripts/build_db_pykrx.py

import pandas as pd
import numpy as np
from pykrx import stock
from tqdm import tqdm
import time
import os
from datetime import datetime
import sys
import io
from logger import log_info, log_warning, log_error

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# --- ✨ 핵심 수정: 데이터 수집 시작 기간 변경 ✨ ---
START_DATE = "20150101"
END_DATE = datetime.now().strftime("%Y%m%d")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'financial_data_pykrx_pit.parquet')

def build_point_in_time_db_pykrx():
    """
    pykrx를 사용하여 시점(Point-in-Time) 재무 지표 데이터베이스를 구축합니다.
    - 자본잠식 기업(PBR <= 0)은 리스크 관리를 위해 제외합니다.
    - 적자 기업(PER < 0)은 분석을 위해 데이터에 포함시킵니다.
    """
    log_info("=" * 60)
    log_info("🏗️ pykrx를 이용한 시점(Point-in-Time) 재무 지표 데이터베이스 구축 시작")
    log_info(f"📅 대상 기간: {START_DATE} ~ {END_DATE}")
    log_info("=" * 60)

    try:
        log_info("📋 1단계: 전체 수집 대상 거래일 목록을 조회합니다...")
        trading_days = pd.to_datetime(stock.get_market_ohlcv(START_DATE, END_DATE, "005930").index).strftime('%Y%m%d').tolist()
        log_info(f"   ✅ 총 {len(trading_days):,}개 거래일에 대한 데이터 수집을 시작합니다.")
    except Exception as e:
        log_error(f"거래일 목록을 가져오는 데 실패했습니다: {e}")
        return

    all_daily_data = []

    log_info("💰 2단계: 주식 재무 정보를 수집합니다...")
    log_info(f"   📅 총 {len(trading_days):,}개 거래일의 재무 데이터를 수집합니다")
    log_info(f"   📊 수집 기간: {trading_days[0]} ~ {trading_days[-1]}")
    log_info(f"   💰 각 종목의 PER, PBR, EPS, BPS 등 재무 지표를 가져옵니다")
    log_info("   ⏱️ 예상 소요 시간: 약 30-60분 (API 응답 속도에 따라 달라질 수 있습니다)")
    print()
    
    # 간단한 진행률 표시 (개행 문제 해결)
    completed_days = 0
    total_days = len(trading_days)
    
    print(f"재무 데이터 수집 진행률: 0% (0/{total_days:,})", end='', flush=True)
    
    for i, day in enumerate(trading_days, 1):
        try:
            # 같은 줄에서 진행률 업데이트
            progress_percent = (completed_days / total_days) * 100
            print(f"\r재무 데이터 수집 진행률: {progress_percent:.1f}% ({completed_days:,}/{total_days:,})", end='', flush=True)
            
            df_fundamental = stock.get_market_fundamental(day, market="ALL")

            if df_fundamental.empty:
                completed_days += 1
                time.sleep(0.1)
                continue
            
            if 'PBR' in df_fundamental.columns:
                valid_count_before = len(df_fundamental)
                df_fundamental = df_fundamental[df_fundamental['PBR'] > 0]
                
                if not df_fundamental.empty:
                    df_fundamental.reset_index(inplace=True)
                    df_fundamental.rename(columns={'티커': '종목코드'}, inplace=True)
                    df_fundamental['date'] = pd.to_datetime(day, format='%Y%m%d')
                    all_daily_data.append(df_fundamental)

            completed_days += 1
            time.sleep(0.1)
        except Exception as e:
            completed_days += 1
            continue
    
    # 완료 후 개행
    print()  # 개행 추가
            
    if not all_daily_data:
        print("\n❌ 수집된 데이터가 전혀 없습니다. 라이브러리 또는 KRX 서버 상태를 확인해주세요.")
        return

    print(f"\n3. 수집된 재무 데이터를 정리하고 저장합니다...")
    print(f"   📊 총 {len(all_daily_data):,}개 거래일의 데이터를 하나로 합치는 중...")
    
    final_df = pd.concat(all_daily_data, ignore_index=True)
    print(f"   ✅ 데이터 병합 완료! 총 {len(final_df):,}개의 재무 데이터 레코드")
    
    log_info("🔧 3단계: 재무 지표를 정리하고 추가 계산을 수행합니다...")
    required_cols = ['date', '종목코드', 'PBR', 'PER', 'EPS', 'BPS', 'DIV', 'DPS']
    final_df = final_df[[col for col in required_cols if col in final_df.columns]]
    
    if 'PBR' in final_df.columns and 'PER' in final_df.columns:
        final_df['ROE'] = np.where(final_df['PER'] != 0, final_df['PBR'] / final_df['PER'], np.nan)
        log_info("   ✅ ROE(자기자본이익률) 계산 완료")
    
    final_df.sort_values(by=['date', '종목코드'], inplace=True)
    log_info("   ✅ 데이터 정렬 완료")
    
    log_info("💾 4단계: 주식 재무 데이터베이스를 파일로 저장하는 중...")
    os.makedirs(DATA_DIR, exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH, index=False)
    
    unique_stocks_count = len(final_df['종목코드'].unique())
    log_info("\n" + "="*60)
    log_info("🎉 주식 재무 데이터베이스 구축이 완료되었습니다!")
    log_info("="*60)
    log_info(f"📁 저장 위치: {OUTPUT_PATH}")
    log_info(f"📊 총 데이터 개수: {len(final_df):,}개")
    log_info(f"🏢 포함된 종목 수: {unique_stocks_count:,}개")
    log_info(f"📅 데이터 기간: {final_df['date'].min().strftime('%Y-%m-%d')} ~ {final_df['date'].max().strftime('%Y-%m-%d')}")
    log_info(f"⏱️  수집한 거래일: {len(trading_days):,}일")
    log_info("="*60)

if __name__ == "__main__":
    build_point_in_time_db_pykrx()