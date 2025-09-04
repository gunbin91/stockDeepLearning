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

# --- 기본 설정 ---
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# --- 데이터 수집 기간 및 경로 설정 ---
START_DATE = "20180101"
END_DATE = datetime.now().strftime("%Y%m%d")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_PATH = os.path.join(DATA_DIR, 'financial_data_pykrx_pit.parquet')


def build_point_in_time_db_pykrx():
    """
    pykrx를 사용하여 시점(Point-in-Time) 재무 지표 데이터베이스를 구축합니다.
    - 공식 문서에 명시된 정확한 함수를 사용하여 데이터 수집 문제를 해결했습니다.
    - 자본잠식 기업(PBR <= 0)은 리스크 관리를 위해 제외합니다.
    - 적자 기업(PER < 0)은 분석을 위해 데이터에 포함시킵니다.
    """
    print("=" * 60)
    print("pykrx를 이용한 시점(Point-in-Time) 재무 지표 데이터베이스 구축 시작")
    print(f"대상 기간: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    try:
        print("\n1. 전체 수집 대상 거래일 목록을 조회합니다...")
        trading_days = pd.to_datetime(stock.get_market_ohlcv(START_DATE, END_DATE, "005930").index).strftime('%Y%m%d').tolist()
        print(f"  - 총 {len(trading_days)} 거래일에 대한 데이터 수집을 시작합니다.")
    except Exception as e:
        print(f"  [오류] 거래일 목록을 가져오는 데 실패했습니다: {e}")
        return

    all_daily_data = []

    print("\n2. 일별 재무 지표(PER, PBR 등)를 순차적으로 수집합니다...")
    for day in tqdm(trading_days, desc="일별 데이터 수집 진행률"):
        try:
            # <<< ✨ 핵심 수정: 공식 문서에 명시된 정확한 함수 사용 ✨ >>>
            # 리드미 2.1.1.5 항목의 get_market_fundamental 함수를 사용합니다.
            df_fundamental = stock.get_market_fundamental(day, market="ALL")

            if df_fundamental.empty:
                tqdm.write(f"  - [{day}] 정보: 해당일에 조회된 펀더멘털 데이터가 없습니다. (건너뜀)")
                time.sleep(0.1)
                continue
            
            # [DEBUG] 수신된 컬럼 목록을 확인하여 데이터 무결성 검사
            # tqdm.write(f"  - [{day}] [DEBUG] 수신된 컬럼: {df_fundamental.columns.tolist()}")

            # 'PBR' 컬럼이 존재하는지 명시적으로 확인하여 KeyError 방지
            if 'PBR' in df_fundamental.columns:
                # PBR이 0보다 큰, 즉 자본잠식이 아닌 기업만 필터링
                valid_count_before = len(df_fundamental)
                df_fundamental = df_fundamental[df_fundamental['PBR'] > 0]
                
                if not df_fundamental.empty:
                    df_fundamental.reset_index(inplace=True)
                    df_fundamental.rename(columns={'티커': '종목코드'}, inplace=True)
                    df_fundamental['date'] = pd.to_datetime(day, format='%Y%m%d')
                    all_daily_data.append(df_fundamental)
                    
                    tqdm.write(f"  - [{day}] 수집 완료: 유효 PBR 종목 {len(df_fundamental)}개 확보 (원본 {valid_count_before}개, 적자기업 포함)")
            else:
                tqdm.write(f"  - [{day}] 오류: 수신된 데이터에 'PBR' 컬럼이 없습니다. (해당일 건너뜀)")

            time.sleep(0.1)
        except Exception as e:
            tqdm.write(f"  - [{day}] 처리 중 예상치 못한 오류 발생: {e} (해당일 건너뜀)")
            continue
            
    if not all_daily_data:
        print("\n[오류] 수집된 데이터가 전혀 없습니다. 라이브러리 또는 KRX 서버 상태를 확인해주세요.")
        return

    print("\n3. 전체 수집 데이터 병합 및 최종 후처리를 시작합니다...")
    final_df = pd.concat(all_daily_data, ignore_index=True)
    
    required_cols = ['date', '종목코드', 'PBR', 'PER', 'EPS', 'BPS', 'DIV', 'DPS']
    final_df = final_df[[col for col in required_cols if col in final_df.columns]]
    
    if 'PBR' in final_df.columns and 'PER' in final_df.columns:
        final_df['ROE'] = np.where(final_df['PER'] != 0, final_df['PBR'] / final_df['PER'], np.nan)
    
    final_df.sort_values(by=['date', '종목코드'], inplace=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    final_df.to_parquet(OUTPUT_PATH, index=False)
    
    unique_stocks_count = len(final_df['종목코드'].unique())
    print("-" * 60)
    print(f"✅ pykrx 재무 지표 데이터베이스 구축 완료!")
    print(f"  - 총 {len(final_df):,}개의 레코드가 생성되었습니다.")
    print(f"  - 총 {unique_stocks_count:,}개 종목의 데이터가 최소 1회 이상 수집되었습니다.")
    print(f"  - 저장 경로: {OUTPUT_PATH}")
    print("-" * 60)

if __name__ == "__main__":
    build_point_in_time_db_pykrx()