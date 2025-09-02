
import pandas as pd
from datetime import datetime
import argparse
import json
import os
import sys
import io

# stdout/stderr를 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


# 프로젝트 루트 경로를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 내부 모듈 임포트
import data_fetcher
import scoring
import ml_model
import dl_model
import ensemble

# 결과물을 저장할 캐시 디렉토리 생성
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def run_analysis(analysis_date_str):
    """주어진 날짜를 기준으로 주식 데이터를 분석하고 결과를 파일에 저장합니다."""
    try:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 1. 분석 시작 (기준일: {analysis_date_str})")
        analysis_date = datetime.strptime(analysis_date_str, '%Y-%m-%d')

        print("  - 전체 종목 목록 수신 중...")
        stock_list_df = data_fetcher.fetch_stock_list()
        if stock_list_df.empty:
            print("오류: 종목 목록을 가져오는 데 실패했습니다.")
            return
        print(f"  - {len(stock_list_df)}개 종목 목록 수신 완료.")

        print("  - 재무/가격 데이터 수집 및 기술적 지표 계산 중... (시간이 다소 소요될 수 있습니다)")
        feature_df, actual_analysis_date = data_fetcher.fetch_all_data(stock_list_df, analysis_date)
        
        if feature_df.empty:
            print("오류: 데이터 수집에 실패했습니다.")
            return

        actual_date_str = actual_analysis_date.strftime('%Y-%m-%d')
        print(f"  - 실제 분석 기준일: {actual_date_str}")

        # 시장 현황 데이터 저장
        print("  - 시장 현황 데이터 처리 중...")
        macro_cols = ['KOSPI', 'KOSPI_pct_1d', 'USDKRW', 'USDKRW_pct_1d', 'VIX', 'VIX_pct_1d']
        market_condition = {}
        if all(col in feature_df.columns for col in macro_cols):
            market_condition = feature_df.iloc[0][macro_cols].to_dict()
        
        market_condition_path = os.path.join(CACHE_DIR, 'market_condition.json')
        with open(market_condition_path, 'w', encoding='utf-8') as f:
            json.dump(market_condition, f, ensure_ascii=False, indent=4)
        print(f"  - 시장 현황 데이터를 '{market_condition_path}'에 저장했습니다.")

        print("  - 팩터 점수 계산 중...")
        scored_df = scoring.calculate_factor_scores(feature_df)

        print("  - 머신러닝 모델 예측 중...")
        ml_predicted_df = ml_model.predict_with_ml_model(feature_df)
        if ml_predicted_df is None:
            print("오류: 머신러닝 모델 예측에 실패했습니다.")
            return

        print("  - 앙상블 최종 점수 계산 중...")
        merged_df = pd.merge(scored_df, ml_predicted_df, on='종목코드', how='left')
        # dl_model 및 nlp 모듈은 현재 구현에서 제외됨 (app.py 로직 기반)
        dl_predicted_df = dl_model.predict_with_deep_learning(merged_df)
        nlp_analyzed_df = dl_predicted_df.copy()
        nlp_analyzed_df['sentiment_score'] = 0
        final_ranked_df = ensemble.calculate_final_score(nlp_analyzed_df)

        # 최종 결과에 종목명, 현재가 등 추가 정보 병합
        if '종목코드' not in final_ranked_df.columns:
            final_ranked_df.reset_index(inplace=True)
        final_df_with_names = pd.merge(final_ranked_df, stock_list_df[['종목코드', '종목명']].drop_duplicates(), on='종목코드', how='left')
        
        # [FIX] 병합 시 종목명 컬럼 충돌 해결 로직 추가
        if '종목명_x' in final_df_with_names.columns:
            final_df_with_names['종목명'] = final_df_with_names['종목명_y'].fillna(final_df_with_names['종목명_x'])
            final_df_with_names.drop(columns=['종목명_x', '종목명_y'], inplace=True)

        price_map = feature_df.set_index('종목코드')[['현재가', '기준일가']].to_dict(orient='index')
        final_df_with_names['현재가'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('현재가'))
        final_df_with_names['기준일가'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('기준일가'))

        # [FIX] 최종 데이터프레임에 분석 기준일 컬럼 추가
        final_df_with_names['date'] = actual_analysis_date

        # 결과 파일로 저장
        result_path = os.path.join(CACHE_DIR, 'analysis_result.json')
        # 날짜 필드를 문자열로 변환 (JSON 직렬화 위함)
        final_df_with_names['date'] = final_df_with_names['date'].dt.strftime('%Y-%m-%d')
        final_df_with_names.to_json(result_path, orient='records', force_ascii=False, indent=4)
        print(f"  - 최종 분석 결과를 '{result_path}'에 저장했습니다.")

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 2. 분석 성공적으로 완료.")

    except Exception as e:
        print(f"오류: 분석 프로세스 중 예외 발생 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='주식 분석 스크립트')
    parser.add_argument('--date', type=str, required=True, help='분석 기준일 (YYYY-MM-DD)')
    args = parser.parse_args()
    
    run_analysis(args.date)
