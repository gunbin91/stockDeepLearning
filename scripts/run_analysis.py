import pandas as pd
from datetime import datetime
import argparse
import json
import os
import sys
import io
import numpy as np # numpy 임포트 추가

# 크로스 플랫폼 인코딩 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import encoding_utils  # 인코딩 유틸리티 임포트

# 추가 인코딩 설정 (한글 깨짐 방지)
import locale
import codecs

# 시스템 인코딩을 UTF-8로 강제 설정
try:
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Korean_Korea.utf8')
    except:
        pass

# stdout/stderr를 UTF-8로 설정
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 환경 변수 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'ko_KR.UTF-8'

# 내부 모듈 임포트
import data_fetcher
import scoring
import ml_model
import ensemble
from logger import log_info, log_warning, log_error, log_critical
from exceptions import DataFetchError, ModelPredictionError, AnalysisError

# 결과물을 저장할 캐시 디렉토리 생성
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def run_analysis(analysis_date_str):
    """주어진 날짜를 기준으로 주식 데이터를 분석하고 결과를 파일에 저장합니다."""
    try:
        log_info(f"분석 시작 (기준일: {analysis_date_str})")
        analysis_date = datetime.strptime(analysis_date_str, '%Y-%m-%d')

        log_info("전체 종목 목록 수신 중...")
        try:
            stock_list_df = data_fetcher.fetch_stock_list()
            if stock_list_df.empty:
                error_msg = "종목 목록을 가져오는 데 실패했습니다."
                log_error(error_msg)
                raise AnalysisError(error_msg, step="stock_list_fetch")
            log_info(f"{len(stock_list_df)}개 종목 목록 수신 완료.")
        except DataFetchError as e:
            log_error(f"종목 목록 수신 실패: {e}")
            raise AnalysisError(f"종목 목록 수신 실패: {e.message}", step="stock_list_fetch")

        log_info("재무/가격 데이터 수집 및 기술적 지표 계산 중... (시간이 다소 소요될 수 있습니다)")
        try:
            feature_df, actual_analysis_date = data_fetcher.fetch_all_data(stock_list_df, analysis_date)
        except DataFetchError as e:
            log_error(f"데이터 수집 실패: {e}")
            raise AnalysisError(f"데이터 수집 실패: {e.message}", step="data_fetch")
        
        # <<< ✨ 핵심 수정: JSON 저장 전 NaN 값을 None으로 명시적 변환하여 데이터 유실 방지 ✨ >>>
        feature_df_for_json = feature_df.copy()
        # pandas 2.0 이상에서는 replace(np.nan, None)이 권장되지 않으므로, where/mask를 사용
        feature_df_for_json = feature_df_for_json.where(pd.notna(feature_df_for_json), None)
        
        if '종목코드' in feature_df_for_json.columns:
            feature_df_for_json['종목코드'] = feature_df_for_json['종목코드'].astype(str)
            
        feature_df_path = os.path.join(CACHE_DIR, 'cached_features.json')
        feature_df_for_json.to_json(feature_df_path, orient='records', force_ascii=False, indent=4)
        log_info(f"피처 데이터를 '{feature_df_path}'에 저장했습니다.")
        
        if feature_df.empty:
            error_msg = "데이터 수집에 실패했습니다."
            log_error(error_msg)
            raise AnalysisError(error_msg, step="data_processing")

        actual_date_str = actual_analysis_date.strftime('%Y-%m-%d')
        log_info(f"   📅 실제 분석 기준일: {actual_date_str}")

        # 시장 현황 데이터 저장
        log_info("📊 시장 현황 데이터 처리 중...")
        macro_cols = ['KOSPI', 'KOSPI_pct_1d', 'USDKRW', 'USDKRW_pct_1d', 'VIX', 'VIX_pct_1d']
        market_condition = {}
        if all(col in feature_df.columns for col in macro_cols):
            market_condition = feature_df.iloc[0][macro_cols].to_dict()
            log_info("   ✅ 거시경제 지표 데이터 추출 완료")
        else:
            log_warning("   ⚠️ 일부 거시경제 지표 데이터가 누락되었습니다.")
        
        market_condition_path = os.path.join(CACHE_DIR, 'market_condition.json')
        with open(market_condition_path, 'w', encoding='utf-8') as f:
            json.dump(market_condition, f, ensure_ascii=False, indent=4)
        log_info(f"   💾 시장 현황 데이터를 '{market_condition_path}'에 저장했습니다.")

        log_info("📊 팩터별 점수 계산을 시작합니다...")
        scored_df = scoring.calculate_factor_scores(feature_df)

        log_info("머신러닝 모델 예측 중...")
        try:
            ml_predicted_df = ml_model.predict_with_ml_model(feature_df)
            if ml_predicted_df is None:
                error_msg = "머신러닝 모델 예측에 실패했습니다."
                log_error(error_msg)
                raise AnalysisError(error_msg, step="ml_prediction")
        except ModelPredictionError as e:
            log_error(f"머신러닝 모델 예측 실패: {e}")
            raise AnalysisError(f"머신러닝 모델 예측 실패: {e.message}", step="ml_prediction")

        log_info("🎯 앙상블 최종 점수 계산을 시작합니다...")
        merged_df = pd.merge(scored_df, ml_predicted_df, on='종목코드', how='left')
        log_info(f"   📊 머신러닝 예측 결과 병합 완료: {len(merged_df):,}개 종목")
        
        # 딥러닝 모델 및 감정 분석 점수는 기본값으로 설정
        merged_df['dl_trend_score(더미)'] = 0
        merged_df['sentiment_score'] = 0
        log_info("   ✅ 딥러닝 모델 및 감정 분석 점수 기본값 적용 완료")
        
        final_ranked_df = ensemble.calculate_final_score(merged_df)

        # 최종 결과에 종목명, 현재가 등 추가 정보 병합
        log_info("📋 최종 결과 데이터 정리 중...")
        if '종목코드' not in final_ranked_df.columns:
            final_ranked_df.reset_index(inplace=True)
        final_df_with_names = pd.merge(final_ranked_df, stock_list_df[['종목코드', '종목명']].drop_duplicates(), on='종목코드', how='left')
        log_info(f"   📊 종목명 병합 완료: {len(final_df_with_names):,}개 종목")
        
        # [FIX] 병합 시 종목명 컬럼 충돌 해결 로직 추가
        if '종목명_x' in final_df_with_names.columns:
            final_df_with_names['종목명'] = final_df_with_names['종목명_y'].fillna(final_df_with_names['종목명_x'])
            final_df_with_names.drop(columns=['종목명_x', '종목명_y'], inplace=True)

        log_info("   💰 가격 정보 병합 중...")
        price_map = feature_df.set_index('종목코드')[['현재가', '기준일가']].to_dict(orient='index')
        final_df_with_names['현재가'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('현재가'))
        final_df_with_names['기준일가'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('기준일가'))

        # [FIX] 최종 데이터프레임에 분석 기준일 컬럼 추가
        final_df_with_names['date'] = actual_analysis_date

        # 결과 파일로 저장
        log_info("💾 최종 분석 결과 저장 중...")
        result_path = os.path.join(CACHE_DIR, 'analysis_result.json')
        # 날짜 필드를 문자열로 변환 (JSON 직렬화 위함)
        final_df_with_names['date'] = final_df_with_names['date'].dt.strftime('%Y-%m-%d')
        final_df_with_names.to_json(result_path, orient='records', force_ascii=False, indent=4)
        log_info(f"   ✅ 최종 분석 결과를 '{result_path}'에 저장했습니다.")
        
        # 최종 결과 통계
        top_10_count = len(final_df_with_names[final_df_with_names['최종순위'] <= 10])
        avg_score = final_df_with_names['final_score'].mean()
        log_info(f"   📈 분석 결과: 상위 10위 종목 {top_10_count}개, 평균 점수 {avg_score:.2f}")

        log_info("🎉 주식 분석이 성공적으로 완료되었습니다!")

    except AnalysisError as e:
        log_error(f"분석 프로세스 중 오류 발생: {e}")
        print(f"오류: {e.message}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        log_critical(f"분석 프로세스 중 예상치 못한 예외 발생: {e}")
        print(f"오류: 분석 프로세스 중 예외 발생 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='주식 분석 스크립트')
    parser.add_argument('--date', type=str, required=True, help='분석 기준일 (YYYY-MM-DD)')
    args = parser.parse_args()
    
    run_analysis(args.date)