"""
주식 분석 실행 스크립트
====================

이 파일은 주식 분석의 전체 프로세스를 실행하는 메인 스크립트입니다.
데이터 수집부터 최종 결과 저장까지 모든 과정을 자동화합니다.

주요 기능:
- 종목 데이터 수집 및 전처리
- 팩터 점수 계산
- 머신러닝 예측
- 앙상블 점수 계산
- 결과 저장 및 보고서 생성
"""

import pandas as pd
from datetime import datetime
import argparse
import json
import os
import sys
import time
import warnings

# yfinance의 pandas deprecated API 경고 무시 (yfinance 라이브러리 자체 문제)
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)

# 크로스 플랫폼 인코딩 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 추가 인코딩 설정 (한글 깨짐 방지)
import locale

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
from logger import (log_info, log_warning, log_error, log_critical, log_step, log_success, log_start, log_complete,
                   start_analysis_report, log_data_collection_status, log_processing_status, log_final_results,
                   log_performance_info, log_saved_files, complete_analysis_report)
from exceptions import DataFetchError, ModelPredictionError, AnalysisError

from path_manager import path_manager

def _load_optimal_weights_safe() -> dict:
    """
    가중치 파일을 안전하게 로드합니다.
    - 파일이 없거나 깨져 있어도 기존 동작(모두 계산)을 유지하기 위해 기본값을 반환합니다.
    """
    # 기본값: 기존과 동일하게 모두 계산
    weights = {
        "ml_pred_proba": 0.50,
        "lgbm_pred_proba": 0.50,
        "catboost_pred_proba": 0.0,  # 기본값은 0 (옵션)
    }
    try:
        weights_path = str(path_manager.get_weights_path())
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                # volatility_score는 제거되었으므로 필터링
                loaded = {k: v for k, v in loaded.items() if k != 'volatility_score'}
                # 키 호환: lgb_pred_proba가 있을 수 있음
                if "ml_pred_proba" in loaded:
                    weights["ml_pred_proba"] = float(loaded["ml_pred_proba"])
                if "lgbm_pred_proba" in loaded:
                    weights["lgbm_pred_proba"] = float(loaded["lgbm_pred_proba"])
                elif "lgb_pred_proba" in loaded:
                    weights["lgbm_pred_proba"] = float(loaded["lgb_pred_proba"])
                # CatBoost 가중치 로드
                if "catboost_pred_proba" in loaded:
                    weights["catboost_pred_proba"] = float(loaded["catboost_pred_proba"])
    except Exception as e:
        log_warning(f"가중치 파일 로드 실패(기본값 사용): {e}")
    return weights

def run_analysis(analysis_date_str):
    """
    주식 분석 메인 실행 함수
    
    지정된 날짜를 기준으로 전체 주식 분석 프로세스를 실행합니다.
    데이터 수집, 전처리, 팩터 계산, ML 예측, 앙상블 점수 계산을 순차적으로 수행합니다.
    
    Args:
        analysis_date_str: 분석 기준일 (YYYY-MM-DD 형식)
        
    Returns:
        bool: 분석 성공 여부
    """
    start_time = time.time()
    data_start_time = None
    analysis_start_time = None
    
    try:
        # 분석 시작 보고서 헤더 (기존 로그와 함께)
        start_analysis_report(analysis_date_str)
        log_start(f"주식 분석 시작 (기준일: {analysis_date_str})")
        analysis_date = datetime.strptime(analysis_date_str, '%Y-%m-%d')

        # 1단계: 종목 목록 수집
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

        # 2단계: 데이터 수집 및 기술적 지표 계산
        log_info("재무/가격 데이터 수집 및 기술적 지표 계산 중... (시간이 다소 소요될 수 있습니다)")
        data_start_time = time.time()
        try:
            # 메모리 정리
            import gc
            gc.collect()
            
            # 주식추천 페이지에서는 캐시를 사용하지 않고 실시간 데이터 수집
            feature_df, actual_analysis_date = data_fetcher.fetch_all_data(stock_list_df, analysis_date, use_cache=False)
            
            # 데이터 수집 후 메모리 정리
            gc.collect()
            log_info("   🧹 데이터 수집 후 메모리 정리 완료")
            
        except DataFetchError as e:
            log_error(f"데이터 수집 실패: {e}")
            raise AnalysisError(f"데이터 수집 실패: {e.message}", step="data_fetch")
        
        # JSON 저장 전 NaN 값을 None으로 명시적 변환하여 데이터 유실 방지
        feature_df_for_json = feature_df.copy()
        # pandas 2.0 이상에서는 replace(np.nan, None)이 권장되지 않으므로, where/mask를 사용
        feature_df_for_json = feature_df_for_json.where(pd.notna(feature_df_for_json), None)
        
        if '종목코드' in feature_df_for_json.columns:
            # NASDAQ 티커는 그대로 사용
            feature_df_for_json['종목코드'] = feature_df_for_json['종목코드'].astype(str).str.strip()
            
        feature_df_path = os.path.join(str(path_manager.data_dir), 'cached_features.json')
        feature_df_for_json.to_json(feature_df_path, orient='records', force_ascii=False, indent=4)
        log_info(f"피처 데이터를 '{feature_df_path}'에 저장했습니다.")
        
        if feature_df.empty:
            error_msg = "데이터 수집에 실패했습니다."
            log_error(error_msg)
            raise AnalysisError(error_msg, step="data_processing")

        # 가중치 로드 (가중치 0인 항목은 계산을 스킵)
        weights = _load_optimal_weights_safe()
        do_rf = weights.get("ml_pred_proba", 0) > 0
        do_lgbm = weights.get("lgbm_pred_proba", 0) > 0
        do_catboost = weights.get("catboost_pred_proba", 0) > 0
        log_info("⚙️ 가중치 기반 계산 스킵 설정", context={
            "ml_pred_proba": weights.get("ml_pred_proba"),
            "lgbm_pred_proba": weights.get("lgbm_pred_proba"),
            "catboost_pred_proba": weights.get("catboost_pred_proba"),
            "do_rf": do_rf,
            "do_lgbm": do_lgbm,
            "do_catboost": do_catboost
        })

        actual_date_str = actual_analysis_date.strftime('%Y-%m-%d')
        log_info(f"   📅 실제 분석 기준일: {actual_date_str}")

        # 시장 현황 데이터 저장
        log_info("📊 시장 현황 데이터 처리 중...")
        macro_cols = ['IXIC', 'IXIC_pct_1d', 'IXIC_disparity_20', 'IXIC_MA20_Slope', 'VIX']
        market_condition = {}
        if all(col in feature_df.columns for col in macro_cols):
            market_condition = feature_df.iloc[0][macro_cols].to_dict()
            log_info("   ✅ 거시경제 지표 데이터 추출 완료")
        else:
            log_warning("   ⚠️ 일부 거시경제 지표 데이터가 누락되었습니다.")
        
        market_condition_path = os.path.join(str(path_manager.data_dir), 'market_condition.json')
        with open(market_condition_path, 'w', encoding='utf-8') as f:
            json.dump(market_condition, f, ensure_ascii=False, indent=4)
        log_info(f"   💾 시장 현황 데이터를 '{market_condition_path}'에 저장했습니다.")

        # 팩터 점수 계산 (향후 다른 팩터 추가 대비)
        log_info("📊 팩터별 점수 계산을 시작합니다...")
        try:
            import gc
            gc.collect()

            # 향후 다른 팩터 추가 시 calculate_factor_scores에서 처리
            scored_df = scoring.calculate_factor_scores(feature_df)

            if scored_df is None or scored_df.empty:
                error_msg = "팩터 점수 계산 결과가 비어있습니다."
                log_error(error_msg)
                raise AnalysisError(error_msg, step="factor_scoring")

            log_info(f"   ✅ 팩터 점수 계산 완료: {len(scored_df):,}개 종목")
            gc.collect()
        except Exception as e:
            log_error(f"팩터 점수 계산 중 오류: {e}")
            raise AnalysisError(f"팩터 점수 계산 중 오류: {e}", step="factor_scoring")

        log_info("머신러닝 모델 예측 중...")
        try:
            # 메모리 정리
            import gc
            gc.collect()
            log_info("   🧹 메모리 정리 완료")
            
            # 모델 예측 시도 (RF) - 가중치가 0이면 스킵
            if do_rf:
                log_info("   🤖 [RF] 모델 로딩 및 예측 중...")
                try:
                    ml_predicted_df = ml_model.predict_with_ml_model(feature_df)
                except Exception as predict_error:
                    error_type = type(predict_error).__name__
                    error_msg = str(predict_error)
                    log_error(f"   ❌ [RF] 모델 예측 실패: {error_type}: {error_msg}")
                    raise
            else:
                log_info("   ⏭️ [RF] 가중치가 0이라 예측을 건너뜁니다.")
                ml_predicted_df = feature_df[['종목코드']].copy()
                ml_predicted_df['ml_pred_proba'] = float('nan')
            
            if ml_predicted_df is None or ml_predicted_df.empty:
                error_msg = "[RF] 모델 예측 결과가 비어있습니다."
                log_error(f"   ❌ {error_msg}")
                raise AnalysisError(error_msg, step="ml_prediction_rf")

            log_info(f"   ✅ [RF] 예측 완료: {len(ml_predicted_df):,}개 종목")

            # 모델 예측 시도 (LGBM) - 가중치가 0이면 스킵
            if do_lgbm:
                log_info("   🤖 [LGBM] 모델 로딩 및 예측 중...")
                try:
                    lgbm_predicted_df = ml_model.predict_with_lgbm_model(feature_df)
                    if not lgbm_predicted_df.empty and 'lgbm_pred_proba' in lgbm_predicted_df.columns:
                        log_info(f"   ✅ [LGBM] 예측 완료: {len(lgbm_predicted_df):,}개 종목")
                    else:
                        log_warning("   ⚠️ [LGBM] 예측 결과가 비어있거나 유효하지 않습니다.")
                except Exception as e:
                    log_warning(f"   ⚠️ [LGBM] 예측 중 오류 (건너뜀): {e}")
                    lgbm_predicted_df = pd.DataFrame()
            else:
                log_info("   ⏭️ [LGBM] 가중치가 0이라 예측을 건너뜁니다.")
                lgbm_predicted_df = feature_df[['종목코드']].copy()
                lgbm_predicted_df['lgbm_pred_proba'] = float('nan')

            # 예측 결과 병합
            # RF 결과에 LGBM 결과 병합
            merged_models = ['RF']
            if lgbm_predicted_df is not None and not lgbm_predicted_df.empty:
                ml_predicted_df = pd.merge(ml_predicted_df, lgbm_predicted_df, on='종목코드', how='left')
                merged_models.append('LGBM')

            # 모델 예측 시도 (CatBoost) - 가중치가 0이면 스킵
            if do_catboost:
                log_info("   🤖 [CatBoost] 모델 로딩 및 예측 중...")
                try:
                    catboost_predicted_df = ml_model.predict_with_catboost_model(feature_df)
                    if not catboost_predicted_df.empty and 'catboost_pred_proba' in catboost_predicted_df.columns:
                        log_info(f"   ✅ [CatBoost] 예측 완료: {len(catboost_predicted_df):,}개 종목")
                    else:
                        log_warning("   ⚠️ [CatBoost] 예측 결과가 비어있거나 유효하지 않습니다.")
                except Exception as e:
                    log_warning(f"   ⚠️ [CatBoost] 예측 중 오류 (건너뜀): {e}")
                    catboost_predicted_df = pd.DataFrame()
            else:
                log_info("   ⏭️ [CatBoost] 가중치가 0이라 예측을 건너뜁니다.")
                catboost_predicted_df = feature_df[['종목코드']].copy()
                catboost_predicted_df['catboost_pred_proba'] = float('nan')

            # CatBoost 예측 결과 병합
            if catboost_predicted_df is not None and not catboost_predicted_df.empty:
                ml_predicted_df = pd.merge(ml_predicted_df, catboost_predicted_df, on='종목코드', how='left')
                merged_models.append('CatBoost')
            
            # 통합 로그 출력
            if len(merged_models) > 1:
                log_info(f"   📊 [통합] {' 및 '.join(merged_models)} 예측 결과 병합 완료")

            # 예측 결과 검증
            if '종목코드' not in ml_predicted_df.columns:
                error_msg = "예측 결과에 '종목코드' 컬럼이 없습니다."
                log_error(f"   ❌ {error_msg}")
                raise AnalysisError(error_msg, step="ml_prediction")
            
            if 'ml_pred_proba' not in ml_predicted_df.columns:
                error_msg = "예측 결과에 'ml_pred_proba' 컬럼이 없습니다."
                log_error(f"   ❌ {error_msg}")
                raise AnalysisError(error_msg, step="ml_prediction")
            # lgbm_pred_proba는 가중치가 0이면 NaN으로 존재하도록 유지(호환성)
            if 'lgbm_pred_proba' not in ml_predicted_df.columns:
                ml_predicted_df['lgbm_pred_proba'] = float('nan')
            
            # catboost_pred_proba는 가중치가 0이면 NaN으로 존재하도록 유지(호환성)
            if 'catboost_pred_proba' not in ml_predicted_df.columns:
                ml_predicted_df['catboost_pred_proba'] = float('nan')
                
            log_info(f"   ✅ ML 모델 예측 완료: {len(ml_predicted_df):,}개 종목")
            
        except ModelPredictionError as e:
            log_error(f"   ❌ 머신러닝 모델 예측 실패: {e}")
            import traceback
            log_error(f"   ❌ 스택 트레이스:\n{traceback.format_exc()}")
            raise AnalysisError(f"머신러닝 모델 예측 실패: {e.message}", step="ml_prediction")
        except MemoryError as e:
            log_error(f"   ❌ 메모리 부족으로 인한 ML 모델 예측 실패: {e}")
            import traceback
            log_error(f"   ❌ 스택 트레이스:\n{traceback.format_exc()}")
            raise AnalysisError(f"메모리 부족으로 인한 ML 모델 예측 실패: {e}", step="ml_prediction")
        except AnalysisError:
            # AnalysisError는 그대로 재발생
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            log_error(f"   ❌ ML 모델 예측 중 예상치 못한 오류: {error_type}: {error_msg}")
            import traceback
            log_error(f"   ❌ 스택 트레이스:\n{traceback.format_exc()}")
            raise AnalysisError(f"ML 모델 예측 중 예상치 못한 오류: {error_type}: {error_msg}", step="ml_prediction")

        log_info("🎯 앙상블 최종 점수 계산을 시작합니다...")
        try:
            # 메모리 정리
            import gc
            gc.collect()
            
            merged_df = pd.merge(scored_df, ml_predicted_df, on='종목코드', how='left')
            log_info(f"   📊 머신러닝 예측 결과 병합 완료: {len(merged_df):,}개 종목")
            
            if merged_df.empty:
                error_msg = "병합된 데이터가 비어있습니다."
                log_error(error_msg)
                raise AnalysisError(error_msg, step="data_merge")
            
            log_info("   ✅ 팩터 점수 계산 완료")
            
            # 앙상블 계산
            log_info("   🎯 앙상블 점수 계산 중...")
            final_ranked_df = ensemble.calculate_final_score(merged_df)
            
            if final_ranked_df is None or final_ranked_df.empty:
                error_msg = "앙상블 점수 계산 결과가 비어있습니다."
                log_error(error_msg)
                raise AnalysisError(error_msg, step="ensemble_calculation")
                
            log_info(f"   ✅ 앙상블 점수 계산 완료: {len(final_ranked_df):,}개 종목")
            
        except Exception as e:
            log_error(f"앙상블 계산 중 오류: {e}")
            raise AnalysisError(f"앙상블 계산 중 오류: {e}", step="ensemble_calculation")

        # 최종 결과에 종목명, 현재가 등 추가 정보 병합
        log_info("📋 최종 결과 데이터 정리 중...")
        try:
            # 메모리 정리
            import gc
            gc.collect()
            
            if '종목코드' not in final_ranked_df.columns:
                final_ranked_df.reset_index(inplace=True)
            
            # NASDAQ 티커는 그대로 사용 (KRX 6자리 패딩 제거)
            final_ranked_df['종목코드'] = final_ranked_df['종목코드'].astype(str).str.strip()
            stock_list_df['종목코드'] = stock_list_df['종목코드'].astype(str).str.strip()
            
            merge_cols = ['종목코드', '종목명']
            if '시장구분' in stock_list_df.columns:
                merge_cols.append('시장구분')
            final_df_with_names = pd.merge(
                final_ranked_df,
                stock_list_df[merge_cols].drop_duplicates(),
                on='종목코드',
                how='left'
            )
            log_info(f"   📊 종목명 병합 완료: {len(final_df_with_names):,}개 종목")
            
            if final_df_with_names.empty:
                error_msg = "최종 결과 데이터가 비어있습니다."
                log_error(error_msg)
                raise AnalysisError(error_msg, step="final_data_merge")
            
            # 병합 시 종목명 컬럼 충돌 해결
            if '종목명_x' in final_df_with_names.columns:
                final_df_with_names['종목명'] = final_df_with_names['종목명_y'].fillna(final_df_with_names['종목명_x'])
                final_df_with_names.drop(columns=['종목명_x', '종목명_y'], inplace=True)

            log_info("   💰 가격 정보 병합 중...")
            # 키 정합성: feature_df/결과df 모두 티커 strip
            feature_df_local = feature_df.copy()
            if '종목코드' in feature_df_local.columns:
                feature_df_local['종목코드'] = feature_df_local['종목코드'].astype(str).str.strip()
            final_df_with_names['종목코드'] = final_df_with_names['종목코드'].astype(str).str.strip()

            price_map = feature_df_local.set_index('종목코드')[['현재가', '기준일가', '전날종가', '시가총액']].to_dict(orient='index')
            final_df_with_names['현재가'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('현재가'))
            final_df_with_names['기준일가'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('기준일가'))
            final_df_with_names['전날종가'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('전날종가'))
            final_df_with_names['시가총액'] = final_df_with_names['종목코드'].map(lambda x: price_map.get(x, {}).get('시가총액'))

            # 최종 데이터프레임에 분석 기준일 컬럼 추가
            final_df_with_names['date'] = actual_analysis_date

            # 결과 파일로 저장
            log_info("💾 최종 분석 결과 저장 중...")
            result_path = os.path.join(str(path_manager.data_dir), 'analysis_result.json')
            
            # 날짜 필드를 문자열로 변환 (JSON 직렬화 위함)
            final_df_with_names['date'] = final_df_with_names['date'].dt.strftime('%Y-%m-%d')
            
            # JSON 저장 시도
            try:
                final_df_with_names.to_json(result_path, orient='records', force_ascii=False, indent=4)
                log_info(f"   ✅ 최종 분석 결과를 '{result_path}'에 저장했습니다.")
            except Exception as e:
                log_error(f"결과 파일 저장 실패: {e}")
                raise AnalysisError(f"결과 파일 저장 실패: {e}", step="file_save")
            
            # 최종 결과 통계
            top_10_count = len(final_df_with_names[final_df_with_names['최종순위'] <= 10])
            avg_score = final_df_with_names['final_score'].mean()
            log_info(f"   📈 분석 결과: 상위 10위 종목 {top_10_count}개, 평균 점수 {avg_score:.2f}")

            log_info("🎉 주식 분석이 성공적으로 완료되었습니다!")
            
        except Exception as e:
            log_error(f"최종 결과 처리 중 오류: {e}")
            raise AnalysisError(f"최종 결과 처리 중 오류: {e}", step="final_processing")

    except AnalysisError as e:
        log_error(f"분석 프로세스 중 오류 발생: {e}", exception=e, context={'function': 'main'})
        # 메모리 정리
        import gc
        gc.collect()
    except Exception as e:
        log_critical(f"분석 프로세스 중 예상치 못한 예외 발생: {e}", exception=e, context={'function': 'main'})
        # 메모리 정리
        import gc
        gc.collect()
    finally:
        # 최종 메모리 정리
        import gc
        gc.collect()
        log_info("🧹 최종 메모리 정리 완료")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='주식 분석 스크립트')
    parser.add_argument('--date', type=str, required=True, help='분석 기준일 (YYYY-MM-DD)')
    args = parser.parse_args()
    
    run_analysis(args.date)