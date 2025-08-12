<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:77cd164b2974899741a2b3c93526d10b549c734c6e055abfbf53dffb92154a06
size 2800
=======
import pandas as pd

def calculate_final_score(df):
    """
    다양한 팩터 점수와 모델 예측치를 종합하여 최종 점수를 계산합니다.
    
    [책임]
    - 각 점수에 대한 가중치 설정
    - 가중 평균을 통해 최종 점수 산출
    - 최종 순위 계산
    """
    if df.empty:
        print("경고: calculate_final_score에 빈 데이터프레임이 전달되었습니다. 빈 데이터프레임을 반환합니다.")
        return df.copy()

    final_df = df.copy()

    # 각 팩터 및 모델에 대한 가중치 설정
    # 실제 데이터 기반으로 변경된 컬럼명을 사용하고, DL 및 NLP 모델 가중치는 0으로 설정
    factor_weights = {
        'value_score': 0.15,         # 가치
        'quality_score': 0.15,       # 퀄리티
        'momentum_score': 0.25,      # 모멘텀 (가중치 상향)
        'supply_score': 0.15,        # 수급 (가중치 상향)
        'volatility_score': 0.10,    # 변동성
        'ml_pred_proba': 0.20,       # 머신러닝 예측
        'sentiment_score': 0.00,     # 뉴스 감성 분석 (임시 비활성화)
        'dl_trend_score(더미)': 0.00, # 딥러닝 (더미 데이터, 영향력 제거)
    }

    # 가중치 합이 1이 되도록 정규화 (영향력 0인 모델 제외)
    active_factors = {k: v for k, v in factor_weights.items() if v > 0}
    total_weight = sum(active_factors.values())
    for k in active_factors:
        factor_weights[k] /= total_weight

    # 최종 점수 계산
    final_df['final_score'] = (
        final_df['value_score'] * factor_weights['value_score'] +
        final_df['quality_score'] * factor_weights['quality_score'] +
        final_df['momentum_score'] * factor_weights['momentum_score'] +
        final_df['supply_score'] * factor_weights['supply_score'] +
        final_df['volatility_score'] * factor_weights['volatility_score'] +
        final_df['ml_pred_proba'] * factor_weights['ml_pred_proba']
        # sentiment_score와 dl_trend_score는 가중치가 0이므로 계산에서 제외됨
    )

    # 최종 점수를 0~100 사이로 스케일링
    min_score = final_df['final_score'].min()
    max_score = final_df['final_score'].max()
    if max_score > min_score:
        final_df['final_score'] = 100 * (final_df['final_score'] - min_score) / (max_score - min_score)
    else:
        final_df['final_score'] = 50 # 모든 점수가 같을 경우

    # 최종 순위 계산
    final_df['최종순위'] = final_df['final_score'].rank(ascending=False, method='first').astype(int)
    
    # 소수점 정리
    final_df['final_score'] = final_df['final_score'].round(2)

    return final_df.sort_values(by='최종순위')
>>>>>>> origin/window
