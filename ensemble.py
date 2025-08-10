import pandas as pd

def calculate_final_score(df):
    """
    다양한 팩터 점수와 모델 예측치를 종합하여 최종 점수를 계산합니다.
    
    [책임]
    - 각 점수에 대한 가중치 설정
    - 가중 평균을 통해 최종 점수 산출
    - 최종 순위 계산
    """
    final_df = df.copy()

    # 각 팩터 및 모델에 대한 가중치 설정
    # 실제 데이터 기반으로 변경된 컬럼명을 사용하고, DL 모델 가중치는 0으로 설정
    factor_weights = {
        'value_score': 0.15,         # 가치
        'quality_score': 0.15,       # 퀄리티
        'momentum_score': 0.20,      # 모멘텀 (실제 데이터)
        'supply_score': 0.10,        # 수급
        'ml_pred_proba': 0.25,       # 머신러닝 예측 (실제 모델)
        'sentiment_score': 0.15,     # 뉴스 감성 분석 (실제 모델)
        'dl_trend_score(더미)': 0.00, # 딥러닝 (더미 데이터, 영향력 제거)
    }

    # 가중치 합이 1이 되도록 정규화 (DL 모델 제외)
    total_weight = sum(w for k, w in factor_weights.items() if k != 'dl_trend_score(더미)')
    for k in factor_weights:
        if k != 'dl_trend_score(더미)':
            factor_weights[k] /= total_weight

    # 최종 점수 계산
    final_df['final_score'] = (
        final_df['value_score'] * factor_weights['value_score'] +
        final_df['quality_score'] * factor_weights['quality_score'] +
        final_df['momentum_score'] * factor_weights['momentum_score'] +
        final_df['supply_score'] * factor_weights['supply_score'] +
        final_df['ml_pred_proba'] * factor_weights['ml_pred_proba'] +
        final_df['sentiment_score'] * factor_weights['sentiment_score']
        # dl_trend_score는 가중치가 0이므로 계산에서 사실상 제외됨
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