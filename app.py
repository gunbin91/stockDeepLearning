import streamlit as st
import pandas as pd
import time

# 사용자 정의 모듈 임포트
import data_fetcher
import scoring
import ml_model
import dl_model
import nlp
import ensemble

def main():
    st.set_page_config(layout="wide")
    st.title("주식 추천 시스템")

    st.write("### 1. 종목 데이터 수집")
    
    MAX_RETRIES = 5
    RETRY_DELAY_SECONDS = 5 # 재시도 간격 (초)

    stock_list_df = pd.DataFrame() # 빈 데이터프레임으로 초기화

    with st.spinner("전체 종목 목록을 API로부터 수신하는 중..."):
        for attempt in range(MAX_RETRIES):
            stock_list_df = data_fetcher.fetch_stock_list()
            if not stock_list_df.empty:
                st.success("종목 목록 수신 완료!")
                break
            
            st.warning(f"API 통신에 실패했습니다 ({attempt + 1}/{MAX_RETRIES}). {RETRY_DELAY_SECONDS}초 후 재시도합니다...")
            time.sleep(RETRY_DELAY_SECONDS)
        
    if stock_list_df.empty:
        st.error("API 통신 오류: 최대 재시도 횟수를 초과했습니다. 서버 상태를 확인하거나 잠시 후 페이지를 새로고침해주세요.")
        st.stop()

    st.dataframe(stock_list_df.head())

    if st.button("실제 데이터 수집 및 분석 시작"):
        with st.spinner("데이터 수집 및 분석 중... (최대 5분 소요)"):
            # <<< 개선점: 데이터 처리 로직 수정 >>>
            # 1. 모든 데이터 수집
            # fetch_all_data는 이제 두 개의 데이터프레임을 반환한다고 가정합니다.
            # 1) ml_feature_df: ML 모델 예측에 필요한 원본 피처들이 있는 데이터프레임
            # 2) factor_base_df: 팩터 점수 계산에 필요한 재무, 수급, 가격 데이터가 있는 데이터프레임
            ml_feature_df, factor_base_df = data_fetcher.fetch_all_data(stock_list_df)
            
            # 데이터 수집 실패 시 처리
            if factor_base_df.empty:
                st.error("데이터를 수집하는 데 실패했거나 분석할 종목이 없습니다. 잠시 후 다시 시도해주세요.")
                st.stop()
            
            # 2. 팩터 점수 계산
            st.write("### 2. 팩터 점수 계산")
            # 팩터 점수는 factor_base_df를 기반으로 계산
            scored_df = scoring.calculate_factor_scores(factor_base_df)
            st.dataframe(scored_df.head())

            # 3. 머신러닝 예측
            st.write("### 3. 머신러닝 예측")
            # ML 모델 예측은 ml_feature_df를 사용
            # predict_with_ml_model 함수는 내부적으로 모델을 로드하고, ml_feature_df에서 필요한 피처만 선택하여 예측
            ml_predicted_df = ml_model.predict_with_ml_model(ml_feature_df)
            
            # 모델 로드 실패 시 분석 중단
            if ml_predicted_df is None:
                st.error("머신러닝 모델 예측에 실패했습니다. 콘솔 로그를 확인해주세요.")
                st.stop()

            st.dataframe(ml_predicted_df.head()) # '종목코드', 'ml_pred_proba' 컬럼 등이 있을 것으로 예상

            # --- 예측 결과와 팩터 점수 데이터 병합 ---
            # ml_predicted_df와 scored_df를 '종목코드' 기준으로 병합
            # 이 작업을 통해 각 종목의 팩터 점수와 ML 예측 확률을 하나의 DataFrame으로 합침
            # (두 DataFrame에 공통된 '종목코드' 또는 '종목명' 컬럼이 있어야 함)
            merged_df = pd.merge(scored_df, ml_predicted_df, on='종목코드', how='left')

            # 4. 딥러닝 시계열 예측 (더미 데이터 사용)
            st.write("### 4. 딥러닝 시계열 예측 (더미)")
            # dl_model은 현재 더미 데이터를 반환하도록 유지
            dl_predicted_df = dl_model.predict_with_deep_learning(merged_df)
            st.dataframe(dl_predicted_df.head())

            # 5. 뉴스 감성 분석 (임시 비활성화)
            st.write("### 5. 뉴스 감성 분석 (추후 업데이트 예정)")
            # nlp_analyzed_df = nlp.analyze_news_sentiment(dl_predicted_df)
            # st.dataframe(nlp_analyzed_df.head())
            
            # 임시로 sentiment_score 컬럼을 0으로 채웁니다.
            nlp_analyzed_df = dl_predicted_df.copy()
            nlp_analyzed_df['sentiment_score'] = 0
            st.info("현재 감성 분석 기능은 성능 최적화를 위해 임시 비활성화되었습니다.")

            # 6. 최종 점수 및 순위 계산
            st.write("### 6. 최종 점수 및 순위")
            final_ranked_df = ensemble.calculate_final_score(nlp_analyzed_df)

            # --- 최종 결과 표시용 데이터 가공 ---
            display_df = final_ranked_df.copy()

            # scikit-learn의 predict_proba는 0~1 사이의 확률을 반환하므로 100을 곱해 %로 변환
            if 'ml_pred_proba' in display_df.columns:
                 display_df['ml_pred_proba'] = display_df['ml_pred_proba'] * 100

            # 컬럼 이름에 단위 추가
            rename_map = {
                '현재가': '현재가(원)',
                '시가총액': '시가총액(억)',
                'value_score': '가치(점)',
                'quality_score': '퀄리티(점)',
                'momentum_score': '모멘텀(점)',
                'supply_score': '수급(점)',
                'volatility_score': '변동성(점)',
                'ml_pred_proba': '상승확률(%)',
                'final_score': '최종점수(점)'
            }
            display_df.rename(columns=rename_map, inplace=True)

            # 보여줄 컬럼 순서 재정렬
            display_columns = [
                '최종순위', '종목명', '현재가(원)', '최종점수(점)', '상승확률(%)',
                '모멘텀(점)', '가치(점)', '퀄리티(점)', '수급(점)', '변동성(점)',
                '시가총액(억)'
            ]
            # 혹시 모를 오류 방지를 위해 실제 존재하는 컬럼만 선택
            display_columns = [col for col in display_columns if col in display_df.columns]
            
            st.dataframe(display_df[display_columns])

            st.success("분석 완료!")

if __name__ == "__main__":
    main()