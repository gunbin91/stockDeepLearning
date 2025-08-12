import streamlit as st
import pandas as pd

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
    stock_list_df = data_fetcher.fetch_stock_list() # 전 종목
    st.dataframe(stock_list_df.head())

    if st.button("실제 데이터 수집 및 분석 시작"):
        with st.spinner("데이터 수집 및 분석 중... (최대 5분 소요)"):
            # 1. 모든 데이터 수집 (재무, 수급, 모멘텀)
            all_data_df = data_fetcher.fetch_all_data(stock_list_df)
            st.write("### 2. 팩터 점수 계산")
            scored_df = scoring.calculate_factor_scores(all_data_df)
            st.dataframe(scored_df.head())

            # 3. 머신러닝 예측
            st.write("### 3. 머신러닝 예측")
            # 함수 이름 변경: predict_with_xgboost -> predict_with_ml_model
            ml_predicted_df = ml_model.predict_with_ml_model(scored_df)
            st.dataframe(ml_predicted_df.head())

            # 4. 딥러닝 시계열 예측 (더미 데이터 사용)
            st.write("### 4. 딥러닝 시계열 예측 (더미)")
            # dl_model은 현재 더미 데이터를 반환하도록 유지
            dl_predicted_df = dl_model.predict_with_deep_learning(ml_predicted_df)
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

            # ML 예측 확률은 이미 0~100 사이의 값이므로 추가 변환이 필요 없습니다.

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