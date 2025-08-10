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
    stock_list_df = data_fetcher.fetch_stock_list(top_n=100) # 상위 100개 종목
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

            # 5. 뉴스 감성 분석
            st.write("### 5. 뉴스 감성 분석")
            # 함수 이름 변경: analyze_sentiment -> analyze_news_sentiment
            nlp_analyzed_df = nlp.analyze_news_sentiment(dl_predicted_df)
            st.dataframe(nlp_analyzed_df.head())

            # 6. 최종 점수 및 순위 계산
            st.write("### 6. 최종 점수 및 순위")
            final_ranked_df = ensemble.calculate_final_score(nlp_analyzed_df)
            st.dataframe(final_ranked_df)

            st.success("분석 완료!")

if __name__ == "__main__":
    main()