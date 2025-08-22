import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
import joblib
import streamlit.components.v1 as components

# 사용자 정의 모듈 임포트
import data_fetcher
import scoring
import ml_model
import dl_model
import nlp
import ensemble

MODEL_PATH = 'stock_prediction_model_rf_upgraded.joblib'

# --- 모델 분석 페이지를 위한 함수들 (이전과 동일) ---
@st.cache_data
def load_model_data(model_path):
    try:
        model_data = joblib.load(model_path)
        return model_data
    except FileNotFoundError:
        return None

def display_model_analysis_page():
    st.header("학습 모델 분석 리포트")
    model_data = load_model_data(MODEL_PATH)
    if model_data is None:
        st.error(f"모델 파일({MODEL_PATH})을 찾을 수 없습니다. `train_model.py`를 먼저 실행하여 모델을 생성해주세요.")
        return
    model = model_data['model']
    features = model_data['features']
    st.subheader("모델 개요")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**모델 목표**")
        st.write("15영업일 후 주가가 5% 이상 상승할 확률 예측")
        st.info("**모델 종류**")
        st.write(type(model).__name__)
    with col2:
        st.info("**모델 파일**")
        st.code(MODEL_PATH, language=None)
        last_modified_time = os.path.getmtime(MODEL_PATH)
        last_modified_datetime = datetime.fromtimestamp(last_modified_time)
        st.info("**최종 학습일**")
        st.write(last_modified_datetime.strftime('%Y-%m-%d %H:%M:%S'))
    st.subheader("모델 성능")
    if hasattr(model, 'oob_score_') and model.oob_score_:
        st.metric(label="OOB 점수 (자체 검증 정확도)", 
                  value=f"{model.oob_score_:.4f}",
                  help="모델이 학습에 사용하지 않은 데이터를 가지고 자체적으로 검증한 정확도입니다. 0.5 이상이면 일반적으로 유의미한 성능으로 봅니다.")
    st.subheader("피처 중요도 (Feature Importances)")
    st.info("모델이 어떤 정보를 중요하게 생각하여 주가 상승을 예측하는지 보여줍니다.")
    feature_importances = pd.DataFrame({
        '피처 (Feature)': features,
        '중요도 (Importance)': model.feature_importances_
    }).sort_values('중요도 (Importance)', ascending=False).reset_index(drop=True)
    st.bar_chart(feature_importances.set_index('피처 (Feature)'))
    with st.expander("피처 중요도 상세 데이터 보기"):
        st.dataframe(feature_importances)
    with st.expander("주요 학습 파라미터 보기"):
        params = model.get_params()
        params_to_show = {
            'n_estimators': params.get('n_estimators'), 'max_depth': params.get('max_depth'),
            'min_samples_split': params.get('min_samples_split'), 'min_samples_leaf': params.get('min_samples_leaf'),
            'max_samples': params.get('max_samples'), 'class_weight': params.get('class_weight'),
        }
        st.json(params_to_show)

# --- 주식 추천 페이지 로직 ---
def run_stock_recommendation():
    st.title("주식 추천 시스템")

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None

    st.write("### 1. 종목 데이터 수집")
    with st.spinner("전체 종목 목록을 API로부터 수신하는 중..."):
        stock_list_df = data_fetcher.fetch_stock_list()
    if stock_list_df.empty:
        st.error("API 통신 오류: 종목 목록을 가져오는 데 실패했습니다. 잠시 후 페이지를 새로고침해주세요.")
        st.stop()
    st.success("종목 목록 수신 완료!")
    st.dataframe(stock_list_df.head())

    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        start_analysis = st.button("실제 데이터 수집 및 분석 시작", type="primary")
    with col2:
        if st.session_state.analysis_result is not None:
            if st.button("결과 초기화"):
                st.session_state.analysis_result = None
                st.rerun()

    if start_analysis:
        with st.spinner("데이터 수집 및 분석 중... (최대 5분 소요)"):
            feature_df = data_fetcher.fetch_all_data(stock_list_df)
            if not feature_df.empty:
                scored_df = scoring.calculate_factor_scores(feature_df)
                ml_predicted_df = ml_model.predict_with_ml_model(feature_df)
                if ml_predicted_df is not None:
                    merged_df = pd.merge(scored_df, ml_predicted_df, on='종목코드', how='left')
                    dl_predicted_df = dl_model.predict_with_deep_learning(merged_df)
                    nlp_analyzed_df = dl_predicted_df.copy()
                    nlp_analyzed_df['sentiment_score'] = 0
                    final_ranked_df = ensemble.calculate_final_score(nlp_analyzed_df)
                    display_df = final_ranked_df.copy()
                    if 'ml_pred_proba' in display_df.columns:
                        display_df['ml_pred_proba'] = display_df['ml_pred_proba'] * 100
                    rename_map = {
                        '현재가': '현재가(원)', '시가총액': '시가총액(억)', 'value_score': '가치(점)',
                        'quality_score': '퀄리티(점)', 'momentum_score': '모멘텀(점)', 'supply_score': '수급(점)',
                        'volatility_score': '변동성(점)', 'ml_pred_proba': '상승확률(%)', 'final_score': '최종점수(점)'
                    }
                    display_df.rename(columns=rename_map, inplace=True)
                    display_columns = [
                        '최종순위', '종목명', '현재가(원)', '최종점수(점)', '상승확률(%)',
                        '모멘텀(점)', '가치(점)', '퀄리티(점)', '수급(점)', '변동성(점)', '시가총액(억)'
                    ]
                    display_df = display_df[[col for col in display_columns if col in display_df.columns]]
                    st.session_state.analysis_result = display_df
                else:
                    st.error("머신러닝 모델 예측에 실패했습니다.")
            else:
                st.error("데이터 수집에 실패했습니다.")
    
    if st.session_state.analysis_result is not None:
        st.success("분석이 완료되었습니다. 아래는 종합 분석 결과입니다.")
        st.dataframe(st.session_state.analysis_result)

# --- 백테스팅 리포트 페이지 ---
def display_backtest_report():
    st.header("백테스팅 리포트")
    report_path = 'backtest_report.html'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.success(f"'{report_path}' 파일을 성공적으로 불러왔습니다.")
        components.html(html_content, height=1500, scrolling=True)
    else:
        st.error(f"리포트 파일('{report_path}')을 찾을 수 없습니다. `backtest.py`를 먼저 실행하여 리포트를 생성해주세요.")

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("메뉴")
    page = st.sidebar.radio("페이지 선택", ["주식 추천", "학습 모델 분석", "백테스팅 리포트"])
    if page == "주식 추천":
        run_stock_recommendation()
    elif page == "학습 모델 분석":
        display_model_analysis_page()
    elif page == "백테스팅 리포트":
        display_backtest_report()

if __name__ == "__main__":
    main()