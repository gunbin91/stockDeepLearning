# app.py

import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime, timedelta
import joblib
import streamlit.components.v1 as components
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta

# --- 사용자 정의 모듈 임포트 ---
import data_fetcher
import scoring
import ml_model
import dl_model
import nlp
import ensemble

# 페이지 레이아웃 설정
st.set_page_config(layout="wide")

# --- 설정 ---
MODEL_PATH = 'stock_prediction_model_rf_upgraded.joblib'


# --- 차트 생성 함수 (요청사항 반영) ---
@st.cache_data(ttl=3600) # 1시간 캐싱
def create_stock_chart(ticker_code, stock_name):
    """지정된 종목의 상세 기술적 분석 차트를 생성합니다."""
    try:
        # 데이터는 2년치를 불러와서 장기 이동평균선 계산에 사용
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        df = fdr.DataReader(ticker_code, start_date, end_date)
        if df.empty:
            st.warning("차트 데이터를 불러오는 데 실패했습니다.")
            return None

        # 이동평균선(MA) 계산
        df.ta.bbands(length=20, std=2, append=True)
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA122'] = df['Close'].rolling(window=122).mean()
        df['MA244'] = df['Close'].rolling(window=244).mean()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # 캔들스틱 및 이동평균선 추가
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='캔들스틱'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], name='MA 10', line=dict(color='limegreen', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA 20', line=dict(color='red', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA122'], name='MA 122', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA244'], name='MA 244', line=dict(color='purple', width=1.5)), row=1, col=1)
        
        # 볼린저 밴드
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], name='BB 상단', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], name='BB 하단', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
        # 등락에 따른 거래량 막대 색상
        colors = ['red' if row['Close'] > row['Open'] else 'blue' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='거래량', marker_color=colors), row=2, col=1)
        
        # --- 요청사항 반영: 초기 차트 기간 6개월 및 모든 휴장일 공백 제거 ---
        # 전체 날짜 범위를 생성
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        # 원본 데이터에 없는 날짜(휴장일)를 찾음
        missing_dates = full_date_range.difference(df.index)

        six_months_ago = df.index.max() - timedelta(days=183) # 약 6개월
        fig.update_xaxes(
            range=[six_months_ago, df.index.max()],
            rangebreaks=[
                dict(values=missing_dates)  # 주말 및 공휴일을 모두 제외
            ]
        )

        fig.update_layout(
            title=f'<b>{stock_name} ({ticker_code}) 기술적 분석</b>', yaxis_title='가격 (원)',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        return fig
    except Exception as e:
        st.error(f"차트 생성 중 오류 발생: {e}")
        return None


# --- 모델 분석 페이지 (이전과 동일) ---
@st.cache_data
def load_model_data(model_path):
    try:
        model_data = joblib.load(model_path)
        return model_data
    except FileNotFoundError:
        return None

def display_model_analysis_page():
    # ... (이전과 동일한 내용이므로 생략) ...
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


# --- 주식 추천 페이지 로직 (안정화) ---
def run_stock_recommendation():
    st.title("주식 추천 시스템")

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'market_condition' not in st.session_state:
        st.session_state.market_condition = None

    st.write("### 1. 종목 데이터 수집")
    with st.spinner("전체 종목 목록을 API로부터 수신하는 중..."):
        stock_list_df = data_fetcher.fetch_stock_list()
    if stock_list_df.empty:
        st.error("API 통신 오류: 종목 목록을 가져오는 데 실패했습니다. 잠시 후 페이지를 새로고침해주세요.")
        st.stop()
    st.success("종목 목록 수신 완료!")

    col1, col2, _ = st.columns([1, 1, 5])
    with col1:
        start_analysis = st.button("실제 데이터 수집 및 분석 시작", type="primary")
    with col2:
        if st.session_state.analysis_result is not None:
            if st.button("결과 초기화"):
                st.session_state.analysis_result = None
                st.session_state.market_condition = None
                if 'selected_stock' in st.session_state:
                    del st.session_state['selected_stock']
                st.rerun()

    if start_analysis:
        with st.spinner("데이터 수집 및 분석 중... (최대 5분 소요)"):
            feature_df = data_fetcher.fetch_all_data(stock_list_df)
            if not feature_df.empty:
                macro_cols = ['KOSPI', 'KOSPI_pct_1d', 'USDKRW', 'USDKRW_pct_1d', 'VIX', 'VIX_pct_1d']
                if all(col in feature_df.columns for col in macro_cols):
                    st.session_state.market_condition = feature_df.iloc[0][macro_cols].to_dict()
                
                scored_df = scoring.calculate_factor_scores(feature_df)
                ml_predicted_df = ml_model.predict_with_ml_model(feature_df)

                if ml_predicted_df is not None:
                    merged_df = pd.merge(scored_df, ml_predicted_df, on='종목코드', how='left')
                    dl_predicted_df = dl_model.predict_with_deep_learning(merged_df)
                    nlp_analyzed_df = dl_predicted_df.copy()
                    nlp_analyzed_df['sentiment_score'] = 0
                    final_ranked_df = ensemble.calculate_final_score(nlp_analyzed_df)
                    
                    # '종목코드'가 인덱스로 설정된 경우를 대비해 컬럼으로 변환
                    if '종목코드' not in final_ranked_df.columns:
                        final_ranked_df.reset_index(inplace=True)

                    # '종목명' 누락 방지를 위한 최종 병합
                    final_df_with_names = pd.merge(final_ranked_df, stock_list_df[['종목코드', '종목명']].drop_duplicates(), on='종목코드', how='left')
                    
                    # 병합 시 종목명 컬럼 충돌 해결
                    if '종목명_x' in final_df_with_names.columns:
                        final_df_with_names['종목명'] = final_df_with_names['종목명_y'].fillna(final_df_with_names['종목명_x'])
                        final_df_with_names.drop(columns=['종목명_x', '종목명_y'], inplace=True)

                    display_df = final_df_with_names.copy()
                    if 'ml_pred_proba' in display_df.columns:
                        display_df['ml_pred_proba'] = display_df['ml_pred_proba'] * 100
                    rename_map = { '현재가': '현재가(원)', '시가총액': '시가총액(억)', 'value_score': '가치(점)', 'quality_score': '퀄리티(점)', 'momentum_score': '모멘텀(점)', 'supply_score': '수급(점)', 'volatility_score': '변동성(점)', 'ml_pred_proba': '상승확률(%)', 'final_score': '최종점수(점)'}
                    display_df.rename(columns=rename_map, inplace=True)
                    
                    display_columns = [ '최종순위', '종목명', '종목코드', '현재가(원)', '최종점수(점)', '상승확률(%)', '모멘텀(점)', '가치(점)', '퀄리티(점)', '수급(점)', '변동성(점)', '시가총액(억)']
                    st.session_state.analysis_result = display_df[[col for col in display_columns if col in display_df.columns]]
                else: st.error("머신러닝 모델 예측에 실패했습니다.")
            else: st.error("데이터 수집에 실패했습니다.")
    
    if st.session_state.analysis_result is not None:
        st.success("분석이 완료되었습니다. 아래 목록에서 종목을 클릭하여 상세 차트를 확인하세요.")
        
        market_data = st.session_state.market_condition
        if market_data:
            st.subheader(" 분석 시점 시장 현황")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="KOSPI", value=f"{market_data.get('KOSPI', 0):,.2f}", delta=f"{market_data.get('KOSPI_pct_1d', 0):.2%}")
            with col2:
                st.metric(label="USD/KRW 환율", value=f"{market_data.get('USDKRW', 0):,.2f} 원", delta=f"{market_data.get('USDKRW_pct_1d', 0):.2%}")
            with col3:
                st.metric(label="VIX (변동성 지수)", value=f"{market_data.get('VIX', 0):,.2f}", delta=f"{market_data.get('VIX_pct_1d', 0):.2%}", delta_color="inverse")
            st.markdown("---")

        results_df = st.session_state.analysis_result
        
        st.info("테이블의 행을 클릭하면 아래에 상세 차트가 나타납니다.")
        
        display_cols_in_table = [col for col in results_df.columns if col != '종목코드']
        
        st.dataframe(
            results_df[display_cols_in_table],
            on_select="rerun",
            selection_mode="single-row",
            key="selected_stock",
            hide_index=True,
            use_container_width=True
        )
        
        # 최신 Streamlit 버전에 맞는 안정적인 선택 확인 로직
        selection = st.session_state.get('selected_stock', {}).get('selection', {})
        if selection.get('rows'):
            selected_index = selection['rows'][0]
            selected_row = results_df.iloc[selected_index]
            
            ticker_code = selected_row['종목코드']
            stock_name = selected_row['종목명']
            
            st.markdown("---")
            st.subheader(f" {stock_name} ({ticker_code}) 상세 차트")

            with st.spinner(f"차트 데이터 로딩 중..."):
                fig = create_stock_chart(ticker_code, stock_name)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
# --- 백테스팅 리포트 페이지 (이전과 동일) ---
def display_backtest_report():
    st.header("백테스팅 리포트")
    report_path = 'backtest_report.html'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.success(f"'{report_path}' 파일을 성공적으로 불러왔습니다.")

        # HACK: Streamlit의 iframe 너비 강제 재정의 (사용자 제공 클래스명 기반)
        # Streamlit 업데이트 시 클래스명이 변경되어 작동하지 않을 수 있습니다.
        st.markdown('''
        <style>
            .st-emotion-cache-6osm6r {
                width: 100%;
            }
        </style>
        ''', unsafe_allow_html=True)
        
        components.html(html_content, height=1600, scrolling=True)
    else:
        st.error(f"리포트 파일('{report_path}')을 찾을 수 없습니다. `backtest.py`를 먼저 실행하여 리포트를 생성해주세요.")

# --- 메인 실행 로직 (이전과 동일) ---
def main():
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
