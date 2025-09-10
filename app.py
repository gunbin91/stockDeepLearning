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
import subprocess
import re
import sys
import json
import io

# stdout/stderr를 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# --- 사용자 정의 모듈 임포트 ---
import data_fetcher
import scoring
import ml_model
import dl_model
import ensemble

# 페이지 레이아웃 설정
st.set_page_config(layout="wide")

# [FIX] 백테스팅 리포트 너비 문제 해결을 위한 CSS 주입
st.markdown("""
<style>
.stElementContainer.element-container {
    width: 100% !important;
    max-width: 100% !important;
}
iframe {
    width: 100% !important;
    max-width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# --- 설정 ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stock_prediction_model_rf_upgraded.joblib')


# --- 차트 생성 함수 (요청사항 반영) ---
@st.cache_data(ttl=3600) # 1시간 캐싱
def create_stock_chart(ticker_code, stock_name):
    """지정된 종목의 상세 기술적 분석 차트를 생성합니다."""
    try:
        # Ensure ticker_code is a 6-digit string
        padded_ticker_code = str(ticker_code).zfill(6)

        # 데이터는 2년치를 불러와서 장기 이동평균선 계산에 사용
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        df = fdr.DataReader(padded_ticker_code, start_date, end_date)
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
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0_2.0'], name='BB 상단', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0_2.0'], name='BB 하단', line=dict(color='gray', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
        
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
            title=f'<b>{stock_name} ({padded_ticker_code}) 기술적 분석</b>', yaxis_title='가격 (원)',
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

    # tqdm 진행률 표시줄을 식별하기 위한 정규표현식 (백테스팅에서 복사)
    TQDM_REGEX = re.compile(r'\s*\d{1,3}%|.*')

    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'market_condition' not in st.session_state:
        st.session_state.market_condition = None
    if 'analysis_date' not in st.session_state:
        st.session_state.analysis_date = None
    if 'cached_features_df' not in st.session_state:
        cached_features_path = os.path.join(os.path.dirname(__file__), 'cache', 'cached_features.json')
        if os.path.exists(cached_features_path):
            try:
                # JSON 로드 시 종목코드를 문자열로 유지
                st.session_state.cached_features_df = pd.read_json(cached_features_path, orient='records', dtype={'종목코드': str})
            except Exception as e:
                st.warning(f"캐시된 피처 데이터를 불러오는 데 실패했습니다: {e}")
        else:
            st.session_state.cached_features_df = pd.DataFrame()

    st.write("### 1. 종목 데이터 수집")
    
    # 분석 기준일 선택
    selected_analysis_date = st.date_input(
        "분석 기준일 선택",
        value=datetime.now(),
        max_value=datetime.now(),
        help="선택된 날짜를 기준으로 종목을 분석합니다. 휴장일 선택 시 가장 가까운 이전 거래일이 기준이 됩니다."
    )
    
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
                st.session_state.analysis_date = None
                if 'selected_stock' in st.session_state:
                    del st.session_state['selected_stock']
                st.rerun()

    if start_analysis:
        # --- 분석 로직을 서브프로세스로 실행 ---
        st.subheader("실시간 분석 로그")
        log_placeholder = st.empty()
        log_lines = []
        analysis_completed_successfully = False

        with st.spinner('분석 스크립트를 실행하고 실시간 로그를 수신 중입니다...'):
            try:
                command = [
                    'python', '-u', os.path.join(os.path.dirname(__file__), 'scripts', 'run_analysis.py'),
                    '--date', selected_analysis_date.strftime('%Y-%m-%d')
                ]
                
                process = subprocess.Popen(
                    command, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    encoding='utf-8',
                    bufsize=1
                )

                last_line_was_tqdm = False
                for line in iter(process.stdout.readline, ''):
                    # 터미널 출력
                    if TQDM_REGEX.search(line):
                        sys.stdout.write(line.strip() + '\r')
                        last_line_was_tqdm = True
                    else:
                        if last_line_was_tqdm:
                            sys.stdout.write('\n')
                        sys.stdout.write(line)
                        last_line_was_tqdm = False
                    sys.stdout.flush()

                    # UI 출력
                    if TQDM_REGEX.search(line):
                        if log_lines and TQDM_REGEX.search(log_lines[-1]):
                            log_lines[-1] = line
                        else:
                            log_lines.append(line)
                    else:
                        log_lines.append(line)
                    
                    display_text = '\n'.join([l.strip() for l in log_lines])
                    log_placeholder.code(display_text, language='bash')
                
                process.stdout.close()
                return_code = process.wait()

                if return_code == 0:
                    st.success("분석이 성공적으로 완료되었습니다!")
                    analysis_completed_successfully = True
                else:
                    st.error("분석 실행 중 오류가 발생했습니다. 위 로그를 확인해주세요.")

            except FileNotFoundError:
                st.error("'python' 명령을 찾을 수 없습니다. 가상환경이 올바르게 설정되었는지 확인하세요.")
            except Exception as e:
                st.error(f"스크립트 실행 중 예상치 못한 오류가 발생했습니다: {e}")

        # --- 분석 결과 처리 ---
        if analysis_completed_successfully:
            try:
                # 캐시된 결과 파일 읽기
                result_path = os.path.join(os.path.dirname(__file__), 'cache', 'analysis_result.json')
                market_path = os.path.join(os.path.dirname(__file__), 'cache', 'market_condition.json')

                final_df = pd.read_json(result_path, orient='records')
                with open(market_path, 'r', encoding='utf-8') as f:
                    st.session_state.market_condition = json.load(f)

                # 날짜 형식 변환
                final_df['date'] = pd.to_datetime(final_df['date'])
                st.session_state.analysis_date = final_df['date'].iloc[0].strftime('%Y년 %m월 %d일')

                # --- 기존의 데이터프레임 후처리 및 UI 표시 로직 ---
                display_df = final_df.copy()
                if 'ml_pred_proba' in display_df.columns:
                    display_df['ml_pred_proba'] = display_df['ml_pred_proba'] * 100
                
                display_df['등락율'] = ((display_df['현재가'] - display_df['기준일가']) / display_df['기준일가']) * 100
                
                def format_price_with_change(row):
                    price = f"{int(row['현재가']):,}"
                    change_percent = row['등락율']
                    if pd.isna(change_percent): return f"{price}원"
                    sign = '+' if change_percent > 0 else ''
                    formatted_change = f"{sign}{change_percent:.2f}%"
                    return f"{price}원 ({formatted_change})"

                display_df['현재가(원)_formatted'] = display_df.apply(format_price_with_change, axis=1)

                rename_map = { '현재가': '현재가(원)', '시가총액': '시가총액(억)', 'value_score': '가치(점)', 'quality_score': '퀄리티(점)', 'momentum_score': '모멘텀(점)', 'supply_score': '수급(점)', 'volatility_score': '변동성(점)', 'ml_pred_proba': '상승확률(%)', 'final_score': '최종점수(점)', '기준일가': '기준일가(원)'}
                display_df.rename(columns=rename_map, inplace=True)
                
                display_columns = [ '최종순위', '종목명', '종목코드', '현재가(원)_formatted', '기준일가(원)', '최종점수(점)', '상승확률(%)', '모멘텀(점)', '가치(점)', '퀄리티(점)', '수급(점)', '변동성(점)', '시가총액(억)']
                
                st.session_state.analysis_result = display_df[[col for col in display_columns if col in display_df.columns] + ['등락율']].rename(columns={'현재가(원)_formatted': '현재가(원)'})
                st.rerun()

            except FileNotFoundError:
                st.error("분석 결과 파일을 찾을 수 없습니다. 스크립트가 정상적으로 실행되었는지 확인하세요.")
            except Exception as e:
                st.error(f"분석 결과 처리 중 오류가 발생했습니다: {e}")
    
    if st.session_state.analysis_result is not None:
        analysis_date_str = st.session_state.get('analysis_date', "알 수 없는 날짜")
        st.success(f"분석이 완료되었습니다. (분석 기준일: {analysis_date_str}) 아래 목록에서 종목을 클릭하여 상세 차트를 확인하세요.")
        
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
        
        # <<< ✨ 핵심 수정: 스타일링 기능 복구 및 안정화 ✨ >>>
        
        # --- 스타일링 함수 정의 ---
        def highlight_change(row):
            styles = ['' for _ in row.index] # 모든 컬럼에 대한 기본 스타일
            
            if '등락율' in row.index:
                change_percent = row['등락율']
                if pd.notna(change_percent):
                    # get_loc으로 '현재가(원)' 컬럼의 정확한 위치를 찾아 스타일 적용
                    price_col_index = row.index.get_loc('현재가(원)')
                    if change_percent > 0:
                        styles[price_col_index] = 'color: red;'
                    elif change_percent < 0:
                        styles[price_col_index] = 'color: blue;'
            return styles

        # 1. '등락율'이 포함된 전체 데이터프레임에 스타일 적용
        styled_df = results_df.style.apply(highlight_change, axis=1)

        st.dataframe(
            styled_df,
            on_select="rerun",
            selection_mode="single-row",
            key="selected_stock",
            hide_index=True,
            # 2. Streamlit의 column_config를 사용하여 UI에서만 특정 컬럼 숨기기
            column_config={
                "종목코드": None,
                "등락율": None,
            },
            use_container_width=True
        )
        
        # --- 선택된 종목의 상세 차트 표시 (차트 미표시 오류 수정) ---
        if st.session_state.selected_stock and st.session_state.selected_stock.get("selection", {}).get("rows"):
            try:
                selected_index = st.session_state.selected_stock["selection"]["rows"][0]
                selected_row = results_df.iloc[selected_index]
                ticker_code = str(selected_row['종목코드']) # 종목코드를 문자열로 명시적 변환
                stock_name = selected_row['종목명']
                
                st.markdown("---")
                st.subheader(f"📈 [{stock_name}] 상세 차트")
                
                with st.spinner(f"'{stock_name}'의 상세 차트 데이터를 불러오는 중..."):
                    fig = create_stock_chart(ticker_code, stock_name)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                        # 피처 데이터 표시 로직 안정화
                        if not st.session_state.cached_features_df.empty:
                            # 종목코드는 항상 6자리 문자열로 비교
                            selected_stock_features = st.session_state.cached_features_df[st.session_state.cached_features_df['종목코드'] == str(ticker_code).zfill(6)]
                            
                            if not selected_stock_features.empty:
                                st.subheader(f"📊 {stock_name} ({ticker_code}) 분석 피처 데이터")
                                
                                display_features = selected_stock_features.drop(columns=['종목코드', 'date'], errors='ignore')
                                
                                # 1. 행/열 전환
                                transposed_df = display_features.transpose()
                                # 2. 열 이름 설정
                                transposed_df.columns = ['피처 값']
                                # 3. Null 값을 'N/A' 문자열로 채우기
                                transposed_df.fillna("N/A", inplace=True)
                                # 4. 모든 값을 문자열로 변환하여 타입 일관성 확보 (오류 방지)
                                transposed_df = transposed_df.astype(str)
                                
                                st.dataframe(transposed_df, use_container_width=True)
                            else:
                                st.info(f"{stock_name} ({ticker_code})에 대한 피처 데이터를 찾을 수 없습니다.")
                        else:
                            st.info("캐시된 피처 데이터가 없습니다. 분석을 먼저 실행해주세요.")
                    else:
                        st.warning("차트를 표시할 수 없습니다.")
            except (KeyError, IndexError, ValueError) as e: 
                st.error(f"선택된 행의 인덱스를 처리하는 중 오류가 발생했습니다: {e}. 다시 시도해주세요.")
                pass
            except Exception as e:
                st.error(f"차트 표시 중 오류 발생: {e}")
        

# --- 백테스팅 리포트 페이지 ---
def display_backtest_report():
    st.header("백테스팅 리포트")

    # tqdm 진행률 표시줄을 식별하기 위한 정규표현식
    TQDM_REGEX = re.compile(r'\s*\d{1,3}%|.*')

    if 'show_backtest_form' not in st.session_state:
        st.session_state.show_backtest_form = False

    if st.button("백테스팅 신규 실행", type="primary"):
        st.session_state.show_backtest_form = not st.session_state.show_backtest_form

    form_container = st.empty()

    if st.session_state.show_backtest_form:
        with form_container.container():
            with st.form("backtest_form"):
                st.subheader("백테스팅 조건 설정")
                st.write("백테스팅에 사용할 파라미터를 입력하세요.")
                
                capital = st.number_input("초기 자본 (원)", min_value=1000000, value=10000000, step=1000000)
                max_hold = st.number_input("최대 보유 기간 (일)", min_value=1, value=15, step=1)
                take_profit = st.number_input("수익 실현율 (%)", min_value=0.1, value=5.0, step=0.1)
                stop_loss = st.number_input("손절률 (%)", min_value=0.1, value=3.0, step=0.1)
                top_n = st.number_input("매수 종목 수 (상위 N개)", min_value=1, value=5, step=1)
                buy_universe = st.number_input("매수 대상 범위 (상위 N위)", min_value=top_n, value=20, step=1)
                transaction_fee = st.number_input(
                    "거래 수수료 (%)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.015, # 기본값 0.015%
                    step=0.001,
                    format="%.3f",
                    help="매수 및 매도 시 적용될 거래 수수료율 (예: 0.015는 0.015%)"
                )

                submitted = st.form_submit_button("실행")
                if submitted:
                    st.session_state.show_backtest_form = False
                    
                    st.subheader("백테스팅 실행 로그")
                    log_placeholder = st.empty()
                    log_lines = []

                    with st.spinner('백테스팅 스크립트를 실행하고 실시간 로그를 수신 중입니다...'):
                        try:
                            command = [
                                'python', '-u', os.path.join(os.path.dirname(__file__), 'scripts', 'backtest.py'),
                                '--capital', str(capital),
                                '--max-hold', str(max_hold),
                                '--take-profit', str(take_profit),
                                '--stop-loss', str(stop_loss),
                                '--top-n', str(top_n),
                                '--buy-universe', str(buy_universe),
                                '--fee', str(transaction_fee)
                            ]
                            
                            process = subprocess.Popen(
                                command, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT, 
                                text=True, 
                                encoding='utf-8',
                                bufsize=1
                            )

                            last_line_was_tqdm = False
                            for line in iter(process.stdout.readline, ''):
                                is_tqdm_line = TQDM_REGEX.search(line) is not None

                                # --- 터미널 출력 로직 ---
                                if is_tqdm_line:
                                    # tqdm 라인이면, 줄바꿈 없이 캐리지 리턴으로 덮어쓰기
                                    sys.stdout.write(line.strip() + '\r')
                                    last_line_was_tqdm = True
                                else:
                                    # 일반 로그 라인이 tqdm 라인 바로 다음에 오면, 줄바꿈으로 완료 처리
                                    if last_line_was_tqdm:
                                        sys.stdout.write('\n')
                                    sys.stdout.write(line)
                                    last_line_was_tqdm = False
                                sys.stdout.flush()

                                # --- UI 출력 로직 (기존과 동일) ---
                                if is_tqdm_line:
                                    if log_lines and TQDM_REGEX.search(log_lines[-1]) is not None:
                                        log_lines[-1] = line
                                    else:
                                        log_lines.append(line)
                                else:
                                    log_lines.append(line)
                                
                                display_text = '\n'.join([l.strip() for l in log_lines])
                                log_placeholder.code(display_text, language='bash')
                            
                            process.stdout.close()
                            return_code = process.wait()

                            if return_code == 0:
                                st.success("백테스팅이 성공적으로 완료되었습니다!")
                            else:
                                st.error("백테스팅 실행 중 오류가 발생했습니다. 위 로그를 확인해주세요.")

                        except FileNotFoundError:
                            st.error("'python' 명령을 찾을 수 없습니다. 가상환경이 올바르게 설정되었는지 확인하세요.")
                        except Exception as e:
                            st.error(f"스크립트 실행 중 예상치 못한 오류가 발생했습니다: {e}")
                    
                    time.sleep(2)
                    st.rerun()

    st.markdown("---")
    report_path = os.path.join(os.path.dirname(__file__), 'backtest_report.html')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.success(f"'{report_path}' 파일을 성공적으로 불러왔습니다.")
        components.html(html_content, height=1600, scrolling=True, width=None)
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