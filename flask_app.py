"""
AI 기반 주식 분석 시스템 - 웹 애플리케이션
===============================================

이 파일은 주식 분석 시스템의 웹 인터페이스를 제공합니다.
사용자가 웹 브라우저를 통해 주식 분석을 요청하고 결과를 확인할 수 있습니다.

주요 기능:
- 주식 분석 요청 및 결과 표시
- 실시간 분석 진행 상황 모니터링
- 백테스팅 기능
- 종목별 상세 차트 및 데이터 조회
"""

# Eventlet 몽키 패치 (최상단에 위치해야 함)
# 웹소켓 통신을 위한 비동기 처리 라이브러리
import eventlet
eventlet.monkey_patch()

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import subprocess
import re
import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import FinanceDataReader as fdr
import pandas_ta as ta
import joblib

# 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 기존 모듈들 임포트
import data_fetcher
import data_processor
import scoring
import ml_model
import ensemble
from logger import log_info, log_warning, log_error, log_critical
from exceptions import DataFetchError, ModelPredictionError, AnalysisError
from path_manager import path_manager, ensure_all_directories

# =============================================================================
# 웹 애플리케이션 초기화
# =============================================================================

# Flask 웹 애플리케이션 생성
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock_analysis_secret_key_2024'

# 실시간 통신을 위한 WebSocket 설정
# 사용자에게 분석 진행 상황을 실시간으로 전달하기 위함
socketio = SocketIO(app, 
                   cors_allowed_origins="*",  # 모든 도메인에서 접근 허용
                   ping_timeout=60,           # 연결 유지 시간 60초
                   ping_interval=25           # 연결 확인 간격 25초
)

# Jinja2 필터 추가
@app.template_filter('moment')
def moment_filter(value, format_string='YYYY-MM-DD'):
    """Moment.js 스타일 날짜 포맷팅"""
    if value is None:
        return datetime.now().strftime('%Y-%m-%d')
    if isinstance(value, str):
        return value
    return value.strftime('%Y-%m-%d')

# 설정 (통일된 경로 사용)
# cuML 모델 경로 (우선 사용)
CUML_MODEL_PATH = str(path_manager.data_dir / 'cuml_ensemble_model.joblib')
# 기존 모델 경로 (fallback)
MODEL_PATH = str(path_manager.get_model_path())

# =============================================================================
# 전역 변수 관리
# =============================================================================

# 현재 실행 중인 프로세스들을 추적하는 변수들
current_analysis_process = None  # 주식 분석 프로세스
current_backtest_process = None  # 백테스팅 프로세스

# 플래그를 함수로 관리하여 더 안전하게 처리
def get_analysis_running():
    """분석 실행 상태 확인"""
    return getattr(get_analysis_running, 'value', False)

def set_analysis_running(value):
    """분석 실행 상태 설정"""
    get_analysis_running.value = value

def get_backtest_running():
    """백테스팅 실행 상태 확인"""
    return getattr(get_backtest_running, 'value', False)

def set_backtest_running(value):
    """백테스팅 실행 상태 설정"""
    get_backtest_running.value = value

def reset_flags():
    """플래그 초기화"""
    set_analysis_running(False)
    set_backtest_running(False)

def is_process_running(process):
    """프로세스가 실제로 실행 중인지 확인"""
    if process is None:
        return False
    return process.poll() is None

def cleanup_analysis_process():
    """분석 프로세스 정리"""
    global current_analysis_process
    if current_analysis_process and is_process_running(current_analysis_process):
        try:
            current_analysis_process.terminate()
            current_analysis_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_analysis_process.kill()
            current_analysis_process.wait()
        except Exception:
            pass  # 프로세스가 이미 종료된 경우 무시
    set_analysis_running(False)
    current_analysis_process = None

def is_backtest_process_running(process):
    """백테스팅 프로세스가 실제로 실행 중인지 확인"""
    if process is None:
        return False
    return process.poll() is None

def cleanup_backtest_process():
    """백테스팅 프로세스 정리"""
    global current_backtest_process
    if current_backtest_process and is_backtest_process_running(current_backtest_process):
        try:
            current_backtest_process.terminate()
            current_backtest_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            current_backtest_process.kill()
            current_backtest_process.wait()
        except Exception:
            pass  # 프로세스가 이미 종료된 경우 무시
    set_backtest_running(False)
    current_backtest_process = None

# =============================================================================
# 유틸리티 함수들 - 데이터 포맷팅 및 처리
# =============================================================================

def format_price_with_change(row):
    """가격과 등락율을 함께 포맷팅"""
    price = f"{int(row['현재가']):,}"
    change_percent = row['등락율']
    if pd.isna(change_percent): 
        return f"{price}원"
    sign = '+' if change_percent > 0 else ''
    formatted_change = f"{sign}{change_percent:.2f}%"
    return f"{price}원<br>({formatted_change})"

def format_change_rate(change_percent):
    """등락율 포맷팅"""
    if pd.isna(change_percent):
        return "N/A"
    sign = '+' if change_percent > 0 else ''
    return f"{sign}{change_percent:.2f}%"

def load_cached_analysis_result():
    """분석 결과 로드 (캐시 없이)"""
    # 분석 결과 파일 경로 (data 디렉토리 사용)
    result_path = os.path.join(str(path_manager.data_dir), 'analysis_result.json')
    market_path = os.path.join(str(path_manager.data_dir), 'market_condition.json')
    
    if os.path.exists(result_path) and os.path.exists(market_path):
        try:
            # 기존 분석 결과 로드
            final_df = pd.read_json(result_path, orient='records')
            with open(market_path, 'r', encoding='utf-8') as f:
                market_condition = json.load(f)

            # 날짜 형식 변환
            final_df['date'] = pd.to_datetime(final_df['date'])
            analysis_date = final_df['date'].iloc[0].strftime('%Y년 %m월 %d일')

            # 데이터프레임 후처리
            display_df = final_df.copy()
            
            # 종목코드 6자리 패딩 보장
            if '종목코드' in display_df.columns:
                display_df['종목코드'] = display_df['종목코드'].astype(str).str.zfill(6)
            
            if 'ml_pred_proba' in display_df.columns:
                display_df['ml_pred_proba'] = display_df['ml_pred_proba'] * 100
            
            # volatility_score가 없을 때 기본값 설정
            if 'volatility_score' not in display_df.columns:
                display_df['volatility_score'] = 50.0
            
            display_df['등락율'] = ((display_df['현재가'] - display_df['기준일가']) / display_df['기준일가']) * 100
            display_df['현재가(원)_formatted'] = display_df.apply(format_price_with_change, axis=1)
            
            # 전날 종가 대비 등락율 계산 (증권사 표준 방식)
            if '전날종가' in display_df.columns:
                display_df['전날종가대비등락율'] = ((display_df['현재가'] - display_df['전날종가']) / display_df['전날종가']) * 100
                display_df['등락율(%)'] = display_df['전날종가대비등락율'].apply(format_change_rate)
            else:
                # 전날종가 데이터가 없는 경우 기존 로직 사용 (분석기준일 대비)
                display_df['등락율(%)'] = display_df['등락율'].apply(format_change_rate)

            rename_map = { '현재가': '현재가(원)', '시가총액': '시가총액(억)',  'volatility_score': '변동성(점)', 'ml_pred_proba': '상승확률(%)', 'final_score': '최종점수(점)', '기준일가': '기준일가(원)'}
            display_df.rename(columns=rename_map, inplace=True)
            
            display_columns = [ '최종순위', '종목명', '종목코드', '현재가(원)_formatted', '등락율(%)', '기준일가(원)', '최종점수(점)', '상승확률(%)', '변동성(점)', '시가총액(억)']
            
            result_df = display_df[[col for col in display_columns if col in display_df.columns] + ['등락율']].rename(columns={'현재가(원)_formatted': '현재가(원)'})
            
            return result_df, market_condition, analysis_date
        except Exception as e:
            log_warning(f"기존 분석 결과를 불러오는 데 실패했습니다: {e}")
            return None, None, None
    return None, None, None

def create_stock_chart(ticker_code, stock_name):
    """지정된 종목의 상세 기술적 분석 차트를 생성합니다."""
    try:
        # Ensure ticker_code is a 6-digit string (이미 패딩된 경우 중복 패딩 방지)
        padded_ticker_code = str(ticker_code).zfill(6)

        # 데이터는 2년치를 불러와서 장기 이동평균선 계산에 사용
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        # 하이브리드 방식으로 주가 데이터 수집 (Yahoo Finance → KRX → NAVER)
        df = None
        try:
            df = fdr.DataReader(padded_ticker_code, start_date, end_date)
        except:
            try:
                df = fdr.DataReader(f'KRX:{padded_ticker_code}', start_date, end_date)
            except:
                try:
                    df = fdr.DataReader(f'NAVER:{padded_ticker_code}', start_date, end_date)
                except:
                    df = None
        
        if df is None or df.empty:
            return None

        # 이동평균선(MA) 계산
        df.ta.bbands(length=20, std=2, append=True)
        
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA120'] = df['Close'].rolling(window=120).mean()
        df['MA240'] = df['Close'].rolling(window=240).mean()

        # 네이버 차트 스타일로 개선된 서브플롯 설정
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.02,  # 간격 줄임
            row_heights=[0.75, 0.25],  # 가격 차트 비율 증가
            subplot_titles=('', '')  # 제목 제거로 공간 확보
        )

        # 캔들스틱 (네이버 스타일 색상)
        fig.add_trace(go.Candlestick(
            x=df.index, 
            open=df['Open'], 
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'], 
            name='캔들스틱',
            increasing_line_color='#FF0000',  # 빨간색 (상승)
            decreasing_line_color='#0000FF',  # 파란색 (하락)
            increasing_fillcolor='#FF0000',
            decreasing_fillcolor='#0000FF'
        ), row=1, col=1)
        
        # 이동평균선 (흰 바탕에서 잘 보이는 진한 색상)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], name='MA5', line=dict(color='#FF0000', width=1)), row=1, col=1)  # 빨간색
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='#FFD700', width=1)), row=1, col=1)  # 진한 노란색
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='MA60', line=dict(color='#006400', width=1)), row=1, col=1)  # 진한 초록색
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], name='MA120', line=dict(color='#FF1493', width=2)), row=1, col=1)  # 핫핑크 굵게
        fig.add_trace(go.Scatter(x=df.index, y=df['MA240'], name='MA240', line=dict(color='#000000', width=2)), row=1, col=1)  # 검은색 굵게
        
        # 볼린저 밴드 (반투명하게)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0_2.0'], name='BB상단', line=dict(color='#888888', width=0.8, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0_2.0'], name='BB하단', line=dict(color='#888888', width=0.8, dash='dot'), 
                                fill='tonexty', fillcolor='rgba(136,136,136,0.05)'), row=1, col=1)
        
        # 거래량 (네이버 스타일 + 상세 호버 정보)
        colors = ['#FF0000' if row['Close'] > row['Open'] else '#0000FF' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index, 
            y=df['Volume'], 
            name='거래량', 
            marker_color=colors, 
            opacity=0.7,
            hovertemplate='<b>%{x}</b><br>' +
                         '거래량: %{y:,.0f}주<br>' +
                         '거래대금: %{customdata:,.0f}원<br>' +
                         '<extra></extra>',
            customdata=df['Volume'] * df['Close']  # 거래대금 계산
        ), row=2, col=1)
        
        # 초기 차트 기간 6개월 및 모든 휴장일 공백 제거
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        missing_dates = full_date_range.difference(df.index)

        six_months_ago = df.index.max() - timedelta(days=183) # 약 6개월
        fig.update_xaxes(
            range=[six_months_ago, df.index.max()],
            rangebreaks=[
                dict(values=missing_dates)  # 주말 및 공휴일을 모두 제외
            ]
        )

        # 네이버 차트 스타일 레이아웃 (인터랙티브 기능 개선)
        fig.update_layout(
            title=dict(
                text=f'<b>{stock_name} ({padded_ticker_code})</b>',
                x=0.5,
                font=dict(size=16, color='#333333')
            ),
            height=600,  # 차트 높이 증가
            margin=dict(l=50, r=50, t=60, b=50),  # 여백 최적화
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.01, 
                xanchor="right", 
                x=1,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.8)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            # 인터랙티브 기능 개선
            hovermode='x unified',  # X축 기준 통합 호버
            dragmode='pan',  # 기본 드래그 모드를 팬으로 설정
            selectdirection='d'  # 선택 방향 설정 (d=diagonal)
        )
        
        # Y축 설정 개선
        fig.update_yaxes(
            title_text="가격 (원)", 
            row=1, col=1,
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        )
        fig.update_yaxes(
            title_text="거래량", 
            row=2, col=1,
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        )
        
        # X축 설정 개선
        fig.update_xaxes(
            tickfont=dict(size=10),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        )
        return fig
    except Exception as e:
        log_error(f"차트 생성 중 오류 발생: {e}")
        return None

# =============================================================================
# 웹 페이지 라우트 정의 - 사용자가 접근할 수 있는 페이지들
# =============================================================================

@app.route('/')
def index():
    """메인 페이지 - 주식 추천 시스템"""
    # 기존 분석 결과가 있으면 로드
    analysis_result, market_condition, analysis_date = load_cached_analysis_result()
    
    # 현재 날짜를 템플릿에 전달
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    return render_template('index.html', 
                         analysis_result=analysis_result,
                         market_condition=market_condition,
                         analysis_date=analysis_date,
                         current_date=current_date)

@app.route('/model_analysis')
def model_analysis():
    """학습 모델 분석 페이지"""
    model_info = None
    error = None
    
    try:
        # 메모리 사용량 확인
        import psutil
        import gc
        memory_usage = psutil.virtual_memory()
        if memory_usage.percent > 85:
            raise MemoryError(f"메모리 사용량이 높습니다: {memory_usage.percent:.1f}%")
        
        # cuML 모델 파일 우선 확인, 없으면 기존 모델 파일 확인
        model_path = None
        metadata_path = None
        is_cuml_model = False
        
        if os.path.exists(CUML_MODEL_PATH):
            model_path = CUML_MODEL_PATH
            metadata_path = str(path_manager.data_dir / 'cuml_ensemble_model_metadata.joblib')
            is_cuml_model = True
        elif os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH
            metadata_path = str(path_manager.data_dir / 'model_metadata.joblib')
            is_cuml_model = False
        else:
            error = f"모델 파일을 찾을 수 없습니다. (cuML: {CUML_MODEL_PATH}, 기존: {MODEL_PATH})"
            return render_template('model_analysis.html', model_info=None, error=error)
        
        # 메타데이터 파일이 있으면 메타데이터만 로드 (메모리 최적화)
        # 없으면 모델 파일에서 정보 추출 (메모리 많이 사용)
        if os.path.exists(metadata_path):
            try:
                model_data = joblib.load(metadata_path)
                log_info("메타데이터 파일에서 모델 정보 로드 (메모리 최적화)")
                # 디버깅: 메타데이터 파일 내용 확인
                log_info(f"메타데이터 파일 키: {list(model_data.keys())}")
                if 'training_config' in model_data:
                    log_info(f"training_config 키: {list(model_data['training_config'].keys())}")
                if 'optimization_results' in model_data:
                    log_info(f"optimization_results 키: {list(model_data['optimization_results'].keys())}")
            except Exception as e:
                log_warning(f"메타데이터 파일 로드 실패: {e}. 모델 파일에서 로드합니다.")
                # 메타데이터 로드 실패 시 모델 파일에서 로드 (기존 방식)
                model_data = joblib.load(model_path)
                
                # 메모리 사용량 재확인
                memory_usage = psutil.virtual_memory()
                if memory_usage.percent > 90:
                    del model_data
                    gc.collect()
                    raise MemoryError(f"모델 로드 후 메모리 사용량이 너무 높습니다: {memory_usage.percent:.1f}%")
        else:
            # 메타데이터 파일이 없으면 모델 파일에서 로드 (기존 방식, 메모리 많이 사용)
            log_warning("메타데이터 파일이 없습니다. 모델 파일에서 직접 로드합니다 (메모리 많이 사용).")
            log_warning("다음 학습 시 메타데이터 파일이 자동 생성되어 메모리 사용량이 크게 줄어듭니다.")
            
            # 파일 크기 확인
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if file_size_mb > 1000:  # 1GB 이상
                log_warning(f"모델 파일이 매우 큽니다 ({file_size_mb:.1f}MB). 메모리 부족 위험이 있습니다.")
            
            model_data = joblib.load(model_path)
            
            # 메모리 사용량 재확인
            memory_usage = psutil.virtual_memory()
            if memory_usage.percent > 90:
                del model_data
                gc.collect()
                raise MemoryError(f"모델 로드 후 메모리 사용량이 너무 높습니다: {memory_usage.percent:.1f}%")
            
            # 메타데이터 파일 생성 시도 (다음 접근 시 메모리 절약)
            try:
                # cuML 모델인 경우 (앙상블 또는 단일 모델)
                if is_cuml_model and 'model_type' in model_data:
                    model_type = model_data.get('model_type', 'single_model')
                    metadata_to_save = {
                        'features': model_data.get('features', []),
                        'best_params': model_data.get('best_params', {}),
                        'model_type': model_type,
                        'optimization_results': model_data.get('optimization_results', {}),
                        'training_config': model_data.get('training_config', {}),
                        'feature_importances': model_data.get('feature_importances', None),  # SHAP 값으로 계산된 피처 중요도
                        'parameter_explanations': {
                            'n_estimators': 'RandomForest가 만들 트리의 개수',
                            'max_depth': '각 트리의 최대 깊이 (과적합 방지)',
                            'min_samples_split': '노드 분할에 필요한 최소 샘플 수',
                            'min_samples_leaf': '리프 노드의 최소 샘플 수',
                            'max_samples': '각 트리가 사용할 샘플 비율',
                            'max_features': '각 분할에서 사용할 최대 피처 비율',
                            'split_criterion': '분할 기준 (0: Gini, 1: Entropy)'
                        }
                    }
                else:
                    # 기존 모델인 경우
                    metadata_to_save = {
                        'features': model_data.get('features', []),
                        'training_config': model_data.get('training_config', {}),
                        'optimization_results': model_data.get('optimization_results', {}),
                        'feature_importances': model_data.get('feature_importances', None),  # 피처 중요도 (있을 경우)
                        'parameter_explanations': model_data.get('parameter_explanations', {})
                    }
                
                joblib.dump(metadata_to_save, metadata_path, compress=3)
                log_info(f"메타데이터 파일이 생성되었습니다: {metadata_path} (다음 접근 시 메모리 절약)")
            except Exception as e:
                log_warning(f"메타데이터 파일 생성 실패 (선택사항): {e}")
        
        # 메타데이터 파일인지 확인 ('model' 키가 없으면 메타데이터 파일)
        is_metadata_file = 'model' not in model_data and 'models' not in model_data
        
        # cuML 모델인지 확인 (앙상블 또는 단일 모델)
        is_cuml_single_model = is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'single_model'
        is_cuml_ensemble_model = is_cuml_model and 'model_type' in model_data and model_data['model_type'] == 'mini_batch_ensemble'
        
        # cuML 모델 처리 (메타데이터 파일 또는 모델 파일 모두 지원)
        if is_cuml_ensemble_model or is_cuml_single_model:
            # cuML 모델 처리 (메모리 최적화: 모델 객체 접근 완전 차단)
            
            # 필요한 정보만 먼저 추출 (모델 객체는 절대 추출하지 않음)
            # 주의: model_data['models'] 또는 model_data['model']을 접근하면 모든 모델이 메모리에 로드됨
            features = model_data.get('features', [])
            training_config = model_data.get('training_config', {})
            optimization_results = model_data.get('optimization_results', {})
            best_params = model_data.get('best_params', {})
            parameter_explanations = model_data.get('parameter_explanations', {})
            feature_importances = model_data.get('feature_importances', None)  # SHAP 값으로 계산된 피처 중요도
            permutation_importances = model_data.get('permutation_importances', None)  # 순열 중요도
            
            # 디버깅: 피처 중요도 로드 확인
            if feature_importances is None:
                log_warning("⚠️ 메타데이터 파일에 SHAP 피처 중요도가 없습니다. SHAP 계산이 실패했거나 이전 모델일 수 있습니다.")
            else:
                log_info(f"✅ SHAP 피처 중요도 로드 완료: {len(feature_importances)}개 피처")
            
            if permutation_importances is None:
                log_warning("⚠️ 메타데이터 파일에 순열 중요도가 없습니다. 순열 중요도 계산이 실패했거나 이전 모델일 수 있습니다.")
            else:
                log_info(f"✅ 순열 중요도 로드 완료: {len(permutation_importances)}개 피처")
            
            # 모델 개수는 training_config에서 가져오거나, 없으면 기본값 사용
            # (model_data['models']에 접근하지 않음 - 메모리 최적화)
            if is_cuml_ensemble_model:
                n_models = training_config.get('n_final_models') or training_config.get('n_mini_batches', 5)
                model_type_str = f"cuML 앙상블 ({n_models}개 모델)"
            else:
                n_models = 1
                model_type_str = "cuML 단일 모델"
            
            # model_data에서 models, model, scaler, imputation_values는 메모리를 많이 사용하므로 즉시 삭제
            # (이미 필요한 정보는 추출했으므로)
            if 'models' in model_data:
                del model_data['models']
            if 'model' in model_data:
                del model_data['model']
            if 'scaler' in model_data:
                del model_data['scaler']
            if 'imputation_values' in model_data:
                del model_data['imputation_values']
            del model_data
            gc.collect()
            
            # 피처 중요도는 메타데이터 파일에서 로드 (SHAP 값으로 계산된 경우)
            # 메타데이터에 없으면 None (모델 객체 접근 없이 처리)
            
            # 파라미터는 best_params에서 가져옴 (모델 객체 접근 없음)
            model_params = best_params.copy() if best_params else {}
            
            # optimization_results가 비어있거나 불완전하면 best_params로부터 구성
            if not optimization_results:
                optimization_results = {}
            
            if 'best_params' not in optimization_results and best_params:
                optimization_results['best_params'] = best_params
            
            if 'best_score' not in optimization_results:
                optimization_results['best_score'] = None
            
            if 'total_combinations_tested' not in optimization_results:
                optimization_results['total_combinations_tested'] = None
            
            # training_config가 비어있거나 불완전하면 기본값 구성
            if not training_config:
                training_config = {}
            
            # GPU 버전에서는 사용하지 않는 필드들에 대한 기본값 설정
            if 'n_jobs' not in training_config:
                training_config['n_jobs'] = None  # GPU 버전은 CPU 코어 사용 안 함
            
            if 'test_size' not in training_config:
                training_config['test_size'] = None  # GPU 버전은 교차검증 사용
            
            # 필수 필드들 기본값 설정
            if 'search_method' not in training_config:
                training_config['search_method'] = 'Optuna (TPE Sampler)'
            
            if 'scoring' not in training_config:
                training_config['scoring'] = 'roc_auc'
            
            if 'cv_folds' not in training_config:
                training_config['cv_folds'] = 3
            
            if 'n_iter' not in training_config:
                training_config['n_iter'] = None
            
            if 'max_depth_candidates' not in training_config:
                training_config['max_depth_candidates'] = []
            
            # 모델 목표 정보 기본값 설정 (하위 호환성)
            if 'target_days' not in training_config:
                training_config['target_days'] = 10  # 기본값: 10거래일
            if 'target_percentage' not in training_config:
                training_config['target_percentage'] = 8  # 기본값: 8%
            if 'target_description' not in training_config:
                training_config['target_description'] = f"{training_config.get('target_days', 10)}거래일 사이 한번이라도 {training_config.get('target_percentage', 8)}% 상승이 있었는지 확인"
            
            # 모델 정보 (메모리 최적화: 모델 객체 생성 없이 필요한 정보만 저장)
            model_info = {
                'model_type': model_type_str,
                'model_path': model_path,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
                'oob_score': None,  # cuML은 OOB 지원 안 함
                'features': features,
                'feature_importances': feature_importances,  # SHAP 값으로 계산된 피처 중요도 (있을 경우)
                'permutation_importances': permutation_importances,  # 순열 중요도 (있을 경우)
                'params': model_params,  # best_params 사용 (모델 객체 접근 없음)
                'training_config': training_config,
                'optimization_results': optimization_results,
                'parameter_explanations': parameter_explanations,
                'n_models': n_models
            }
            
            # 모델 객체는 생성하지 않았으므로 삭제 불필요
            # models와 scaler는 이미 model_data에서 삭제했으므로 추가 삭제 불필요
            gc.collect()
        elif is_metadata_file:
            # 메타데이터 파일인데 cuML 모델이 아닌 경우 (기존 CPU 모델의 메타데이터)
            features = model_data.get('features', [])
            training_config = model_data.get('training_config', {})
            optimization_results = model_data.get('optimization_results', {})
            parameter_explanations = model_data.get('parameter_explanations', {})
            feature_importances = model_data.get('feature_importances', None)  # 메타데이터에 저장된 피처 중요도
            permutation_importances = model_data.get('permutation_importances', None)  # 순열 중요도
            
            # best_params는 optimization_results에서 가져오거나 직접 가져오기
            best_params = model_data.get('best_params', {})
            if not best_params and optimization_results:
                best_params = optimization_results.get('best_params', {})
            
            # 모델 목표 정보 기본값 설정 (하위 호환성)
            if 'target_days' not in training_config:
                training_config['target_days'] = 10  # 기본값: 10거래일
            if 'target_percentage' not in training_config:
                training_config['target_percentage'] = 8  # 기본값: 8%
            if 'target_description' not in training_config:
                training_config['target_description'] = f"{training_config.get('target_days', 10)}거래일 사이 한번이라도 {training_config.get('target_percentage', 8)}% 상승이 있었는지 확인"
            
            del model_data
            gc.collect()
            
            # 모델 정보 (메타데이터만 사용, 모델 객체 없음)
            model_info = {
                'model_type': '기존 모델 (메타데이터)',
                'model_path': model_path,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
                'oob_score': None,
                'features': features,
                'feature_importances': feature_importances,  # 메타데이터에 저장된 피처 중요도 (있을 경우)
                'permutation_importances': permutation_importances,  # 순열 중요도 (있을 경우)
                'params': best_params,
                'training_config': training_config,
                'optimization_results': optimization_results,
                'parameter_explanations': parameter_explanations
            }
        else:
            # 기존 모델 구조 (sklearn, 메모리 최적화)
            # 메타데이터 파일이 아닌 경우에만 모델 객체 접근
            model = model_data['model']
            features = model_data['features']
            training_config = model_data.get('training_config', {})
            optimization_results = model_data.get('optimization_results', {})
            parameter_explanations = model_data.get('parameter_explanations', {})
            
            # 모델 목표 정보 기본값 설정 (하위 호환성)
            if 'target_days' not in training_config:
                training_config['target_days'] = 10  # 기본값: 10거래일
            if 'target_percentage' not in training_config:
                training_config['target_percentage'] = 8  # 기본값: 8%
            if 'target_description' not in training_config:
                training_config['target_description'] = f"{training_config.get('target_days', 10)}거래일 사이 한번이라도 {training_config.get('target_percentage', 8)}% 상승이 있었는지 확인"
            
            # model_data에서 필요한 정보 추출 후 즉시 삭제
            del model_data
            gc.collect()
            
            # 피처 중요도 데이터 정리 (CPU/GPU 모델 호환)
            feature_importances = None
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    # cuML 모델도 feature_importances_를 제공하지만, cuDF Series나 다른 형태로 반환될 수 있음
                    if hasattr(importances, 'to_pandas'):
                        # cuDF Series인 경우 pandas로 변환
                        importances = importances.to_pandas().values
                    elif hasattr(importances, 'values'):
                        # pandas Series나 다른 형태인 경우
                        importances = importances.values
                    elif hasattr(importances, 'to_numpy'):
                        # numpy 배열로 변환 가능한 경우
                        importances = importances.to_numpy()
                    elif not isinstance(importances, np.ndarray):
                        # 기타 타입은 numpy 배열로 변환
                        importances = np.array(importances)
                    
                    # numpy 배열로 확실히 변환
                    if not isinstance(importances, np.ndarray):
                        importances = np.array(importances)
                    
                    # 피처와 중요도 매칭
                    if len(importances) == len(features):
                        feature_importances = list(zip(features, importances))
                        feature_importances.sort(key=lambda x: x[1], reverse=True)
                    else:
                        # 길이가 맞지 않는 경우 경고
                        log_warning(f"피처 중요도 길이 불일치: features={len(features)}, importances={len(importances)}")
                        feature_importances = None
            except (AttributeError, TypeError, ValueError) as e:
                # cuML 모델 등 feature_importances_ 처리 실패
                log_warning(f"피처 중요도 추출 실패: {e}")
                feature_importances = None
            
            # 모델 정보 (메모리 최적화: 필요한 정보만 저장)
            # 순열 중요도는 모델 파일에서 직접 로드할 수 없으므로 None (메타데이터 파일에만 저장됨)
            model_info = {
                'model_type': type(model).__name__,
                'model_path': model_path,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
                'oob_score': getattr(model, 'oob_score_', None),
                'features': features,
                'feature_importances': feature_importances,
                'permutation_importances': None,  # 모델 파일 직접 로드 시에는 순열 중요도 없음 (메타데이터 파일에만 저장)
                'params': model.get_params() if hasattr(model, 'get_params') else {},
                'training_config': training_config,
                'optimization_results': optimization_results,
                'parameter_explanations': parameter_explanations
            }
            
            # 모델 객체는 더 이상 필요 없으므로 삭제
            del model
            gc.collect()
        
    except FileNotFoundError:
        error = "모델 파일을 찾을 수 없습니다. 먼저 모델을 학습해주세요."
    except MemoryError as e:
        error = f"메모리 부족으로 모델을 로드할 수 없습니다: {str(e)}"
    except Exception as e:
        error = f"모델 로드 중 오류: {str(e)}"
        import traceback
        log_error(f"모델 분석 페이지 오류: {traceback.format_exc()}")
    
    return render_template('model_analysis.html', model_info=model_info, error=error)

@app.route('/backtest')
def backtest():
    """백테스팅 페이지"""
    # 기존 백테스팅 리포트가 있는지 확인 (JSON만)
    json_report_path = str(path_manager.data_dir / 'backtest_report.json')
    has_report = os.path.exists(json_report_path)
    
    return render_template('backtest.html', has_report=has_report)

# =============================================================================
# API 엔드포인트들 - 웹 페이지와 백엔드 간의 데이터 통신
# =============================================================================

@app.route('/api/start_analysis', methods=['POST'])
def start_analysis():
    """분석 시작 API"""
    try:
        data = request.get_json()
        analysis_date = data.get('analysis_date')
        
        if not analysis_date:
            return jsonify({'error': '분석 기준일이 필요합니다.'}), 400
        
        # 기존 프로세스 정리
        global current_analysis_process
        if current_analysis_process and is_process_running(current_analysis_process):
            cleanup_analysis_process()
        
        # 분석 중복 실행 방지 (개선된 체크)
        if get_analysis_running():
            return jsonify({'error': '이미 분석이 실행 중입니다. 완료될 때까지 기다려주세요.'}), 400
        
        # 분석 실행 중 플래그 설정
        set_analysis_running(True)
        
        # 분석 프로세스 시작 (subprocess.Popen 방식)
        def run_analysis_process():
            global current_analysis_process
            process = None
            try:
                # 분석 스크립트 실행
                command = [
                    sys.executable, '-u', 
                    os.path.join(os.path.dirname(__file__), 'scripts', 'run_analysis.py'),
                    '--date', analysis_date
                ]
                
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['LANG'] = 'ko_KR.UTF-8'
                env['LC_ALL'] = 'ko_KR.UTF-8'
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    env=env
                )
                
                # 현재 분석 프로세스 저장
                current_analysis_process = process
                
                # 실시간 로그 전송
                PROGRESS_REGEX = re.compile(r'\[PROGRESS\].*\(\d+/\d+ - \d+\.\d+%\)')
                last_line_was_tqdm = False
                last_log_time = time.time()
                process_timeout = 7200  # 2시간 타임아웃
                no_output_timeout = 600  # 10분간 출력이 없으면 타임아웃
                
                # 프로세스 상태 모니터링을 위한 변수
                process_start_time = time.time()
                last_activity_time = time.time()
                
                while True:
                    # 프로세스가 종료되었는지 확인
                    if process.poll() is not None:
                        # 프로세스가 종료됨
                        break
                    
                    # 전체 타임아웃 확인 (2시간)
                    elapsed_time = time.time() - process_start_time
                    if elapsed_time > process_timeout:
                        error_msg = f"분석 프로세스가 타임아웃되었습니다 (최대 {process_timeout//60}분 초과)"
                        socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                        return
                    
                    # 출력이 없는 시간 확인 (10분)
                    if time.time() - last_activity_time > no_output_timeout:
                        error_msg = f"분석 프로세스가 응답하지 않습니다 (10분간 출력 없음)"
                        socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                        return
                    
                    # stdout에서 읽기 (논블로킹)
                    try:
                        # Windows에서는 select가 작동하지 않으므로 직접 읽기 시도
                        line = process.stdout.readline()
                        if line:
                            last_activity_time = time.time()
                            last_log_time = time.time()
                            
                            # 터미널 출력 (진행률 메시지만 특별 처리)
                            if PROGRESS_REGEX.search(line):
                                # 진행률 메시지만 덮어쓰기 처리
                                sys.stdout.write(line.strip() + '\r')
                                last_line_was_tqdm = True
                            else:
                                if last_line_was_tqdm:
                                    sys.stdout.write('\n')
                                sys.stdout.write(line)
                                last_line_was_tqdm = False
                            sys.stdout.flush()
                            
                            # 이모지 대체
                            emoji_replacements = {
                                '🎉': '[SUCCESS]',
                                '✅': '[OK]',
                                '⚠️': '[WARN]',
                                '🔄': '[PROC]',
                                '🌐': '[NET]',
                                '📅': '[DATE]',
                                '❌': '[ERROR]',
                                '🔍': '[SEARCH]',
                                '💾': '[SAVE]',
                                '📊': '[DATA]',
                                '💰': '[PRICE]',
                                '📈': '[CHART]',
                                '🎯': '[TARGET]',
                                '📋': '[LIST]',
                                '🔧': '[TOOL]',
                                '⚡': '[FAST]',
                                '🛡️': '[SAFE]',
                                '🎪': '[SHOW]',
                                '🏆': '[WIN]',
                                '💡': '[IDEA]'
                            }
                            
                            # 이모지 대체 적용
                            processed_line = line.strip()
                            for emoji, replacement in emoji_replacements.items():
                                processed_line = processed_line.replace(emoji, replacement)
                            
                            # 진행률 메시지 감지 및 접두사 추가
                            if PROGRESS_REGEX.search(line):
                                message = f"[PROGRESS] {processed_line}"
                            else:
                                message = processed_line
                            
                            # WebSocket으로 로그 전송 (터미널과 동시)
                            socketio.emit('analysis_log', {'message': message})
                        else:
                            # 더 이상 읽을 데이터가 없으면 잠시 대기
                            socketio.sleep(0.1)
                    except Exception as e:
                        # 읽기 오류 발생 시 로그 출력
                        error_msg = f"프로세스 출력 읽기 오류: {e}"
                        socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                        socketio.sleep(0.1)
                
                # 프로세스 종료 대기 (타임아웃 10초)
                try:
                    process.stdout.close()
                    return_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # 타임아웃 발생 시 강제 종료
                    process.kill()
                    return_code = process.wait()
                    error_msg = "분석 프로세스가 정상적으로 종료되지 않아 강제 종료되었습니다."
                    socketio.emit('analysis_log', {'message': f"[ERROR] {error_msg}"})
                    socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                    return
                
                if return_code == 0:
                    socketio.emit('analysis_complete', {'success': True})
                else:
                    error_msg = f'분석 실행 중 오류가 발생했습니다. (종료 코드: {return_code})'
                    socketio.emit('analysis_complete', {'success': False, 'error': error_msg})
                    
            except Exception as e:
                socketio.emit('analysis_complete', {'success': False, 'error': str(e)})
            finally:
                # 분석 완료 시 플래그 해제 및 프로세스 초기화
                set_analysis_running(False)
                current_analysis_process = None
        
        # 백그라운드에서 분석 실행 (SocketIO 컨텍스트 유지)
        socketio.start_background_task(run_analysis_process)
        
        return jsonify({'message': '분석이 시작되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시 플래그 해제
        set_analysis_running(False)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_analysis', methods=['POST'])
def stop_analysis():
    """분석 중단 API"""
    try:
        global current_analysis_process
        
        # 실제 프로세스 상태 확인
        if current_analysis_process is None or not is_process_running(current_analysis_process):
            # 이미 종료된 프로세스인 경우 플래그만 해제
            cleanup_analysis_process()
            return jsonify({'message': '실행 중인 분석이 없습니다.'})
        
        # 프로세스 종료
        current_analysis_process.terminate()
        try:
            current_analysis_process.wait(timeout=5)  # 타임아웃 증가
        except subprocess.TimeoutExpired:
            current_analysis_process.kill()
            current_analysis_process.wait()
        
        # 플래그 해제 및 프로세스 초기화
        cleanup_analysis_process()
        
        # WebSocket으로 중단 알림
        socketio.emit('analysis_complete', {'success': False, 'error': '사용자에 의해 분석이 중단되었습니다.'})
        
        return jsonify({'message': '분석이 중단되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시에도 플래그 강제 해제
        cleanup_analysis_process()
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_backtest', methods=['POST'])
def start_backtest():
    """백테스팅 시작 API"""
    try:
        data = request.get_json()
        
        # 백테스팅 파라미터 검증
        required_params = ['capital', 'max_hold', 'take_profit', 'stop_loss', 'top_n', 'buy_universe', 'transaction_fee']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f'{param} 파라미터가 필요합니다.'}), 400
        
        # 날짜 파라미터 검증 (선택적이지만 제공되면 유효성 검증)
        from datetime import datetime as dt
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if start_date and end_date:
            try:
                start_dt = dt.strptime(start_date, '%Y-%m-%d')
                end_dt = dt.strptime(end_date, '%Y-%m-%d')
                today = dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
                
                if end_dt > today:
                    return jsonify({'error': '종료일은 오늘 이전이어야 합니다.'}), 400
                
                if start_dt >= end_dt:
                    return jsonify({'error': '시작일은 종료일보다 이전이어야 합니다.'}), 400
            except ValueError as e:
                return jsonify({'error': f'날짜 형식이 올바르지 않습니다: {str(e)}'}), 400
        
        # 기존 백테스팅 프로세스 정리
        global current_backtest_process
        if current_backtest_process and is_backtest_process_running(current_backtest_process):
            cleanup_backtest_process()
        
        # 백테스팅 중복 실행 방지
        if get_backtest_running():
            return jsonify({'error': '이미 백테스팅이 실행 중입니다. 완료될 때까지 기다려주세요.'}), 400
        
        # 백테스팅 실행 중 플래그 설정
        set_backtest_running(True)
        
        def run_backtest_process():
            global current_backtest_process
            process = None
            try:
                # 백테스팅 스크립트 실행
                command = [
                    sys.executable, '-u',
                    os.path.join(os.path.dirname(__file__), 'scripts', 'backtest.py'),
                    '--capital', str(data['capital']),
                    '--max-hold', str(data['max_hold']),
                    '--take-profit', str(data['take_profit']),
                    '--stop-loss', str(data['stop_loss']),
                    '--top-n', str(data['top_n']),
                    '--buy-universe', str(data['buy_universe']),
                    '--fee', str(data['transaction_fee'])
                ]
                
                # 날짜 파라미터 추가 (제공된 경우)
                if start_date:
                    command.extend(['--start-date', start_date])
                if end_date:
                    command.extend(['--end-date', end_date])
                
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                env['LANG'] = 'ko_KR.UTF-8'
                env['LC_ALL'] = 'ko_KR.UTF-8'
                
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    env=env
                )
                
                # 현재 백테스팅 프로세스 저장
                current_backtest_process = process
                
                # 실시간 로그 전송
                PROGRESS_REGEX = re.compile(r'\[PROGRESS\].*\(\d+/\d+ - \d+\.\d+%\)')
                last_line_was_tqdm = False
                
                for line in iter(process.stdout.readline, ''):
                    # 터미널 출력 (진행률 메시지만 특별 처리)
                    if PROGRESS_REGEX.search(line):
                        # 진행률 메시지만 덮어쓰기 처리
                        sys.stdout.write(line.strip() + '\r')
                        last_line_was_tqdm = True
                    else:
                        if last_line_was_tqdm:
                            sys.stdout.write('\n')
                        sys.stdout.write(line)
                        last_line_was_tqdm = False
                    sys.stdout.flush()
                    
                    # 이모지 대체
                    emoji_replacements = {
                        '🎉': '[SUCCESS]',
                        '✅': '[OK]',
                        '⚠️': '[WARN]',
                        '🔄': '[PROC]',
                        '🌐': '[NET]',
                        '📅': '[DATE]',
                        '❌': '[ERROR]',
                        '🔍': '[SEARCH]',
                        '💾': '[SAVE]',
                        '📊': '[DATA]',
                        '💰': '[PRICE]',
                        '📈': '[CHART]',
                        '🎯': '[TARGET]',
                        '📋': '[LIST]',
                        '🔧': '[TOOL]',
                        '⚡': '[FAST]',
                        '🛡️': '[SAFE]',
                        '🎪': '[SHOW]',
                        '🏆': '[WIN]',
                        '💡': '[IDEA]'
                    }
                    
                    # 이모지 대체 적용
                    processed_line = line.strip()
                    for emoji, replacement in emoji_replacements.items():
                        processed_line = processed_line.replace(emoji, replacement)
                    
                    # 진행률 메시지 감지 및 접두사 추가
                    if PROGRESS_REGEX.search(line):
                        message = f"[PROGRESS] {processed_line}"
                    else:
                        message = processed_line
                    
                    # WebSocket으로 로그 전송 (터미널과 동시)
                    socketio.emit('backtest_log', {'message': message})
                    
                    # 실행 양보 (이벤트 루프가 블로킹되지 않도록)
                    socketio.sleep(0.01)
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    socketio.emit('backtest_complete', {'success': True})
                else:
                    socketio.emit('backtest_complete', {'success': False, 'error': '백테스팅 실행 중 오류가 발생했습니다.'})
                    
            except Exception as e:
                socketio.emit('backtest_complete', {'success': False, 'error': str(e)})
            finally:
                # 백테스팅 완료 시 플래그 해제 및 프로세스 초기화
                set_backtest_running(False)
                current_backtest_process = None
        
        # 백그라운드에서 백테스팅 실행 (SocketIO 컨텍스트 유지)
        socketio.start_background_task(run_backtest_process)
        
        return jsonify({'message': '백테스팅이 시작되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시 플래그 해제
        set_backtest_running(False)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_backtest', methods=['POST'])
def stop_backtest():
    """백테스팅 중단 API"""
    try:
        global current_backtest_process
        
        # 실제 프로세스 상태 확인
        if current_backtest_process is None or not is_backtest_process_running(current_backtest_process):
            # 이미 종료된 프로세스인 경우 플래그만 해제
            cleanup_backtest_process()
            return jsonify({'message': '실행 중인 백테스팅이 없습니다.'})
        
        # 프로세스 종료
        current_backtest_process.terminate()
        try:
            current_backtest_process.wait(timeout=5)  # 타임아웃 증가
        except subprocess.TimeoutExpired:
            current_backtest_process.kill()
            current_backtest_process.wait()
        
        # 플래그 해제 및 프로세스 초기화
        cleanup_backtest_process()
        
        # WebSocket으로 중단 알림
        socketio.emit('backtest_complete', {'success': False, 'error': '사용자에 의해 백테스팅이 중단되었습니다.'})
        
        return jsonify({'message': '백테스팅이 중단되었습니다.'})
        
    except Exception as e:
        # 오류 발생 시에도 플래그 강제 해제
        cleanup_backtest_process()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_chart/<ticker_code>')
def get_stock_chart(ticker_code):
    """종목 차트 데이터 API"""
    try:
        # 종목명 가져오기 (종목코드 정규화)
        stock_list_df = data_fetcher.fetch_stock_list()
        normalized_ticker = str(ticker_code).zfill(6)
        stock_info = stock_list_df[stock_list_df['종목코드'] == normalized_ticker]
        if stock_info.empty:
            return jsonify({'error': '종목을 찾을 수 없습니다.'}), 404
        
        stock_name = stock_info.iloc[0]['종목명']
        
        # 차트 생성
        fig = create_stock_chart(normalized_ticker, stock_name)
        if fig is None:
            return jsonify({'error': '차트를 생성할 수 없습니다.'}), 500
        
        # Plotly JSON으로 변환
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'chart': chart_json})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_features/<ticker_code>')
def get_stock_features(ticker_code):
    """종목 피처 데이터 API"""
    try:
        cached_features_path = os.path.join(str(path_manager.data_dir), 'cached_features.json')
        if not os.path.exists(cached_features_path):
            return jsonify({'error': '피처 데이터를 찾을 수 없습니다.'}), 404
        
        cached_features_df = pd.read_json(cached_features_path, orient='records', dtype={'종목코드': str})
        selected_stock_features = cached_features_df[cached_features_df['종목코드'] == str(ticker_code).zfill(6)]
        
        if selected_stock_features.empty:
            return jsonify({'error': '해당 종목의 피처 데이터를 찾을 수 없습니다.'}), 404
        
        # 학습 모델에서 사용하는 피처 리스트 (train_gpu_main.py와 동일)
        model_features = [
            'log_mktcap',
            '52주_신고가_비율',
            'ADX_14',
            'disparity_120',  # 120일 이격도
            'disparity_240',  # 240일 이격도
            'KOSPI_disparity_20',  # KOSPI 20일 이격도
            # 추가된 피처
            'Z_Score_20',
            'Position_Range_60',
            'KOSPI_변동성(1M)',
            '변동성(1W)',  # 변동성 1주 (표준편차/평균)
            '변동성(3M)',  # 변동성 3개월 (표준편차/평균)
            'MA120_Slope',  # 120일 이동평균선 기울기
            'MA240_Slope',  # 240일 이동평균선 기울기
            'KOSPI_MA20_Slope',  # KOSPI 20일 이동평균선 기울기
            'PBR_log',  # PBR 로그 변환
            # 새로 추가된 피처
            'RVOL',  # 상대 거래량 (Relative Volume)
            '시총 회전율(1W)',  # 시총 회전율 1주 (5일 평균 거래대금 / 시가총액 * 100)
            '시총 회전율(3M)',  # 시총 회전율 3개월 (60일 평균 거래대금 / 시가총액 * 100)
            'RSI_Signal_Oscillator',  # RSI 신호 오실레이터 (RSI_14 - RSI_14.rolling(9).mean())
            'ATRr_20',  # ATR 비율 20일 (기준 - 1M)
            'ATR_Ratio_Short',  # ATR 비율 단기 (1W / 1M)
            'ATR_Ratio_Trend',  # ATR 비율 추세 (1M / 3M)
            'Eff_Ratio_10'  # 효율성 비율 10일
        ]
        
        # 등락율 계산 등 다른 부분에 영향있는 필수 피처 (표시용)
        essential_features = ['시가', '종가', '현재가', '기준일가', '전날종가', '종목명', '시가총액']
        
        # 학습 모델 피처 + 필수 피처만 필터링
        allowed_features = set(model_features + essential_features)
        
        # 피처 데이터 정리
        display_features = selected_stock_features.drop(columns=['종목코드', 'date'], errors='ignore')
        
        # 학습 모델에서 사용하는 피처만 필터링
        filtered_features = {col: display_features[col] for col in display_features.columns if col in allowed_features}
        
        # 객체를 문자열로 변환
        features_dict = {}
        for column, series in filtered_features.items():
            value = series.iloc[0]
            if pd.isna(value):
                features_dict[column] = "N/A"
            elif isinstance(value, (dict, list)):
                # 중첩된 객체를 JSON 문자열로 변환
                features_dict[column] = json.dumps(value, ensure_ascii=False)
            else:
                # 일반 값은 문자열로 변환
                features_dict[column] = str(value)
        
        return jsonify({'features': features_dict})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest_report')
def get_backtest_report():
    """백테스팅 리포트 JSON 데이터 제공"""
    json_report_path = str(path_manager.data_dir / 'backtest_report.json')
    
    if os.path.exists(json_report_path):
        try:
            with open(json_report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            return jsonify(report_data)
        except Exception as e:
            log_error(f"백테스팅 리포트 로드 실패: {e}")
            return jsonify({'error': '리포트를 로드할 수 없습니다.'}), 500
    else:
        return jsonify({'error': '백테스팅 리포트를 찾을 수 없습니다.'}), 404


@app.route('/favicon.ico')
def favicon():
    """favicon.ico 서빙"""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/api/status')
def get_status():
    """서버 상태 확인 API (디버그용)"""
    return jsonify({
        'analysis_running': get_analysis_running(),
        'backtest_running': get_backtest_running(),
        'process_exists': current_analysis_process is not None,
        'process_running': is_process_running(current_analysis_process) if current_analysis_process else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analysis_status')
def get_analysis_status():
    """분석 상태 상세 확인 API"""
    return jsonify({
        'analysis_running': get_analysis_running(),
        'process_exists': current_analysis_process is not None,
        'process_running': is_process_running(current_analysis_process) if current_analysis_process else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/backtest_status')
def get_backtest_status():
    """백테스팅 상태 상세 확인 API"""
    return jsonify({
        'backtest_running': get_backtest_running(),
        'process_exists': current_backtest_process is not None,
        'process_running': is_backtest_process_running(current_backtest_process) if current_backtest_process else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/feature_correlation/calculate', methods=['POST'])
def calculate_feature_correlation():
    """피처 상관관계 계산 API"""
    try:
        import glob
        from pathlib import Path
        
        # 학습 데이터 경로 확인
        data_path = os.path.expanduser("~/stock_data/processed_feather")
        feather_files = glob.glob(os.path.join(data_path, "*.feather"))
        
        # 학습 모델에서 사용하는 피처 리스트 (train_gpu_main.py와 동일)
        model_features = [
            'log_mktcap',
            '52주_신고가_비율',
            'ADX_14',
            'disparity_120',  # 120일 이격도
            'disparity_240',  # 240일 이격도
            'KOSPI_disparity_20',  # KOSPI 20일 이격도
            # 추가된 피처
            'Z_Score_20',
            'Position_Range_60',
            'KOSPI_변동성(1M)',
            '변동성(1W)',  # 변동성 1주 (표준편차/평균)
            '변동성(3M)',  # 변동성 3개월 (표준편차/평균)
            'MA120_Slope',  # 120일 이동평균선 기울기
            'MA240_Slope',  # 240일 이동평균선 기울기
            'KOSPI_MA20_Slope',  # KOSPI 20일 이동평균선 기울기
            'PBR_log',  # PBR 로그 변환
            # 새로 추가된 피처
            'RVOL',  # 상대 거래량 (Relative Volume)
            '시총 회전율(1W)',  # 시총 회전율 1주 (5일 평균 거래대금 / 시가총액 * 100)
            '시총 회전율(3M)',  # 시총 회전율 3개월 (60일 평균 거래대금 / 시가총액 * 100)
            'RSI_Signal_Oscillator',  # RSI 신호 오실레이터 (RSI_14 - RSI_14.rolling(9).mean())
            'ATRr_20',  # ATR 비율 20일 (기준 - 1M)
            'ATR_Ratio_Short',  # ATR 비율 단기 (1W / 1M)
            'ATR_Ratio_Trend',  # ATR 비율 추세 (1M / 3M)
            'Eff_Ratio_10'  # 효율성 비율 10일
        ]
        
        data_source = None
        correlation_matrix = None
        feature_data = None
        
        # 1단계: 학습 데이터 파일이 있으면 사용
        if feather_files and len(feather_files) > 0:
            log_info(f"학습 데이터 파일 사용: {len(feather_files)}개 파일")
            data_source = "training_data"
            
            # 모든 feather 파일 로드 및 합치기
            dfs = []
            loaded_count = 0
            for feather_file in feather_files[:500]:  # 최대 500개 파일만 로드 (메모리 절약)
                try:
                    df = pd.read_feather(feather_file)
                    # model_features에 있는 피처만 선택
                    available_features = [f for f in model_features if f in df.columns]
                    if available_features:
                        df_subset = df[available_features].copy()
                        dfs.append(df_subset)
                        loaded_count += 1
                except Exception as e:
                    log_warning(f"Feather 파일 로드 실패: {feather_file}, {e}")
                    continue
            
            if dfs:
                feature_data = pd.concat(dfs, ignore_index=True)
                log_info(f"학습 데이터 로드 완료: {len(feature_data):,}행, {loaded_count}개 파일")
            else:
                log_warning("학습 데이터 파일을 로드할 수 없습니다. 실시간 데이터 수집으로 전환합니다.")
                data_source = None
        
        # 2단계: 학습 데이터가 없으면 1년치 실시간 데이터 수집
        if data_source is None or feature_data is None or feature_data.empty:
            log_info("1년치 실시간 데이터 수집 시작...")
            data_source = "realtime_data"
            
            # 1년 전 날짜 계산
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # data_processor를 통해 데이터 수집
            feature_data = data_processor.get_preprocessed_data(start_date, end_date, skip_factor_scores=True)
            
            if feature_data is None or feature_data.empty:
                return jsonify({'error': '데이터를 수집할 수 없습니다.'}), 500
            
            # model_features에 있는 피처만 선택
            available_features = [f for f in model_features if f in feature_data.columns]
            if available_features:
                feature_data = feature_data[available_features].copy()
            else:
                return jsonify({'error': '필요한 피처가 데이터에 없습니다.'}), 500
            
            log_info(f"실시간 데이터 수집 완료: {len(feature_data):,}행")
        
        # 3단계: 상관관계 계산
        log_info("상관관계 계산 중...")
        
        # 숫자형 컬럼만 선택
        numeric_cols = feature_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data_numeric = feature_data[numeric_cols].copy()
        
        # 결측치 처리 (중앙값으로 대체)
        feature_data_numeric = feature_data_numeric.fillna(feature_data_numeric.median())
        
        # 상관관계 계산
        correlation_matrix = feature_data_numeric.corr()
        
        # NaN이나 Inf 값 처리
        correlation_matrix = correlation_matrix.replace([np.inf, -np.inf], np.nan)
        correlation_matrix = correlation_matrix.fillna(0)
        
        # 상관관계 행렬을 JSON으로 변환 가능한 형태로 변환
        correlation_dict = correlation_matrix.to_dict()
        
        # 피처 리스트 (행/열 순서)
        feature_list = list(correlation_matrix.columns)
        
        # 상관관계 높은 쌍 찾기 (|r| > 0.7)
        high_correlation_pairs = []
        for i, feat1 in enumerate(feature_list):
            for j, feat2 in enumerate(feature_list):
                if i < j:  # 중복 방지
                    corr_value = correlation_matrix.loc[feat1, feat2]
                    if abs(corr_value) > 0.7:
                        high_correlation_pairs.append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'correlation': float(corr_value)
                        })
        
        # 상관계수 절댓값 기준으로 정렬
        high_correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        log_info(f"상관관계 계산 완료: {len(feature_list)}개 피처, {len(high_correlation_pairs)}개 높은 상관관계 쌍")
        
        return jsonify({
            'success': True,
            'data_source': data_source,
            'feature_list': feature_list,
            'correlation_matrix': correlation_dict,
            'high_correlation_pairs': high_correlation_pairs,
            'data_rows': len(feature_data),
            'message': f'상관관계 계산 완료 ({data_source}, {len(feature_data):,}행)'
        })
        
    except Exception as e:
        log_error(f"피처 상관관계 계산 중 오류: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# WebSocket 이벤트 핸들러 - 실시간 통신 처리
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 시"""
    emit('connected', {'message': '연결되었습니다.'})

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제 시"""
    pass

# =============================================================================
# 포트 찾기 함수 - 사용 가능한 포트 자동 검색
# =============================================================================

def find_available_port(start_port=5000, max_port=5100):
    """사용 가능한 포트를 찾는 함수"""
    import socket
    
    # 첫 번째 시도: 기본 범위 (5000-5100)
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Windows와 macOS 호환성을 위해 0.0.0.0으로 바인딩 테스트
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    # 두 번째 시도: 확장 범위 (8000-8100)
    print("⚠️  기본 포트 범위(5000-5100)에서 사용 가능한 포트를 찾을 수 없습니다.")
    print("🔍 확장 범위(8000-8100)에서 포트를 검색합니다...")
    
    for port in range(8000, 8100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    # 모든 포트가 사용 중인 경우
    raise RuntimeError(f"포트 {start_port}-{max_port} 및 8000-8100 범위에서 사용 가능한 포트를 찾을 수 없습니다.")

# =============================================================================
# 메인 실행 - 웹 서버 시작
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='AI 주식 분석 시스템 - Flask 앱')
    parser.add_argument('--port', type=int, default=None, help='사용할 포트 번호 (기본값: 자동 검색)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='바인딩할 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    args = parser.parse_args()
    
    # 플래그 초기화
    reset_flags()
    
    # 통일된 경로로 필요한 디렉토리 생성
    ensure_all_directories()
    
    # 포트 결정
    if args.port:
        # 사용자가 지정한 포트 사용
        port = args.port
        print(f"🚀 Flask 앱을 지정된 포트 {port}에서 시작합니다...")
    else:
        # 사용 가능한 포트 자동 검색
        try:
            port = find_available_port()
            print(f"🚀 Flask 앱을 포트 {port}에서 시작합니다...")
        except RuntimeError as e:
            print(f"❌ 오류: {e}")
            print("💡 해결 방법:")
            print("   1. 다른 Flask 앱을 종료하세요")
            print("   2. 또는 수동으로 포트를 지정하세요: python flask_app.py --port 5001")
            print("   3. 더 넓은 범위의 포트를 시도해보세요: python flask_app.py --port 8000")
            sys.exit(1)
    
    print(f"🌐 브라우저에서 http://localhost:{port} 으로 접속하세요.")
    print(f"🔧 디버그 모드: {'활성화' if args.debug else '비활성화'}")
    
    # Flask 앱 실행
    if args.debug:
        # 디버그 모드: 파일 변경 감지, 상세 오류 정보, 자동 재시작
        print("🔧 디버그 모드 활성화: 파일 변경 시 자동 재시작됩니다.")
        socketio.run(app, debug=True, host=args.host, port=port, use_reloader=True, log_output=True)
    else:
        # 일반 모드: 안정적이고 빠름
        socketio.run(app, debug=False, host=args.host, port=port, use_reloader=False, log_output=False)
