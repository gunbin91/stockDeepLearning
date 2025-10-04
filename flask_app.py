# flask_app.py - Flask 기반 주식 분석 시스템

# Eventlet 몽키 패치 (최상단에 위치해야 함)
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
import threading
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
import scoring
import ml_model
import ensemble
from logger import log_info, log_warning, log_error, log_critical
from exceptions import DataFetchError, ModelPredictionError, AnalysisError

# Flask 앱 초기화
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock_analysis_secret_key_2024'
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   ping_timeout=60,      # 핑 타임아웃 60초 (기본 25초)
                   ping_interval=25      # 핑 간격 25초 (기본 25초)
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

# 설정
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'stock_prediction_model_rf_upgraded.joblib')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

# 전역 변수
analysis_processes = {}  # 진행 중인 분석 프로세스 추적
current_analysis_process = None  # 현재 분석 프로세스

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

# =============================================================================
# 유틸리티 함수들 (기존 app.py에서 가져옴)
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
    """캐시된 분석 결과 로드"""
    result_path = os.path.join(CACHE_DIR, 'analysis_result.json')
    market_path = os.path.join(CACHE_DIR, 'market_condition.json')
    
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
# 라우트 정의
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
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        features = model_data['features']
        
        # 피처 중요도 데이터 정리
        feature_importances = list(zip(features, model.feature_importances_))
        feature_importances.sort(key=lambda x: x[1], reverse=True)
        
        # 모델 정보
        model_info = {
            'model_type': type(model).__name__,
            'model_path': MODEL_PATH,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S'),
            'oob_score': getattr(model, 'oob_score_', None),
            'features': features,
            'feature_importances': feature_importances,
            'params': model.get_params()
        }
        
        return render_template('model_analysis.html', model_info=model_info)
    except FileNotFoundError:
        return render_template('model_analysis.html', error="모델 파일을 찾을 수 없습니다.")
    except Exception as e:
        return render_template('model_analysis.html', error=f"모델 로드 중 오류: {str(e)}")

@app.route('/backtest')
def backtest():
    """백테스팅 페이지"""
    # 기존 백테스팅 리포트가 있는지 확인
    report_path = os.path.join(os.path.dirname(__file__), 'backtest_report.html')
    has_report = os.path.exists(report_path)
    
    return render_template('backtest.html', has_report=has_report)

# =============================================================================
# API 엔드포인트들
# =============================================================================

@app.route('/api/start_analysis', methods=['POST'])
def start_analysis():
    """분석 시작 API"""
    try:
        data = request.get_json()
        analysis_date = data.get('analysis_date')
        
        if not analysis_date:
            return jsonify({'error': '분석 기준일이 필요합니다.'}), 400
        
        # 분석 중복 실행 방지
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
                TQDM_REGEX = re.compile(r'\s*\d{1,3}%|.*')
                PROGRESS_REGEX = re.compile(r'그룹 \d+/\d+ 처리 중 \(\d+/\d+ - \d+\.\d+%\)')
                last_line_was_tqdm = False
                
                for line in iter(process.stdout.readline, ''):
                    if TQDM_REGEX.search(line):
                        sys.stdout.write(line.strip() + '\r')
                        last_line_was_tqdm = True
                    else:
                        if last_line_was_tqdm:
                            sys.stdout.write('\n')
                        sys.stdout.write(line)
                        last_line_was_tqdm = False
                    sys.stdout.flush()
                    
                    # 진행률 메시지 감지 및 접두사 추가
                    if PROGRESS_REGEX.search(line):
                        # 진행률 메시지에 [PROGRESS] 접두사 추가
                        message = f"[PROGRESS] {line.strip()}"
                    else:
                        message = line.strip()
                    
                    # WebSocket으로 로그 전송
                    socketio.emit('analysis_log', {'message': message})
                    
                    # 실행 양보 (이벤트 루프가 블로킹되지 않도록)
                    socketio.sleep(0.01)
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    socketio.emit('analysis_complete', {'success': True})
                else:
                    socketio.emit('analysis_complete', {'success': False, 'error': '분석 실행 중 오류가 발생했습니다.'})
                    
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
        
        if current_analysis_process is None:
            return jsonify({'error': '실행 중인 분석이 없습니다.'}), 400
        
        # 프로세스 종료
        if current_analysis_process and current_analysis_process.poll() is None:
            current_analysis_process.terminate()
            try:
                current_analysis_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                current_analysis_process.kill()
                current_analysis_process.wait()
        
        # 플래그 해제
        set_analysis_running(False)
        current_analysis_process = None
        
        # WebSocket으로 중단 알림
        socketio.emit('analysis_complete', {'success': False, 'error': '사용자에 의해 분석이 중단되었습니다.'})
        
        return jsonify({'message': '분석이 중단되었습니다.'})
        
    except Exception as e:
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
        
        def run_backtest_process():
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
                
                # 실시간 로그 전송
                TQDM_REGEX = re.compile(r'\s*\d{1,3}%|.*')
                PROGRESS_REGEX = re.compile(r'그룹 \d+/\d+ 처리 중 \(\d+/\d+ - \d+\.\d+%\)')
                last_line_was_tqdm = False
                
                for line in iter(process.stdout.readline, ''):
                    if TQDM_REGEX.search(line):
                        sys.stdout.write(line.strip() + '\r')
                        last_line_was_tqdm = True
                    else:
                        if last_line_was_tqdm:
                            sys.stdout.write('\n')
                        sys.stdout.write(line)
                        last_line_was_tqdm = False
                    sys.stdout.flush()
                    
                    # 진행률 메시지 감지 및 접두사 추가
                    if PROGRESS_REGEX.search(line):
                        # 진행률 메시지에 [PROGRESS] 접두사 추가
                        message = f"[PROGRESS] {line.strip()}"
                    else:
                        message = line.strip()
                    
                    # WebSocket으로 로그 전송
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
        
        # 백그라운드에서 백테스팅 실행 (SocketIO 컨텍스트 유지)
        socketio.start_background_task(run_backtest_process)
        
        return jsonify({'message': '백테스팅이 시작되었습니다.'})
        
    except Exception as e:
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
        cached_features_path = os.path.join(CACHE_DIR, 'cached_features.json')
        if not os.path.exists(cached_features_path):
            return jsonify({'error': '피처 데이터를 찾을 수 없습니다.'}), 404
        
        cached_features_df = pd.read_json(cached_features_path, orient='records', dtype={'종목코드': str})
        selected_stock_features = cached_features_df[cached_features_df['종목코드'] == str(ticker_code).zfill(6)]
        
        if selected_stock_features.empty:
            return jsonify({'error': '해당 종목의 피처 데이터를 찾을 수 없습니다.'}), 404
        
        # 피처 데이터 정리
        display_features = selected_stock_features.drop(columns=['종목코드', 'date'], errors='ignore')
        
        # 객체를 문자열로 변환
        features_dict = {}
        for column in display_features.columns:
            value = display_features.iloc[0][column]
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

@app.route('/static/backtest_report.html')
def serve_backtest_report():
    """백테스팅 리포트 정적 파일 서빙"""
    report_path = os.path.join(os.path.dirname(__file__), 'backtest_report.html')
    if os.path.exists(report_path):
        return send_from_directory(os.path.dirname(report_path), 'backtest_report.html')
    else:
        return "백테스팅 리포트를 찾을 수 없습니다.", 404

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
        'timestamp': datetime.now().isoformat()
    })


# =============================================================================
# WebSocket 이벤트 핸들러
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
# 포트 찾기 함수
# =============================================================================

def find_available_port(start_port=5000, max_port=5100):
    """사용 가능한 포트를 찾는 함수"""
    import socket
    
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    # 모든 포트가 사용 중인 경우
    raise RuntimeError(f"포트 {start_port}-{max_port} 범위에서 사용 가능한 포트를 찾을 수 없습니다.")

# =============================================================================
# 메인 실행
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
    
    # 템플릿과 정적 파일 디렉토리 생성
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
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
