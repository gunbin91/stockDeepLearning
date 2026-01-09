# 🚀 AI 기반 주식 종목 분석 및 추천 시스템

## 📋 프로젝트 개요

본 프로젝트는 **KOSPI 및 KOSDAQ 상장 종목**을 대상으로 **기술적 지표, 재무 데이터, 거시 경제 지표**를 종합 분석하여 **투자 매력도가 높은 종목**을 발굴하는 **AI 기반 주식 추천 시스템**입니다.

### 🎯 핵심 목표
- **다양한 팩터**(가치, 퀄리티, 모멘텀) 기반 종목 점수 산출
- **Random Forest 머신러닝 모델**을 통한 **'15일 내 5% 이상 상승 확률'** 예측
- **백테스팅 기반 가중치 최적화**로 **샤프 지수 최대화**
- **Flask 웹 애플리케이션**을 통한 **시각적 분석 결과 제공**

---

## 🏗️ 프로젝트 구조

### 📁 핵심 파일 구조
```
stockDeepLearning/
├── 📱 웹 애플리케이션
│   ├── flask_app.py              # Flask 메인 애플리케이션
│   └── config.py                 # 설정 파일
│
├── 🔧 핵심 모듈
│   ├── data_fetcher.py           # 데이터 수집 및 피처 계산
│   ├── data_processor.py         # 실시간 데이터 처리 시스템
│   ├── scoring.py                # 팩터 점수 계산
│   ├── ml_model.py               # 머신러닝 모델 예측
│   ├── ensemble.py               # 앙상블 최종 점수 계산
│   ├── path_manager.py           # 통일된 경로 관리 시스템
│   ├── logger.py                 # 로깅 시스템
│   └── exceptions.py             # 예외 처리
│
├── 🚀 실행 스크립트
│   ├── scripts/
│   │   ├── train_gpu_main.py     # GPU 머신러닝 모델 학습
│   │   ├── weight_optimizer.py    # 가중치 최적화
│   │   ├── backtest.py            # 백테스팅 실행
│   │   └── run_analysis.py        # 분석 실행
│   └── run/                       # 실행 스크립트 (WSL 환경)
│
├── 💾 데이터 저장소
│   ├── data/
│   │   ├── stock_prediction_model_rf_upgraded.joblib  # ML 모델
│   │   ├── optimal_weights.json               # 최적 가중치
│   │   ├── analysis_result.json              # 분석 결과
│   │   ├── market_condition.json             # 시장 현황
│   │   └── cached_features.json              # 피처 데이터
│
└── 📊 결과물
    ├── backtest_report.html       # 백테스팅 보고서
    └── logs/                      # 로그 파일들
```

---

## 🔄 전체 데이터 플로우

### 1️⃣ **데이터 수집 단계**
```python
def fetch_all_data(stock_list, selected_analysis_date, use_cache=True):
    # 1. 종목 리스트 수집 (KOSPI, KOSDAQ)
    # 2. 재무 데이터 수집 (Point-in-Time)
    # 3. 거시경제 데이터 수집 (KOSPI, USD/KRW, VIX)
    # 4. 개별 종목 주가 데이터 수집 (하이브리드: FDR → KRX → NAVER)
```

### 2️⃣ **피처 계산 단계**
```python
def process_single_ticker_data(stock_info, start_date, end_date, df_marcap_long, pbar_lock):
    # 기술적 지표 계산
    # (정리) 수익률(1M/3M) 피처는 현재 파이프라인에서 사용하지 않음
    df['변동성(1M)'] = df['종가'].rolling(20).std() / df['종가'].rolling(20).mean()
    df.ta.atr(high='고가', low='저가', close='종가', length=14, append=True)
    df.ta.obv(close='종가', volume='거래량', append=True)
    df.ta.adx(high='고가', low='저가', close='종가', length=14, append=True)
    
    # 볼린저 밴드 계산
    bbands = df.ta.bbands(close='종가', length=20, std=2)
    df['BBW_20_2'] = (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0']) / bbands['BBM_20_2.0_2.0']
    df['BB_Position'] = (df['종가'] - bbands['BBL_20_2.0_2.0']) / (bbands['BBU_20_2.0_2.0'] - bbands['BBL_20_2.0_2.0'])
    
    # 이격도 계산
    for p in [120, 240]:
        ma = df['종가'].rolling(window=p).mean()
        df[f'disparity_{p}'] = (df['종가'] / ma) * 100
    
    # 52주 신고가 비율
    df['52주_최고가'] = df['종가'].rolling(250).max()
    df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
    
    # 타겟 변수 생성
    df['target'] = (df['종가'].shift(-15) / df['종가'] > 1.05).astype(int)
```

### 3️⃣ **팩터 점수 계산**
```python
# scoring.py
def calculate_factor_scores(df):
    # 변동성 점수 계산 (낮을수록 좋음)
    scored_df['volatility_score'] = df['변동성(1M)'].rank(
        method='min', ascending=True, pct=True, na_option='bottom'
    ) * 100
```

### 4️⃣ **머신러닝 예측**
```python
# ml_model.py
def predict_with_ml_model(df):
    model = joblib.load(MODEL_PATH)
    y_pred_proba = model.predict_proba(X_pred_scaled)[:, 1]
```

### 5️⃣ **앙상블 최종 점수 계산**
```python
# ensemble.py
def calculate_final_score(df):
    # 가중치 적용
    factor_weights = {
        'volatility_score': 0.10,
        'ml_pred_proba': 0.90,
    }
    
    # 정규화 및 가중합
    for factor, weight in normalized_weights.items():
        final_df['final_score'] += final_df[factor + '_norm'].fillna(50) * weight
    
    # 최종 순위 계산
    final_df['최종순위'] = final_df['final_score'].rank(ascending=False, method='first').astype(int)
```

---

## 🎯 핵심 기능 상세

### 📈 **데이터 수집 시스템**
- **하이브리드 데이터 소스**: FinanceDataReader → KRX → NAVER 순으로 폴백
- **Point-in-Time 재무 데이터**: 시점별 정확한 재무 지표 보장
- **거시경제 지표**: KOSPI, USD/KRW, VIX 등 시장 상황 반영
- **실시간 거래일 확인**: 삼성전자(005930) 기준 실제 거래일 확인

### 🧠 **머신러닝 모델**
- **모델**: RandomForestClassifier (RF), LightGBM (LGBM)
- **타겟**: 향후 10거래일 내 *최저가가 -5% 이하로 내려가지 않고* *최고가가 +8% 이상 한 번이라도 상승* 여부 (Binary Classification)
- **피처**: 30+ 기술적/재무적/거시경제적 지표
- **최적화**: 하이퍼파라미터 튜닝 (스크립트별 상이)

### ⚖️ **가중치 최적화 시스템**
```python
# scripts/weight_optimizer.py
def find_optimal_weights(top_n_stocks, data):
    # 그리드 서치로 모든 가중치 조합 테스트
    # 샤프 지수 최대화하는 최적 조합 탐색
```

### 📊 **백테스팅 시스템**
- **매수 규칙**: 상위 N개 종목 매수
- **매도 규칙**: 익절(+5%), 손절(-3%), 기간만료(15일)
- **성과 지표**: 총수익률, 연환산수익률, MDD, 샤프지수
- **시각화**: HTML 보고서 생성

---

## 🚀 설치 및 실행

### 1️⃣ **환경 설정**
```bash
# 가상환경 생성
python3.12 -m venv venv

# 가상환경 활성화
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2️⃣ **데이터베이스 구축 (불필요 - 실시간 수집으로 전환)**
```bash
# 캐시 시스템 제거로 인해 데이터베이스 구축 단계 불필요
# 모든 데이터는 실시간으로 수집됩니다.
```

### 3️⃣ **모델 학습**
```bash
# GPU 머신러닝 모델 학습 (WSL 환경)
run/run_train_gpu.bat      # Windows (WSL 실행)
# 또는 WSL에서 직접 실행
bash run/sh/train_gpu.sh --n_iter 100 --max_depth 10 20 30 40
```

### 4️⃣ **가중치 최적화**
```bash
# 최적 가중치 탐색 (시간 소요)
python scripts/weight_optimizer.py
# 또는
run/run_weight_optimizer.command  # macOS
run/run_weight_optimizer.bat      # Windows
```

### 5️⃣ **백테스팅 실행 (선택사항)**
```bash
# 최종 백테스팅
python scripts/backtest.py
# 또는
run/backtest.command  # macOS
run/backtest.bat      # Windows
```

### 6️⃣ **웹 애플리케이션 실행**

#### **Flask 버전 (권장)**
```bash
# Flask 앱 실행
python flask_app.py
# 또는
run/start_flask_app.command  # macOS
run/start_flask_app.bat      # Windows
```
- **URL**: http://localhost:5500
- **특징**: 현대적인 웹 UI, 실시간 WebSocket 통신, 반응형 디자인


---

## 💾 실시간 데이터 수집 시스템

### 🗂️ **데이터 파일 구조**
```
data/
├── stock_prediction_model_rf_upgraded.joblib  # ML 모델
├── optimal_weights.json               # 최적 가중치
├── analysis_result.json              # 분석 결과
├── market_condition.json             # 시장 현황
└── cached_features.json              # 피처 데이터
```

### ⚡ **효율성 최적화 전략**
- **월초 수집**: 시가총액/재무 데이터는 월초 거래일만 수집
- **일별 분배**: 월초 데이터를 해당 월의 모든 거래일에 분배
- **실시간 수집**: 가격/거래량 데이터는 실시간 수집
- **병렬 처리**: 16개 워커로 동시 데이터 수집
- **메모리 관리**: 자동 가비지 컬렉션으로 메모리 효율성
- **API 최적화**: 하이브리드 데이터 소스로 안정성 확보

---

## 📊 주요 피처 (Features)

### 🔢 **기술적 지표**
- **이동평균**: MA5, MA20, MA60, MA120, MA240
- **볼린저 밴드**: BBW_20_2, BB_Position
- **변동성**: 변동성(1W), 변동성(1M), 변동성(3M), ATRr_14
- **이격도**: disparity_120, disparity_240
- **추세 지표**: ADX_14, OBV
- **52주 신고가**: 52주_신고가_비율

### 💰 **재무 지표**
- **가치 지표**: PER, PBR, EPS, BPS
- **수익성**: ROE, 이익수익률
- **시가총액**: log_mktcap (로그 변환)

### 🌐 **거시경제 지표**
- **시장 지수**: KOSPI, KOSPI_pct_1d, KOSPI_pct_5d
- **환율**: USDKRW, USDKRW_pct_1d, USDKRW_pct_5d
- **변동성**: VIX, VIX_pct_1d, VIX_pct_5d

---

## 🎛️ 웹 애플리케이션 기능

### 📱 **Flask 버전 (권장) - 현대적 웹 UI**

#### **주요 페이지**
1. **주식 추천**: 분석 기준일 선택 및 종목 추천
   - 📅 날짜 선택기 (오늘까지 제한)
   - 🔄 실시간 분석 진행 상황 (WebSocket)
   - 📊 인터랙티브 종목 테이블 (DataTables)
   - 📈 네이버 스타일 캔들스틱 차트 (Plotly.js)
   - 🔍 종목별 상세 피처 데이터

2. **학습 모델 분석**: ML 모델 성능 및 피처 중요도
   - 📊 피처 중요도 차트 (Chart.js)
   - 📋 모델 파라미터 및 성능 지표
   - 📈 상세 데이터 테이블

3. **백테스팅 리포트**: 투자 전략 검증
   - ⚙️ 백테스팅 파라미터 설정
   - 📊 실시간 백테스팅 진행 상황
   - 📈 HTML 리포트 표시

#### **핵심 기능**
- **실시간 WebSocket 통신**: 분석/백테스팅 진행 상황 실시간 모니터링
- **반응형 디자인**: Bootstrap 5 기반 모바일 친화적 UI
- **인터랙티브 차트**: Plotly.js 기반 고성능 차트
- **등락율 색상 표시**: 상승(빨간색), 하락(파란색) 자동 색상 적용
- **실시간 로그**: 분석 과정의 모든 로그를 실시간으로 표시


#### **주요 페이지**
1. **홈**: 프로젝트 개요 및 시스템 상태
2. **주식 추천**: 분석 기준일 선택 및 종목 추천
3. **백테스팅 리포트**: 투자 시뮬레이션 결과
4. **설정**: 시스템 설정 및 관리

#### **분석 기능**
- **실시간 분석**: 선택한 날짜 기준 실시간 분석
- **캐시 활용**: 과거 날짜 분석 시 캐시 데이터 활용
- **상세 차트**: 주가 차트 + 기술적 지표 시각화
- **피처 데이터**: 종목별 상세 피처 값 표시

---

## ⚙️ 시스템 설정

### 🔧 **핵심 설정 파일**
- **`config.py`**: 전역 설정
- **`requirements.txt`**: Python 패키지 의존성
- **`data/optimal_weights.json`**: 최적화된 가중치
- **`data/stock_prediction_model_rf_upgraded.joblib`**: 학습된 ML 모델

### 📝 **로그 시스템**
- **구조화된 로깅**: JSON 형태 로그 저장
- **레벨별 로깅**: INFO, WARNING, ERROR, CRITICAL
- **컨텍스트 정보**: 함수명, 파라미터, 예외 정보 포함

---

## 🚨 주의사항

### ⚠️ **면책 조항**
- 본 프로젝트는 **학습 및 연구 목적**으로 개발
- 제공되는 정보는 **투자 자문이 아님**
- **실제 투자 결정에 따른 책임은 투자자 본인**에게 있음

### 🔒 **데이터 보안**
- **API 키 관리**: 환경변수 또는 설정 파일 사용
- **캐시 데이터**: 민감한 정보 제외하고 저장
- **로그 보안**: 개인정보 로깅 방지

---

## 🛠️ 개발자 가이드

### 📝 **코드 수정 시 주의사항**
1. **로깅 추가**: 모든 주요 함수에 적절한 로깅 추가
2. **예외 처리**: try-catch 블록으로 에러 처리
3. **메모리 관리**: 대용량 데이터 처리 시 gc.collect() 사용
4. **캐시 고려**: 성능을 위해 캐시 활용 고려

### 🔄 **데이터 업데이트**
- **실시간 수집**: `data_processor.py`의 실시간 데이터 수집 기능 활용
- **수동 업데이트**: 캐시 삭제 후 재실행
- **데이터 검증**: 수집된 데이터 품질 확인

### 🐛 **문제 해결**
1. **메모리 부족**: 배치 크기 조정
2. **API 제한**: 요청 간격 조정
3. **캐시 오류**: 캐시 디렉토리 삭제 후 재생성
4. **모델 오류**: 모델 재학습 필요

---

## 📈 성능 지표

### ⚡ **처리 속도**
- **종목 수집**: ~2,000개 종목/5-10분
- **피처 계산**: 병렬 처리로 3-5배 속도 향상
- **캐시 활용**: 90% 이상 속도 향상

### 💾 **메모리 사용량**
- **기본 사용량**: ~2GB
- **최대 사용량**: ~8GB (대용량 처리 시)
- **자동 정리**: gc.collect()로 메모리 관리

---

## 🔮 향후 개선 계획

### 🚀 **기능 개선**
- **포트폴리오 최적화**: 마코위츠 모델 기반 포트폴리오 구성
- **실시간 알림**: 추천 종목 실시간 알림 시스템

### 🛠️ **기술 개선**
- **마이크로서비스**: 모듈별 독립적 서비스화
- **클라우드 배포**: AWS, GCP 클라우드 배포
- **API 서버**: RESTful API 서버 구축
- **모니터링**: 시스템 상태 실시간 모니터링

---

## 📞 지원 및 문의

### 🆘 **문제 신고**
- **GitHub Issues**: 버그 리포트 및 기능 요청
- **이메일**: 프로젝트 관련 문의
- **문서**: 상세한 사용법은 각 모듈별 docstring 참조

### 📚 **추가 자료**
- **API 문서**: 각 함수별 상세 설명
- **예제 코드**: 사용 예제 및 샘플 코드
- **성능 벤치마크**: 시스템 성능 측정 결과

---

**🎯 이 README 파일 하나로 프로젝트의 모든 것을 이해하고 정확하게 작업할 수 있습니다!**