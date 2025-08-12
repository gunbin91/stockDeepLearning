<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:8f33bccee539da8fdf658e1fcbd1affff80ddc5b3eeb51c75fdd1ffe2f7d6b1c
size 3773
=======
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# GPU 사용 비활성화 (M1/M2 Mac의 경우 Metal 사용 관련 충돌 방지)
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_dl_training_data(tickers, start_date, end_date, look_back=60, look_forward=20):
    """LSTM 모델 학습을 위한 시계열 데이터를 생성합니다."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_X, all_y = [], []

    for i, ticker in enumerate(tickers):
        name = stock.get_market_ticker_name(ticker)
        print(f"({i+1}/{len(tickers)}) {name}({ticker}) 데이터 처리 중...")
        time.sleep(0.1)
        try:
            df_price = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
            if len(df_price) < look_back + look_forward:
                print(f"  - 데이터 부족으로 건너뜁니다.")
                continue

            # 종가 데이터만 사용하고 정규화
            data_scaled = scaler.fit_transform(df_price[['종가']])

            # 시퀀스 데이터 생성
            X, y = [], []
            for j in range(len(data_scaled) - look_back - look_forward + 1):
                X.append(data_scaled[j:(j + look_back), 0])
                # 20일 후의 종가가 현재보다 높으면 1, 아니면 0
                target_price = data_scaled[j + look_back + look_forward - 1, 0]
                current_price = data_scaled[j + look_back - 1, 0]
                y.append(1 if target_price > current_price else 0) 
            
            all_X.extend(X)
            all_y.extend(y)
        except Exception as e:
            print(f"  - 에러 발생: {e}")

    return np.array(all_X), np.array(all_y)

def train_and_save_dl_model(model_path='time_series_model.h5'):
    """LSTM 모델을 학습하고 저장합니다."""
    print("딥러닝 모델 학습을 시작합니다...")
    
    # 1. 데이터 준비
    today_str = stock.get_nearest_business_day_in_a_week()
    today = datetime.strptime(today_str, '%Y%m%d')
    start_date = (today - timedelta(days=365*3)).strftime('%Y%m%d') # 3년치 데이터 사용
    
    try:
        tickers = stock.get_market_cap_by_ticker(today_str).head(50).index.tolist() # 시총 50개로 학습
    except Exception as e:
        print(f"시가총액 데이터 조회 실패, 백업 종목 사용: {e}")
        tickers = ['005930', '000660', '005380', '035420', '035720']

    X_train, y_train = create_dl_training_data(tickers, start_date, today_str)

    if X_train.shape[0] == 0:
        print("학습 데이터가 없어 모델 학습을 건너뜁니다.")
        return

    # LSTM 입력을 위해 차원 변경 (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # 2. 모델 정의
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1, activation='sigmoid') # 긍정(1)/부정(0) 예측이므로 sigmoid
    ])

    # 3. 모델 컴파일 및 학습
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("\n모델 학습 중...")
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # 4. 모델 저장
    model.save(model_path)
    print(f"딥러닝 모델을 '{model_path}'에 저장했습니다.")

if __name__ == '__main__':
    train_and_save_dl_model()
>>>>>>> origin/window
