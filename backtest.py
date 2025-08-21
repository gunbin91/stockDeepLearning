import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import joblib
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import pandas_ta as ta # pandas-ta 라이브러리 임포트 추가
from datetime import datetime
import concurrent.futures

# 내부 모듈 임포트
# weight_optimizer에서 함수를 가져오는 대신, 독립적인 함수로 재정의
from scoring import calculate_factor_scores
import ensemble

# --- 설정 변수 ---
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = datetime.now().strftime('%Y-%m-%d')
WEIGHTS_FILE = 'optimal_weights.json'
MODEL_FILE = 'stock_prediction_model_rf_upgraded.joblib'
TOP_N_STOCKS = 5
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522" # 본인의 키로 교체

# --- 데이터 준비 함수들 (train_model.py와 완전히 동일하게 수정) ---

def get_financial_data_for_training_http(corp_codes, start_year, end_year):
    # ... (train_model.py와 동일)
    all_fs_data = {}
    for year in range(start_year, end_year + 1):
        # ... (이하 로직은 이미 안정화되었으므로 그대로 둠)
        year_fs_data = []
        for i in tqdm(range(0, len(corp_codes), 100), desc=f"{year}년 재무 데이터 수집"):
            corp_code_chunk = corp_codes[i:i+100]
            corp_code_str = ','.join(corp_code_chunk)
            url = "https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
            params = { 'crtfc_key': DART_API_KEY, 'corp_code': corp_code_str, 'bsns_year': str(year), 'reprt_code': '11011' }
            try:
                res = requests.get(url, params=params)
                if res.status_code == 200 and res.json().get('status') == '000': year_fs_data.extend(res.json()['list'])
            except Exception: continue
            time.sleep(0.1)
        if year_fs_data:
            df = pd.DataFrame(year_fs_data)
            df['thstrm_amount'] = pd.to_numeric(df['thstrm_amount'].str.replace(',', ''), errors='coerce')
            df_pivot = df.pivot_table(index='stock_code', columns='account_nm', values='thstrm_amount')
            all_fs_data[year] = df_pivot.to_dict('index')
    return all_fs_data

def fetch_stock_list():
    # ... (train_model.py와 동일)
    try:
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        stock_list = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        stock_list = stock_list[~stock_list['Name'].str.contains('스팩|리츠', na=False)].copy()
        stock_list.rename(columns={'Code': '종목코드', 'Name': '종목명'}, inplace=True)
        return stock_list[['종목코드', '종목명']]
    except Exception: return pd.DataFrame()

# <<< 핵심 수정: train_model.py의 함수를 그대로 복사하여 피처 생성 로직 동기화 >>>
def process_single_ticker_data(stock_info, start_date, end_date, all_fs_data, df_marcap_long):
    ticker = stock_info['종목코드']
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 251 + 60: return None
        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        df = df_price[['종가', '거래량']].copy()
        df['연도'] = df.index.year
        df_marcap_ticker = df_marcap_long[df_marcap_long['Code'] == ticker].copy()
        if df_marcap_ticker.empty: return None
        df.sort_index(inplace=True)
        df_marcap_ticker.sort_values(by='Date', inplace=True)
        df = pd.merge_asof(left=df, right=df_marcap_ticker[['Date', 'Marcap']], left_index=True, right_on='Date', direction='backward')
        df.rename(columns={'Marcap': '시가총액'}, inplace=True)
        df['거래대금'] = df['종가'] * df['거래량']
        df.dropna(subset=['시가총액'], inplace=True)
        if df.empty: return None
        for year, fs_year_data in all_fs_data.items():
            if ticker in fs_year_data:
                fs_data = fs_year_data[ticker]
                df.loc[df['연도'] == year, '당기순이익'] = fs_data.get('당기순이익')
                df.loc[df['연도'] == year, '자본총계'] = fs_data.get('자본총계')
        if '당기순이익' not in df.columns or '자본총계' not in df.columns: return None
        df[['당기순이익', '자본총계']] = df[['당기순이익', '자본총계']].ffill()
        if df[['당기순이익', '자본총계']].isnull().values.any(): return None
        
        # --- 피처 생성 로직 (train_model.py와 100% 동일하게) ---
        df['PER'] = df['시가총액'] / df['당기순이익']
        df['PBR'] = df['시가총액'] / df['자본총계']
        df['ROE'] = df['당기순이익'] / df['자본총계']
        df['log_mktcap'] = np.log(df['시가총액'])
        df['수익률(1W)'] = df['종가'].pct_change(periods=5)
        df['수익률(2W)'] = df['종가'].pct_change(periods=10)
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df['거래대금_MA20'] = df['거래대금'].rolling(window=20).mean()
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        df['단기 정배열'] = (df['MA5'] > df['MA20']).astype(int)
        df['52주_최고가'] = df['종가'].rolling(window=250).max()
        df['52주_신고가_비율'] = df['종가'] / df['52주_최고가']
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        
        df['종목코드'] = ticker
        df.set_index('Date', inplace=True)
        return df
    except Exception: return None

def prepare_full_data(start_date, end_date):
    # ... (weight_optimizer.py와 동일하게 이미 안정화된 로직)
    print(f"백테스트 데이터 준비 중 ({start_date} ~ {end_date})...")
    stock_list = fetch_stock_list()
    # ... 이하 로직은 동일
    if stock_list.empty: raise ValueError("종목 리스트를 가져올 수 없습니다.")
    try:
        df_corp_map = pd.read_csv('corp_code_map.csv', dtype={'corp_code': str, '종목코드': str})
    except FileNotFoundError: raise FileNotFoundError("corp_code_map.csv 파일이 없습니다.")
    target_stocks = pd.merge(stock_list, df_corp_map, on='종목코드')
    corp_codes = target_stocks['corp_code'].unique().tolist()
    all_fs_data = get_financial_data_for_training_http(corp_codes, int(start_date[:4]) - 1, int(end_date[:4]))
    if not all_fs_data: raise ValueError("재무 데이터를 가져올 수 없습니다.")
    try:
        month_end_dates = pd.date_range(start=start_date, end=end_date, freq='ME').strftime('%Y%m%d').tolist()
        marcap_dfs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_date = {executor.submit(fdr.StockListing, 'KRX-MARCAP', date): date for date in month_end_dates}
            for future in tqdm(concurrent.futures.as_completed(future_to_date), total=len(month_end_dates), desc="시가총액 데이터 수집 (백테스트용)"):
                try:
                    date_str = future_to_date[future]
                    result_df = future.result()
                    if not result_df.empty:
                        result_df['Date'] = pd.to_datetime(date_str)
                        marcap_dfs.append(result_df)
                except Exception: continue
        if not marcap_dfs: raise Exception("수집된 시가총액 데이터가 없습니다.")
        df_marcap_long = pd.concat(marcap_dfs, ignore_index=True)
        df_marcap_long.sort_values(by=['Code', 'Date'], inplace=True)
    except Exception as e:
        raise ConnectionError(f"과거 시가총액 데이터 수집 실패: {e}")
    all_data = []
    stock_records = target_stocks.to_dict('records')
    for row in tqdm(stock_records, desc="피처 데이터 생성 (백테스트용)"):
        result_df = process_single_ticker_data(row, start_date, end_date, all_fs_data, df_marcap_long)
        if result_df is not None:
            all_data.append(result_df)
    if not all_data: raise ValueError("처리된 데이터가 없습니다.")
    final_df = pd.concat(all_data).reset_index()
    final_df.rename(columns={'Date': 'date'}, inplace=True)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['date', '종목코드'], inplace=True)
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df.sort_values(by=['date', '종목코드'], inplace=True)
    return final_df

# ... (이하 백테스트 실행 및 리포트 생성 함수는 수정 필요 없음)
def run_detailed_backtest(data, weights, initial_capital=1_000_000_000, top_n=5):
    """상세한 백테스트를 수행하고, 수익률, MDD 등 다양한 성과 지표를 포함한 결과를 반환합니다."""
    
    # 1. 최종 점수 계산
    # calculate_final_score 함수를 사용하여 일관성 유지
    final_df_list = []
    # 그룹화 전에 인덱스 리셋
    data_reset = data.reset_index()
    for date, daily_data in tqdm(data_reset.groupby('date'), desc="일별 최종 점수 계산"):
        # ensemble 모듈에 최적 가중치를 임시로 적용하는 방식 대신, 가중치를 직접 전달
        temp_df = ensemble.calculate_final_score(daily_data.copy())
        temp_df['date'] = date
        final_df_list.append(temp_df)
    
    final_scores_df = pd.concat(final_df_list).set_index(['date', '종목코드'])
    data = data.merge(final_scores_df[['final_score']], left_index=True, right_index=True, how='left')
    data.dropna(subset=['final_score'], inplace=True)

    # 2. 시뮬레이션 변수 초기화
    cash = initial_capital
    portfolio = {}
    portfolio_history = []
    trade_log = []
    daily_dates = data.index.get_level_values('date').unique().sort_values()

    # 3. 일별 시뮬레이션
    for date in tqdm(daily_dates, desc="상세 백테스팅 중"):
        # 3.1. 매도 로직
        for ticker in list(portfolio.keys()):
            if (date, ticker) in data.index:
                stock_info = portfolio[ticker]
                current_price = data.loc[(date, ticker), '종가']
                
                sell_condition = (
                    (current_price >= stock_info['buy_price'] * 1.05) or # 익절
                    (current_price <= stock_info['buy_price'] * 0.97) or # 손절
                    ((date - stock_info['buy_date']).days >= 15) # 기간만료
                )
                
                if sell_condition:
                    sell_value = current_price * stock_info['shares']
                    cash += sell_value
                    trade_log.append({'type': 'sell', 'date': date, 'ticker': ticker, 'price': current_price, 'return': (current_price / stock_info['buy_price']) - 1})
                    del portfolio[ticker]

        # 3.2. 매수 로직
        investment_per_stock = cash / top_n if top_n > 0 else 0
        
        # date에 해당하는 데이터가 있는지 확인
        if date in data.index.get_level_values('date'):
            daily_data = data.loc[date]
            buy_candidates = daily_data[~daily_data.index.get_level_values('종목코드').isin(portfolio.keys())].nlargest(top_n, 'final_score')

            for ticker, row in buy_candidates.iterrows():
                if cash >= investment_per_stock and investment_per_stock > 0:
                    buy_price = row['종가']
                    shares = investment_per_stock // buy_price
                    if shares > 0:
                        cash -= buy_price * shares
                        portfolio[ticker] = {'buy_date': date, 'buy_price': buy_price, 'shares': shares}
                        trade_log.append({'type': 'buy', 'date': date, 'ticker': ticker, 'price': buy_price})

        # 3.3. 일일 자산 기록
        current_portfolio_value = sum(data.loc[(date, ticker), '종가'] * info['shares'] for ticker, info in portfolio.items() if (date, ticker) in data.index)
        total_asset = cash + current_portfolio_value
        portfolio_history.append(total_asset)

    # 4. 성과 지표 계산
    portfolio_ts = pd.Series(portfolio_history, index=daily_dates)
    daily_returns = portfolio_ts.pct_change().fillna(0)
    
    total_return = (portfolio_ts.iloc[-1] / portfolio_ts.iloc[0]) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_ts)) - 1 if len(portfolio_ts) > 0 else 0
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    rolling_max = portfolio_ts.cummax()
    drawdown = (portfolio_ts - rolling_max) / rolling_max
    mdd = drawdown.min()

    return {
        "portfolio_history": portfolio_ts,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "mdd": mdd,
        "trade_log": pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    }

def create_html_report(results, output_path='backtest_report.html'):
    """백테스팅 결과를 사용하여 Plotly로 시각적인 HTML 리포트를 생성합니다."""
    
    if results["portfolio_history"].empty:
        print("백테스트 결과가 없어 리포트를 생성할 수 없습니다.")
        return

    # 1. 벤치마크(KOSPI) 데이터 가져오기
    kospi = fdr.DataReader('KS11', start=results["portfolio_history"].index.min(), end=results["portfolio_history"].index.max())
    
    # 2. 포트폴리오와 벤치마크 누적 수익률 계산
    portfolio_cumulative = (1 + results["portfolio_history"].pct_change().fillna(0)).cumprod()
    kospi_cumulative = (1 + kospi['Close'].pct_change().fillna(0)).cumprod()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3], specs=[[{"type": "scatter"}], [{"type": "table"}]])

    fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, name='포트폴리오', line=dict(color='royalblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=kospi_cumulative.index, y=kospi_cumulative, name='KOSPI', line=dict(color='grey', width=1, dash='dash')), row=1, col=1)

    metrics_df = pd.DataFrame({
        '지표': ['총수익률', '연환산 수익률', '최대 낙폭 (MDD)', '샤프 지수'],
        '값': [
            f"{results['total_return']:.2%}",
            f"{results['annual_return']:.2%}",
            f"{results['mdd']:.2%}",
            f"{results['sharpe_ratio']:.2f}"
        ]
    })
    fig.add_trace(go.Table(
        header=dict(values=list(metrics_df.columns), fill_color='paleturquoise', align='left'),
        cells=dict(values=[metrics_df.지표, metrics_df.값], fill_color='lavender', align='left')
    ), row=2, col=1)

    fig.update_layout(title_text='백테스팅 성과 분석 리포트', height=800, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title='누적 수익률')
    fig.update_xaxes(title_text='기간', row=1, col=1)
    fig.write_html(output_path)

def run_final_backtest():
    print("1. 최종 백테스트 시작...")
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"{WEIGHTS_FILE}을 찾을 수 없습니다. weight_optimizer.py를 먼저 실행해주세요.")
    with open(WEIGHTS_FILE, 'r') as f:
        optimal_weights = json.load(f)
    print(f"  - 최적 가중치를 {WEIGHTS_FILE}에서 불러왔습니다.")
    
    # --- 데이터 준비 기간 조정 ---
    # 모델 예측에 필요한 최소 과거 데이터(약 1년치)를 포함하여 데이터를 불러옴
    backtest_start_date_with_warmup = (pd.to_datetime(TEST_START_DATE) - pd.DateOffset(days=400)).strftime('%Y-%m-%d')
    test_data = prepare_full_data(backtest_start_date_with_warmup, TEST_END_DATE)

    print("  - 정식 모델 로딩 중...")
    try:
        model_data = joblib.load(MODEL_FILE)
        model = model_data['model']
        features = model_data['features']
        scaler = model_data['scaler']
    except FileNotFoundError:
        raise FileNotFoundError(f"{MODEL_FILE}을 찾을 수 없습니다. train_model.py를 먼저 실행해주세요.")
    
    # --- 예측 전 피처 결측치 처리 ---
    test_data_for_pred = test_data[features].copy()
    # 결측치를 0으로 채우는 것은 간단한 방법이며, 평균/중앙값 대체가 더 좋을 수 있음
    test_data_for_pred.fillna(0, inplace=True) 

    print("  - 테스트 데이터에 ML 예측 적용 중...")
    X_test_scaled = scaler.transform(test_data_for_pred)
    test_data['ml_pred_proba'] = model.predict_proba(X_test_scaled)[:, 1]
    
    # --- 실제 백테스트 기간 데이터만 필터링 ---
    test_data = test_data[test_data['date'] >= TEST_START_DATE]

    print("  - 팩터 점수 계산 중...")
    scored_data_list = []
    for date, daily_data in tqdm(test_data.groupby('date'), desc="일별 팩터 점수 계산"):
        daily_scored_data = calculate_factor_scores(daily_data.copy())
        daily_scored_data['date'] = date
        scored_data_list.append(daily_scored_data)

    if not scored_data_list:
        raise ValueError("점수 계산된 데이터가 없습니다.")
    all_scored_df = pd.concat(scored_data_list)
    score_cols_to_merge = ['종목코드', 'date'] + [col for col in all_scored_df.columns if '_score' in col]
    test_data = pd.merge(test_data, all_scored_df[score_cols_to_merge], on=['date', '종목코드'], how='left')
    
    test_data.drop_duplicates(subset=['date', '종목코드'], keep='first', inplace=True)
    test_data.set_index(['date', '종목코드'], inplace=True)
    test_data.sort_index(inplace=True)
    
    print("\n3. 최종 백테스팅 시뮬레이션 실행 중...")
    backtest_results = run_detailed_backtest(test_data, optimal_weights, top_n=TOP_N_STOCKS)
    
    print("\n4. HTML 리포트 생성 중...")
    create_html_report(backtest_results, output_path='backtest_report.html')
    print(f"\n✅ 백테스팅 완료. `backtest_report.html` 파일이 생성되었습니다.")

if __name__ == '__main__':
    run_final_backtest()
