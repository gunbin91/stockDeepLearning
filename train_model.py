import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import dart_fss as dart
from datetime import datetime, timedelta
import joblib
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import concurrent.futures
from tqdm import tqdm
import warnings
from data_fetcher import fetch_stock_list

warnings.filterwarnings('ignore', category=FutureWarning)

# -----------------------------------------------------------------------------
# ⚠️ [필수] data_fetcher.py와 동일한 DART API 인증키를 입력하세요!
# -----------------------------------------------------------------------------
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522"
# -----------------------------------------------------------------------------

if DART_API_KEY != "여기에_발급받은_DART_인증키를_붙여넣으세요":
    dart.set_api_key(api_key=DART_API_KEY)
else:
    print("="*80); print(" [경고] train_model.py 파일에 DART API 키를 입력해야 합니다. "); print("="*80)

def get_financial_data_for_training(stock_list, start_year, end_year):
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요": return {}
    
    all_fs_data = {}
    corp_list = dart.get_corp_list()
    
    for year in range(start_year, end_year + 1):
        print(f"{year}년 재무 데이터 수집 중...")
        year_data = {}
        for index, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc=f"{year}년 재무 데이터 수집"):
            stock_code_fdr = row['종목코드']
            
            corp_obj = corp_list.find_by_stock_code(stock_code_fdr)
            
            if corp_obj is None:
                continue
            
            corp_code = corp_obj.corp_code
            
            try:
                fs = dart.fs.extract(corp_code=corp_code, bgn_de=f'{year}0101', fs_tp=('bs', 'is'))
                
                if fs is not None:
                    bs_df = fs['bs']
                    is_df = fs['is']
                    
                    combined_fs = pd.concat([bs_df, is_df], ignore_index=True)
                    
                    required_accounts = ['자본총계', '당기순이익']
                    filtered_fs = combined_fs[combined_fs['account_nm'].isin(required_accounts)]
                    
                    if not filtered_fs.empty:
                        fs_pivot = filtered_fs.pivot_table(index='stock_code', 
                                                         columns='account_nm', 
                                                         values='thstrm_amount').reset_index()
                        fs_pivot.rename(columns={'stock_code':'종목코드'}, inplace=True)
                        
                        for col in required_accounts:
                            if col in fs_pivot.columns:
                                fs_pivot[col] = pd.to_numeric(fs_pivot[col].str.replace(',', ''), errors='coerce')
                        
                        if not fs_pivot.empty:
                            year_data[stock_code_fdr] = fs_pivot.set_index('종목코드').iloc[0].to_dict()
            except Exception as e:
                print(f"Error fetching data for {stock_code_fdr} ({corp_code}): {e}")
                pass
        all_fs_data[year] = year_data
    print("✅ 전체 재무 데이터 수집 완료")
    return all_fs_data


def fetch_and_process_ticker_data(stock_info, start_date, end_date, all_fs_data):
    ticker = stock_info['종목코드']
    shares = stock_info['상장주식수']
    
    try:
        df_price = fdr.DataReader(ticker, start_date, end_date)
        if df_price.empty or len(df_price) < 120: return None
        df_price.rename(columns={'Close':'종가', 'Volume':'거래량'}, inplace=True)
        
        df = df_price[['종가', '거래량']].copy()
        df['연도'] = df.index.year
        
        for year, fs_year_data in all_fs_data.items():
            if ticker in fs_year_data:
                fs_data = fs_year_data[ticker]
                df.loc[df['연도'] == year, '당기순이익'] = fs_data.get('당기순이익')
                df.loc[df['연도'] == year, '자본총계'] = fs_data.get('자본총계')
        
        df[['당기순이익', '자본총계']] = df[['당기순이익'], '자본총계'].ffill().bfill()
        if df[['당기순이익', '자본총계']].isnull().values.any(): return None

        df['시가총액'] = df['종가'] * shares
        df['PER'] = df['시가총액'] / df['당기순이익']
        df['PBR'] = df['시가총액'] / df['자본총계']
        df['ROE'] = df['당기순이익'] / df['자본총계']
        
        df['수익률(1M)'] = df['종가'].pct_change(periods=20)
        df['수익률(3M)'] = df['종가'].pct_change(periods=60)
        df['변동성(1M)'] = df['종가'].rolling(window=20).std() / df['종가'].rolling(window=20).mean()
        df.ta.rsi(close='종가', length=14, append=True)
        df.ta.macd(close='종가', fast=12, slow=26, signal=9, append=True)
        
        df['target'] = (df['종가'].shift(-20) / df['종가'] > 1.05).astype(int)
        df['종목코드'] = ticker
        return df
    except Exception:
        return None

def create_training_data(stock_list, period_days=365*3):
    print("안정적인 API(FDR, DART) 기반으로 학습 데이터 생성을 시작합니다...")
    today = datetime.now()
    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=period_days)).strftime('%Y-%m-%d')
    
    all_fs_data = get_financial_data_for_training(stock_list, today.year - 4, today.year - 1)
    if not all_fs_data: return None, None, None

    all_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_and_process_ticker_data, row, start_date, end_date, all_fs_data): row
            for row in stock_list.to_dict('records')
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(stock_list), desc="학습 데이터 생성"):
            result_df = future.result()
            if result_df is not None:
                all_data.append(result_df)

    if not all_data: return None, None, None

    final_df = pd.concat(all_data)
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    features = [
        '수익률(1M)', '수익률(3M)', '변동성(1M)', 'PER', 'PBR', 'ROE', 'RSI_14',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    ]
    target = 'target'
    
    final_df.dropna(subset=[target] + features, inplace=True)
    
    X = final_df[features].astype(np.float32)
    y = final_df[target]
    
    print("학습 데이터 생성 완료!")
    return X, y, features

def train_evaluate_and_save_model(X, y, features, model_path='stock_prediction_model_rf_upgraded.joblib'):
    if X is None or y is None or X.empty or y.empty:
        print("학습 데이터가 없어 모델링을 건너뜁니다."); return

    print("모델 학습 및 평가를 시작합니다...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\nSMOTE 적용 전 학습 데이터 타겟 분포:\n", y_train.value_counts(normalize=True))
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("\nSMOTE 적용 후 학습 데이터 타겟 분포:\n", y_train_res.value_counts(normalize=True))

    print("\nRandomForest 모델 학습...")
    model = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=10, 
                                   min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    
    print("\n테스트 데이터로 모델 평가...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- 최종 모델 평가 결과 ---")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n분류 보고서 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['하락(0)', '상승(1)']))
    
    joblib.dump({'model': model, 'features': features}, model_path)
    print(f"\n✅ 새로운 데이터로 학습된 모델을 '{model_path}' 경로에 저장했습니다.")

if __name__ == '__main__':
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요": exit()

    stock_list = fetch_stock_list()
    if not stock_list.empty:
        X, y, features = create_training_data(stock_list)
        if X is not None:
            train_evaluate_and_save_model(X, y, features)
