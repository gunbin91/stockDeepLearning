<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:a5fea22368d84c37c2730d2b2936df5a84c7c389fcdb1161fc2006e1813a7972
size 4476
=======
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from transformers import pipeline
import time

# Hugging Face 모델 로드를 위한 파이프라인 초기화
# 모델은 처음 호출될 때 다운로드 및 로드되며, 이후에는 캐시된 모델을 사용합니다.
SENTIMENT_ANALYZER = None

def initialize_analyzer():
    """감성 분석기 파이프라인을 초기화합니다."""
    global SENTIMENT_ANALYZER
    if SENTIMENT_ANALYZER is None:
        print("Hugging Face 감성 분석 모델을 로드합니다... (최초 실행 시 시간이 소요될 수 있습니다)")
        try:
            SENTIMENT_ANALYZER = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
            print("모델 로드 완료.")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            # 모델 로드 실패 시, 더미 분석기를 설정하여 프로그램 중단을 방지
            SENTIMENT_ANALYZER = "dummy"

def analyze_news_sentiment(df):
    """
    각 종목의 최신 뉴스에 대한 감성 분석을 수행하고 점수를 매깁니다.
    """
    if df.empty:
        print("경고: analyze_news_sentiment에 빈 데이터프레임이 전달되었습니다. 빈 데이터프레임을 반환합니다.")
        return df.copy()

    # 모델이 초기화되었는지 확인 및 초기화
    initialize_analyzer()
    print(f"SENTIMENT_ANALYZER 상태: {SENTIMENT_ANALYZER}")

    analyzed_df = df.copy()
    sentiment_scores = []
    news_headlines_list = []

    # 데이터 조회를 위한 날짜 설정 (최근 3일)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    today_str = end_date.strftime('%Y%m%d')
    start_date_str = start_date.strftime('%Y%m%d')

    for i, row in analyzed_df.iterrows():
        ticker = row['종목코드']
        print(f"({i+1}/{len(analyzed_df)}) {row['종목명']}({ticker}) 뉴스 감성 분석 중...")
        time.sleep(0.1)
        
        try:
            df_news = stock.get_market_news_by_ticker(start_date_str, today_str, ticker)
            print(f"  - 뉴스 수집 결과 (empty): {df_news.empty}")
            if not df_news.empty:
                headlines = df_news['제목'].head(5).tolist() # 최신 5개 뉴스
                news_headlines_list.append(" | ".join(headlines))
                print(f"  - 분석할 헤드라인: {headlines}")
                
                # 모델 로드에 실패했거나, 분석기가 더미일 경우
                if SENTIMENT_ANALYZER == "dummy":
                    sentiment_scores.append(50.0) # 중립 점수
                    print("  - 더미 분석기 사용: 50.0점")
                    continue

                # 감성 분석 수행
                results = SENTIMENT_ANALYZER(headlines)
                print(f"  - 감성 분석 결과: {results}")
                
                # 점수 변환 및 평균 계산
                scores = []
                for res in results:
                    label = res['label']
                    score = res['score']
                    if label == 'neutral':
                        scores.append(50)
                    elif label == 'positive':
                        # 긍정일수록 100점에 가깝게
                        scores.append(50 + (score * 50))
                    elif label == 'negative':
                        # 부정일수록 0점에 가깝게
                        scores.append(50 - (score * 50))
                
                avg_score = np.mean(scores) if scores else 50.0
                print(f"  - 평균 감성 점수: {avg_score}")
                sentiment_scores.append(avg_score)

            else:
                news_headlines_list.append("최신 뉴스 없음")
                sentiment_scores.append(50.0) # 뉴스가 없으면 중립
                print("  - 뉴스 없음: 50.0점")
        except Exception as e:
            print(f"  - {row['종목명']} 뉴스 분석 중 에러: {e}")
            news_headlines_list.append("뉴스 조회 실패")
            sentiment_scores.append(50.0) # 에러 발생 시 중립
            print("  - 뉴스 분석 에러: 50.0점")

    analyzed_df['sentiment_score'] = pd.Series(sentiment_scores, index=analyzed_df.index).round(2)
    analyzed_df['최신뉴스'] = pd.Series(news_headlines_list, index=analyzed_df.index)
    
    return analyzed_df
>>>>>>> origin/window
