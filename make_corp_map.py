# make_corp_map.py
import dart_fss as dart
import pandas as pd

# -----------------------------------------------------------------------------
# ⚠️ [필수] 여기에 발급받은 DART API 인증키를 딱 한 번만 입력하세요!
# -----------------------------------------------------------------------------
DART_API_KEY = "03ac38be54eb9bb095c2304b254c756ebe73c522"
# -----------------------------------------------------------------------------

def create_and_save_corp_code_map():
    """
    dart-fss 라이브러리를 이용해 DART 고유번호와 종목코드 매핑 파일을 생성합니다.
    이 스크립트는 프로젝트 설정 시 단 한 번만 실행하면 됩니다.
    """
    if DART_API_KEY == "여기에_발급받은_DART_인증키를_붙여넣으세요":
        print("DART API 키를 입력하고 스크립트를 다시 실행해주세요.")
        return
        
    print("DART API 키를 설정합니다...")
    dart.set_api_key(api_key=DART_API_KEY)
    
    print("전체 기업 고유번호 목록을 가져옵니다...")
    corp_list = dart.get_corp_list()
    
    # CorpList 객체를 순회하며 데이터를 추출하여 리스트에 저장
    all_corps_data = []
    for corp in corp_list:
        # stock_code가 있는(상장된) 기업만 필터링
        if corp.stock_code:
            all_corps_data.append({
                'corp_code': corp.corp_code,
                'stock_code': corp.stock_code
            })

    # 리스트를 DataFrame으로 변환
    df_corp_map = pd.DataFrame(all_corps_data)
    
    # 혹시 모를 비상장 기업 제외
    df_corp_map = df_corp_map[~df_corp_map['stock_code'].isna()]
    
    # 컬럼명 변경
    df_corp_map.rename(columns={'stock_code':'종목코드'}, inplace=True)
    
    output_path = 'corp_code_map.csv'
    df_corp_map.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"'{output_path}' 파일이 성공적으로 생성되었습니다.")
    print("이제 이 스크립트는 더 이상 실행할 필요가 없습니다.")

if __name__ == "__main__":
    create_and_save_corp_code_map()