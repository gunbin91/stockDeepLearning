#!/bin/bash

# 정리할 데이터 경로 설정
DATA_PATH=~/stock_data/processed_feather
IMPUTATION_VALUES_PATH=~/stock_data/imputation_values.joblib
FOLD_CACHE_PATH=~/stock_data/fold_cache

echo "🧹 전처리 데이터 정리를 시작합니다..."
echo "   - 대상 경로: $DATA_PATH"
echo "   - imputation_values 파일: $IMPUTATION_VALUES_PATH"
echo "   - fold_cache 디렉토리: $FOLD_CACHE_PATH"

# 삭제할 항목이 있는지 확인
HAS_DATA=false
if [ -d "$DATA_PATH" ]; then
    HAS_DATA=true
fi
if [ -f "$IMPUTATION_VALUES_PATH" ]; then
    HAS_DATA=true
fi
if [ -d "$FOLD_CACHE_PATH" ]; then
    HAS_DATA=true
fi

if [ "$HAS_DATA" = true ]; then
    echo "   'y'를 입력하면 데이터가 영구적으로 삭제됩니다."
    read -p "   정말로 삭제하시겠습니까? [y/n]: " choice
    echo ""
    
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "   삭제를 진행합니다..."
        
        # 전처리 데이터 디렉토리 삭제
        if [ -d "$DATA_PATH" ]; then
        rm -rf "$DATA_PATH"
            echo "   ✅ 전처리 데이터 디렉토리가 삭제되었습니다."
        fi
        
        # imputation_values 파일 삭제
        if [ -f "$IMPUTATION_VALUES_PATH" ]; then
            rm -f "$IMPUTATION_VALUES_PATH"
            echo "   ✅ imputation_values 파일이 삭제되었습니다."
        fi
        
        # fold_cache 디렉토리 삭제
        if [ -d "$FOLD_CACHE_PATH" ]; then
            rm -rf "$FOLD_CACHE_PATH"
            echo "   ✅ fold_cache 디렉토리가 삭제되었습니다."
        fi
        
        echo "   ✅ 모든 데이터가 성공적으로 삭제되었습니다."
    else
        echo "   삭제를 취소했습니다."
    fi
else
    echo "   ✅ 삭제할 데이터가 없습니다. (경로가 존재하지 않음)"
fi

echo "🧹 정리 작업이 완료되었습니다."
