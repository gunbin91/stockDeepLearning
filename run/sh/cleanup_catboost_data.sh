#!/bin/bash

# 정리할 데이터 경로 설정 (3모델 공용 feather + 구 CatBoost 전용 잔여)
DATA_PATH=~/stock_data/processed_feather
OLD_DATA_PATH=~/stock_data/processed_feather_catboost
IMPUTATION_VALUES_PATH=~/stock_data/imputation_values_catboost.joblib
FOLD_CACHE_PATH=~/stock_data/fold_cache_catboost

echo "🧹 CatBoost/공용 전처리 데이터 정리를 시작합니다..."
echo "   - 공용 경로: $DATA_PATH"
echo "   - 구 CatBoost 경로: $OLD_DATA_PATH"
echo "   - imputation_values 파일: $IMPUTATION_VALUES_PATH"
echo "   - fold_cache 디렉토리: $FOLD_CACHE_PATH"

# 삭제할 항목이 있는지 확인
HAS_DATA=false
if [ -d "$DATA_PATH" ]; then
    HAS_DATA=true
fi
if [ -d "$OLD_DATA_PATH" ]; then
    HAS_DATA=true
fi
if [ -f "$IMPUTATION_VALUES_PATH" ]; then
    HAS_DATA=true
fi
if [ -d "$FOLD_CACHE_PATH" ]; then
    HAS_DATA=true
fi

if [ "$HAS_DATA" = true ]; then
    echo "   ⚠️ 공용 feather($DATA_PATH)도 삭제됩니다. (RF/LGBM와 공유)"
    echo "   'y'를 입력하면 데이터가 영구적으로 삭제됩니다."
    read -p "   정말로 삭제하시겠습니까? [y/n]: " choice
    echo ""
    
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        echo "   삭제를 진행합니다..."
        
        if [ -d "$DATA_PATH" ]; then
            rm -rf "$DATA_PATH"
            echo "   ✅ 공용 전처리 데이터 디렉토리가 삭제되었습니다."
        fi

        if [ -d "$OLD_DATA_PATH" ]; then
            rm -rf "$OLD_DATA_PATH"
            echo "   ✅ 구 CatBoost 전처리 데이터 디렉토리가 삭제되었습니다."
        fi
        
        if [ -f "$IMPUTATION_VALUES_PATH" ]; then
            rm -f "$IMPUTATION_VALUES_PATH"
            echo "   ✅ imputation_values 파일이 삭제되었습니다."
        fi
        
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
