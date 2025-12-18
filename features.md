# 실제 학습에 사용되는 피처(Features) 설명 (gpuStock과 통일)

이 문서는 **현재 루트 프로젝트(CPU)**가 사용하는 학습 피처를 `gpuStock` 프로젝트의 **실제 학습 스크립트**(`gpuStock/scripts/train_gpu_main.py`)와 **동일하게** 유지하기 위한 문서입니다.

- **정답 소스(ground truth)**: `gpuStock/scripts/train_gpu_main.py`의 `features` 하드코딩 리스트
- **참고 문서**: `gpuStock/FEATURES.md` (해당 문서는 “총 피처 수” 표기가 버전별로 흔들려 있으므로, 학습 스크립트를 우선합니다.)

---

## 1) 학습 피처 리스트 (최신 gpuStock 기준)

| 카테고리 | 피처 |
|---|---|
| **기본/규모** | `log_mktcap`, `52주_신고가_비율` |
| **추세** | `ADX_14`, `MA20_Slope`, `MA120_Slope`, `MA240_Slope`, `KOSPI_MA20_Slope` |
| **이격/시장 이격** | `disparity_120`, `disparity_240`, `disparity_20`, `KOSPI_disparity_20` |
| **거래량/수급** | `RVOL`, `RVOL(1W)`, `시총 회전율(1W)`, `시총 회전율(3M)` |
| **모멘텀** | `RSI_Signal_Oscillator` |
| **ATR 변동성 비율** | `ATRr_5`, `ATRr_20` |
| **복합 점수** | `Trend_Pullback_Score` |
| **위치** | `Position_Range_60` |
| **리스크/변동성** | `HV_Volatility_5`, `HV_Volatility_20`, `HV_Volatility_60`, `Max_Drawdown_20` |
| **가격/체결** | `Log_Return_20`, `VWAP_Disparity_5` |

---

## 2) 참고 사항

- **계산되지만 학습 피처에 포함되지 않는 보조 컬럼이 있을 수 있습니다.** (예: RSI_14, OBV 등) 이는 `gpuStock/FEATURES.md`에도 동일한 방식으로 언급됩니다.
- 최종적으로 학습에 들어가는 피처는 **항상 `scripts/train_model.py`에서 gpuStock 학습 스크립트로부터 자동 추출된 리스트**가 결정합니다.

