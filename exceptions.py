"""
예외 처리 모듈
=============

이 파일은 주식 분석 시스템에서 발생할 수 있는 다양한 오류 상황을
체계적으로 처리하기 위한 커스텀 예외 클래스들을 정의합니다.

주요 예외 클래스:
- StockAnalysisError: 기본 예외 클래스
- DataFetchError: 데이터 수집 오류
- DataValidationError: 데이터 검증 오류
- ModelPredictionError: 모델 예측 오류
- AnalysisError: 분석 프로세스 오류
"""

class StockAnalysisError(Exception):
    """
    주식 분석 시스템의 기본 예외 클래스
    
    모든 주식 분석 관련 오류의 기본 클래스입니다.
    오류 코드와 상세 정보를 포함하여 디버깅을 용이하게 합니다.
    """
    
    def __init__(self, message, error_code=None, details=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class DataFetchError(StockAnalysisError):
    """데이터 수집 실패 예외"""
    
    def __init__(self, message, source=None, **kwargs):
        super().__init__(message, error_code="DATA_FETCH_ERROR", **kwargs)
        self.source = source

class ModelPredictionError(StockAnalysisError):
    """모델 예측 실패 예외"""
    
    def __init__(self, message, model_name=None, **kwargs):
        super().__init__(message, error_code="MODEL_PREDICTION_ERROR", **kwargs)
        self.model_name = model_name

class DataValidationError(StockAnalysisError):
    """데이터 검증 실패 예외"""
    
    def __init__(self, message, field_name=None, **kwargs):
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)
        self.field_name = field_name

class ConfigurationError(StockAnalysisError):
    """설정 관련 예외"""
    
    def __init__(self, message, config_key=None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_key = config_key

class CacheError(StockAnalysisError):
    """캐시 관련 예외"""
    
    def __init__(self, message, cache_key=None, **kwargs):
        super().__init__(message, error_code="CACHE_ERROR", **kwargs)
        self.cache_key = cache_key

class AnalysisError(StockAnalysisError):
    """분석 프로세스 예외"""
    
    def __init__(self, message, step=None, **kwargs):
        super().__init__(message, error_code="ANALYSIS_ERROR", **kwargs)
        self.step = step
