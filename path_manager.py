"""
경로 관리 시스템
===============

이 파일은 프로젝트의 모든 파일 경로를 중앙에서 관리합니다.
개발 환경과 운영 환경에서 일관된 경로를 사용할 수 있도록 합니다.

주요 기능:
- 프로젝트 루트 경로 관리
- 데이터, 로그, 모델 파일 경로 관리
- 필요한 디렉토리 자동 생성
- 크로스 플랫폼 경로 처리
"""

import os
import shutil
from pathlib import Path
from typing import Optional

class PathManager:
    """
    프로젝트 경로 관리 클래스 (싱글톤 패턴)
    
    프로젝트의 모든 파일 경로를 중앙에서 관리합니다.
    싱글톤 패턴을 사용하여 어디서든 동일한 경로를 참조할 수 있습니다.
    """
    
    _instance = None
    _project_root = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._project_root is None:
            self._project_root = self._find_project_root()
    
    def _find_project_root(self) -> Path:
        """프로젝트 루트 디렉토리를 찾는 함수"""
        current_file = Path(__file__).resolve()
        
        # 현재 파일이 run 디렉토리에 있는지 확인
        if current_file.parent.name == 'run':
            return current_file.parent.parent
        else:
            return current_file.parent
    
    @property
    def project_root(self) -> Path:
        """프로젝트 루트 디렉토리"""
        return self._project_root
    
    @property
    def data_dir(self) -> Path:
        """데이터 디렉토리"""
        return self.project_root / 'data'
    
    
    @property
    def logs_dir(self) -> Path:
        """로그 디렉토리"""
        return self.project_root / 'logs'
    
    @property
    def templates_dir(self) -> Path:
        """템플릿 디렉토리"""
        return self.project_root / 'templates'
    
    @property
    def static_dir(self) -> Path:
        """정적 파일 디렉토리"""
        return self.project_root / 'static'
    
    @property
    def run_dir(self) -> Path:
        """실행 스크립트 디렉토리"""
        return self.project_root / 'run'
    
    
    def get_model_path(self) -> Path:
        """모델 파일 경로 (기존 모델)"""
        return self.data_dir / 'stock_prediction_model_rf_upgraded.joblib'
    
    def get_cuml_model_path(self) -> Path:
        """cuML 앙상블 모델 파일 경로"""
        return self.data_dir / 'cuml_ensemble_model.joblib'
    
    def get_weights_path(self) -> Path:
        """가중치 파일 경로"""
        return self.data_dir / 'optimal_weights.json'
    
    def get_backtest_report_path(self) -> Path:
        """백테스팅 리포트 파일 경로"""
        return self.project_root / 'backtest_report.html'
    
    
    def ensure_directories(self):
        """필요한 디렉토리들을 생성 (중복 생성 방지)"""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.templates_dir,
            self.static_dir,
            self.static_dir / 'css',
            self.static_dir / 'js'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_temp_dir(self, temp_name: str = 'temp') -> Path:
        """임시 디렉토리 경로 (프로젝트 루트 기준)"""
        temp_dir = self.project_root / f'.{temp_name}'
        temp_dir.mkdir(exist_ok=True)
        return temp_dir
    
    def cleanup_temp_dir(self, temp_name: str = 'temp'):
        """임시 디렉토리 정리"""
        import shutil
        temp_dir = self.project_root / f'.{temp_name}'
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

# 전역 인스턴스
path_manager = PathManager()

# 편의 함수들
def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 반환"""
    return path_manager.project_root

def get_data_dir() -> Path:
    """데이터 디렉토리 반환"""
    return path_manager.data_dir


def get_logs_dir() -> Path:
    """로그 디렉토리 반환"""
    return path_manager.logs_dir

def get_templates_dir() -> Path:
    """템플릿 디렉토리 반환"""
    return path_manager.templates_dir

def get_static_dir() -> Path:
    """정적 파일 디렉토리 반환"""
    return path_manager.static_dir

def ensure_all_directories():
    """모든 필요한 디렉토리 생성"""
    path_manager.ensure_directories()
