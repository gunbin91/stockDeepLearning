# smart_cache.py - 스마트 캐싱 시스템

import pandas as pd
import numpy as np
import os
import json
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import shutil

class SmartCache:
    """스마트 캐싱 시스템 - 파일시스템 기반"""
    
    def __init__(self, cache_dir: str = None, max_cache_size_gb: float = 5.0):
        if cache_dir is None:
            # 통일된 경로 사용
            try:
                from path_manager import get_cache_dir
                cache_dir = str(get_cache_dir())
            except ImportError:
                # path_manager가 없는 경우 기본 경로 사용
                import os
                project_root = os.path.dirname(os.path.abspath(__file__))
                cache_dir = os.path.join(project_root, "cache")
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_gb = max_cache_size_gb
        self.logger = logging.getLogger(__name__)
        
        # 캐시 구조 설정
        self.metadata_dir = self.cache_dir / "metadata"
        self.data_dir = self.cache_dir / "data"
        self.temp_dir = self.cache_dir / "temp"
        
        # 디렉토리 생성
        self._setup_directories()
        
        # 메타데이터 파일 경로
        self.metadata_file = self.metadata_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _setup_directories(self):
        """캐시 디렉토리 구조 설정"""
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """캐시 메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"메타데이터 로드 실패: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """캐시 메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"메타데이터 저장 실패: {e}")
    
    def _generate_cache_key(self, data_type: str, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        params_str = json.dumps(params, sort_keys=True)
        return f"{data_type}_{hashlib.md5(params_str.encode()).hexdigest()}"
    
    def _get_cache_path(self, cache_key: str, file_type: str = "parquet") -> Path:
        """캐시 파일 경로 생성"""
        # 계층적 디렉토리 구조 (첫 2자리로 분할)
        subdir = cache_key[:2]
        subdir_path = self.data_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        
        return subdir_path / f"{cache_key}.{file_type}"
    
    def _is_expired(self, created_at: float, ttl_seconds: int) -> bool:
        """캐시 만료 확인"""
        return time.time() - created_at > ttl_seconds
    
    def _get_cache_size(self) -> float:
        """캐시 디렉토리 크기 계산 (GB)"""
        total_size = 0
        for file_path in self.data_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 ** 3)  # GB로 변환
    
    def _cleanup_old_cache(self):
        """오래된 캐시 정리"""
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in self.metadata.items():
            if self._is_expired(metadata['created_at'], metadata['ttl_seconds']):
                expired_keys.append(key)
        
        # 만료된 캐시 삭제
        for key in expired_keys:
            self._delete_cache(key)
            self.logger.info(f"만료된 캐시 삭제: {key}")
    
    def _delete_cache(self, cache_key: str):
        """캐시 삭제"""
        # 파일 삭제
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
        
        # 메타데이터 삭제
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()
    
    def _evict_lru_cache(self):
        """LRU 방식으로 캐시 제거"""
        if not self.metadata:
            return
        
        # 접근 시간 기준으로 정렬
        sorted_items = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('last_accessed', 0)
        )
        
        # 가장 오래된 25% 제거
        remove_count = max(1, len(sorted_items) // 4)
        for key, _ in sorted_items[:remove_count]:
            self._delete_cache(key)
            self.logger.info(f"LRU 캐시 제거: {key}")
    
    def get(self, data_type: str, params: Dict[str, Any], 
            ttl_seconds: int = 3600) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 조회"""
        cache_key = self._generate_cache_key(data_type, params)
        
        # 메타데이터 확인
        if cache_key not in self.metadata:
            return None
        
        metadata = self.metadata[cache_key]
        
        # 만료 확인
        if self._is_expired(metadata['created_at'], metadata['ttl_seconds']):
            self._delete_cache(cache_key)
            return None
        
        # 파일 존재 확인
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            self._delete_cache(cache_key)
            return None
        
        try:
            # 데이터 로드
            if cache_path.suffix == '.parquet':
                data = pd.read_parquet(cache_path)
            elif cache_path.suffix == '.pkl':
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                return None
            
            # 접근 시간 업데이트
            self.metadata[cache_key]['last_accessed'] = time.time()
            self._save_metadata()
            
            self.logger.info(f"캐시 히트: {cache_key}")
            return data
            
        except Exception as e:
            self.logger.error(f"캐시 로드 실패: {e}")
            self._delete_cache(cache_key)
            return None
    
    def set(self, data_type: str, params: Dict[str, Any], data, 
            ttl_seconds: int = 3600, file_type: str = "parquet"):
        """캐시에 데이터 저장"""
        cache_key = self._generate_cache_key(data_type, params)
        cache_path = self._get_cache_path(cache_key, file_type)
        
        try:
            # 데이터 저장
            if file_type == "parquet":
                if isinstance(data, dict):
                    # dict 객체는 pickle로 저장
                    with open(cache_path, 'wb') as f:
                        pickle.dump(data, f)
                else:
                    data.to_parquet(cache_path, index=False)
            elif file_type == "pkl":
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"지원하지 않는 파일 타입: {file_type}")
            
            # 메타데이터 저장
            if isinstance(data, dict):
                rows = len(data)
                columns = list(data.keys())
            else:
                rows = len(data)
                columns = list(data.columns)
                
            self.metadata[cache_key] = {
                'data_type': data_type,
                'params': params,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'ttl_seconds': ttl_seconds,
                'file_type': file_type,
                'size_bytes': cache_path.stat().st_size,
                'rows': rows,
                'columns': columns
            }
            self._save_metadata()
            
            if isinstance(data, dict):
                self.logger.info(f"캐시 저장: {cache_key} (dict, {len(data)}개 항목)")
            else:
                self.logger.info(f"캐시 저장: {cache_key} ({len(data)}행)")
            
            # 캐시 크기 확인 및 정리
            if self._get_cache_size() > self.max_cache_size_gb:
                self._cleanup_old_cache()
                if self._get_cache_size() > self.max_cache_size_gb:
                    self._evict_lru_cache()
                    
        except Exception as e:
            self.logger.error(f"캐시 저장 실패: {e}")
            if cache_path.exists():
                cache_path.unlink()
    
    def get_partial_data(self, data_type: str, params: Dict[str, Any], 
                        date_range: Tuple[str, str], 
                        ttl_seconds: int = 3600) -> Optional[pd.DataFrame]:
        """부분 데이터 조회 (날짜 범위)"""
        cache_key = self._generate_cache_key(data_type, params)
        
        # 전체 데이터 조회
        full_data = self.get(data_type, params, ttl_seconds)
        if full_data is None:
            return None
        
        # 날짜 범위 필터링
        if 'date' in full_data.columns:
            start_date, end_date = date_range
            mask = (full_data['date'] >= start_date) & (full_data['date'] <= end_date)
            return full_data[mask].copy()
        
        return full_data
    
    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 정보 조회"""
        total_size = self._get_cache_size()
        total_files = len(self.metadata)
        
        # 데이터 타입별 통계
        type_stats = {}
        for key, metadata in self.metadata.items():
            data_type = metadata['data_type']
            if data_type not in type_stats:
                type_stats[data_type] = {'count': 0, 'size_bytes': 0}
            type_stats[data_type]['count'] += 1
            type_stats[data_type]['size_bytes'] += metadata['size_bytes']
        
        return {
            'total_size_gb': total_size,
            'total_files': total_files,
            'type_stats': type_stats,
            'cache_dir': str(self.cache_dir)
        }
    
    def clear_cache(self, data_type: Optional[str] = None):
        """캐시 전체 또는 특정 타입 삭제"""
        if data_type is None:
            # 전체 캐시 삭제
            shutil.rmtree(self.data_dir)
            self.data_dir.mkdir(exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            self.logger.info("전체 캐시 삭제 완료")
        else:
            # 특정 타입만 삭제
            keys_to_delete = [key for key, metadata in self.metadata.items() 
                            if metadata['data_type'] == data_type]
            for key in keys_to_delete:
                self._delete_cache(key)
            self.logger.info(f"{data_type} 타입 캐시 삭제 완료")

# 캐싱 데코레이터
def cached(data_type: str, ttl_seconds: int = 3600, file_type: str = "parquet"):
    """캐싱 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = SmartCache()
            
            # 캐시 키 생성
            cache_params = {
                'args': str(args),
                'kwargs': kwargs,
                'func_name': func.__name__
            }
            
            # 캐시에서 조회
            cached_result = cache.get(data_type, cache_params, ttl_seconds)
            if cached_result is not None:
                return cached_result
            
            # 캐시 미스 시 함수 실행
            result = func(*args, **kwargs)
            
            # 결과 캐싱
            if isinstance(result, pd.DataFrame):
                cache.set(data_type, cache_params, result, ttl_seconds, file_type)
            
            return result
        return wrapper
    return decorator

# 전역 캐시 인스턴스
_global_cache = None

def get_cache() -> SmartCache:
    """전역 캐시 인스턴스 반환"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache()
    return _global_cache
