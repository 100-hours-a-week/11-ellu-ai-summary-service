"""
AI Processing API Application Package

이 패키지는 위키 요약과 회의록 처리를 위한 FastAPI 애플리케이션을 포함합니다.

주요 모듈:
- main: FastAPI 앱 초기화 및 엔드포인트 정의
- config: 애플리케이션 설정 관리
- dependencies: 의존성 관리 (ChromaDB, 데이터베이스 등)
- middleware: CORS, 로깅 등 미들웨어 설정
- exceptions: 사용자 정의 예외 처리
"""

from .main import app

__version__ = "1.0.0"
__all__ = ["app"]