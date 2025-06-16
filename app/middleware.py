from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .config import (
    CORS_ORIGINS,
    CORS_CREDENTIALS, 
    CORS_METHODS,
    CORS_HEADERS,
    LOG_LEVEL,
    LOG_FORMAT
)

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT
    )

def add_cors_middleware(app: FastAPI):
    """CORS 미들웨어 추가"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=CORS_CREDENTIALS,
        allow_methods=CORS_METHODS,
        allow_headers=CORS_HEADERS,
    )

def setup_middleware(app: FastAPI):
    """모든 미들웨어 설정"""
    setup_logging()
    add_cors_middleware(app)