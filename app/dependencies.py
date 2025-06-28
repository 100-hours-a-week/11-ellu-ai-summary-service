import chromadb
import logging
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from models.wiki.wiki_chain import WikiSummarizer
from nodes.graph import MeetingWorkflow
from .config import CHROMA_HOST, CHROMA_PORT, USER_INFO_DB_URL
from .exceptions import ChromaDBConnectionError, DatabaseError
from vectordb.chroma_store import ChromaDBManager

logger = logging.getLogger(__name__)

# ChromaDB dependency
@lru_cache()
def get_chroma_client():
    """ChromaDB 클라이언트 인스턴스를 반환합니다."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        logger.info("Connected to ChromaDB successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return None

# Database engine dependency
@lru_cache()
def get_database_engine():
    """데이터베이스 엔진을 반환합니다."""
    if not USER_INFO_DB_URL:
        logger.warning("USER_INFO_DB_URL not configured")
        return None
    
    try:
        engine = create_engine(USER_INFO_DB_URL)
        logger.info("Database engine created successfully")
        return engine
    except SQLAlchemyError as e:
        logger.error(f"Failed to create database engine: {e}")
        raise DatabaseError(f"Database connection failed: {e}")

# Service dependencies
@lru_cache()
def get_wiki_summarizer():
    try:
        chroma = ChromaDBManager() 
        return WikiSummarizer(embed_func=chroma)
    except Exception as e:
        logger.error(f"Failed to initialize WikiSummarizer: {e}")
        raise

@lru_cache()
def get_meeting_workflow():
    """MeetingWorkflow 인스턴스를 반환합니다."""
    try:
        return MeetingWorkflow()
    except Exception as e:
        logger.error(f"Failed to initialize MeetingWorkflow: {e}")
        raise

# Dependency functions for FastAPI
def chroma_dependency():
    """FastAPI dependency for ChromaDB client"""
    return get_chroma_client()

def database_dependency():
    """FastAPI dependency for database engine"""
    return get_database_engine()

def wiki_summarizer_dependency():
    """FastAPI dependency for WikiSummarizer"""
    return get_wiki_summarizer()

def meeting_workflow_dependency():
    """FastAPI dependency for MeetingWorkflow"""
    return get_meeting_workflow()