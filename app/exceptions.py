from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

class ChromaDBConnectionError(Exception):
    """ChromaDB 연결 오류"""
    pass

class WikiValidationError(Exception):
    """위키 URL 검증 오류"""
    pass

class CallbackError(Exception):
    """콜백 전송 오류"""
    pass

class DatabaseError(Exception):
    """데이터베이스 오류"""
    pass

# HTTP Exception handlers
def raise_chroma_unavailable():
    """ChromaDB 서비스 사용 불가 예외"""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="ChromaDB service is currently unavailable"
    )

def raise_invalid_wiki_url():
    """유효하지 않은 위키 URL 예외"""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, 
        detail="유효하지 않은 URL 입니다. wiki url을 그대로 올려주세요. ('/wiki'로 끝나야 합니다.)"
    )

def raise_project_id_mismatch():
    """프로젝트 ID 불일치 예외"""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="URL의 프로젝트 ID와 요청 본문의 프로젝트 ID가 일치하지 않습니다"
    )

def raise_audio_file_download_error(audio_file: str):
    """S3 URL 오디오 파일 다운로드 실패 예외"""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Failed to download audio file from S3: {audio_file}"
    )

# Global exception handler
async def global_exception_handler(request, exc):
    """전역 예외 처리기"""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )