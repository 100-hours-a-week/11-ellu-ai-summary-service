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

def raise_unsupported_audio_extension(ext, supported_exts):
    """지원하지 않는 오디오 파일 확장자 예외"""
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"지원하지 않는 오디오 파일 형식입니다: {ext}. 지원 형식: {', '.join(supported_exts)}"
    )

def raise_audio_file_save_error(e):
    """오디오 파일 임시 저장 실패 예외"""
    logger.error(f"오디오 파일 저장 실패: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"오디오 파일 저장 중 오류가 발생했습니다: {str(e)}"
    )

# Global exception handler
async def global_exception_handler(request, exc):
    """전역 예외 처리기"""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal server error"
    )