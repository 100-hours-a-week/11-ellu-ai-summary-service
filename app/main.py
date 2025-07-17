from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status, Response, UploadFile, File, Form
from contextlib import asynccontextmanager
import logging
import json
import httpx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from .config import API_TITLE, API_DESCRIPTION, BE_URL, AI_NOTES_URL
from .middleware import setup_middleware
from .dependencies import (
    chroma_dependency,
    database_dependency, 
    wiki_processor_dependency,
    meeting_workflow_dependency
)
from .exceptions import (
    raise_chroma_unavailable,
    raise_invalid_wiki_url,
    raise_project_id_mismatch
)
from schemas.main_schema import WikiInput, MeetingNote, InsertInfo
import tempfile
import os
from app.exceptions import raise_unsupported_audio_extension, raise_audio_file_save_error
from models.stt.audio_transcriber import GeminiSTT

logger = logging.getLogger(__name__)

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    logger.info("Application starting up...")
    # Initialize any resources here
    
    yield
    
    # Cleanup tasks on shutdown
    logger.info("Application shutting down...")
    # Close any connections here

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# Setup middleware
setup_middleware(app)



def detect_url_type(url: str) -> str:
    if "/wiki" in url and "github.com" in url:
        return "github_wiki"
    else:
        return "general_web"

async def validate_general_url(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.head(url)
            return 200 <= response.status_code < 400
    except Exception as e:
        logger.error(f"URL 검증 실패: {url}, 오류: {e}")
        return False
    
    
# Utility functions
async def validate_github_url_http(url: str) -> bool:
    """GitHub 위키 URL 검증"""
    if not url.endswith("/wiki"):
        logger.error(f"URL does not end with '/wiki': {url}")
        return False
    try:
        base_url = url[:-5]
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.head(base_url)
            
        if response.status_code == 200:
            logger.info(f"Repository accessible: {base_url}")
            return True
        else:
            logger.warning(f"Repository not accessible: {base_url} (status: {response.status_code})")
            return False
            
    except httpx.TimeoutException:
        logger.error(f"HTTP request timed out for {base_url}")
        return False
    except Exception as e:
        logger.error(f"HTTP validation failed for {base_url}: {e}")
        return False

# Callback functions
async def wiki_callback(project_id: int, status: str):
    """위키 처리 완료 후 백엔드에 콜백 전송"""
    try:
        backend_callback_url = f"{BE_URL}/ai-callback/wiki"
        callback_payload = {
            "project_id": project_id,
            "status": status
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(backend_callback_url, json=callback_payload)
            response.raise_for_status()
            logger.info(f"Callback sent successfully for project_id={project_id}")
    except Exception as e:
        logger.error(f"Callback failed for project_id={project_id}: {e}")

async def meeting_note_callback(project_id: int, status: str, task_data: dict = None):
    """회의록 처리 완료 후 백엔드에 콜백 전송"""
    try:
        backend_callback_url = f"{BE_URL}/ai-callback/projects/{project_id}/preview"
        callback_payload = {
            "message": "subtasks_created" if status == "completed" else "failed",
            "detail": []
        }
        
        # 성공 시 태스크 데이터도 함께 전송
        if status == "completed" and task_data and "detail" in task_data:
            callback_payload["detail"] = task_data["detail"]
        logger.info(f"callback_payload: {callback_payload}")
        async with httpx.AsyncClient() as client:
            response = await client.post(backend_callback_url, json=callback_payload)
            response.raise_for_status()
            
            logger.info(f"회의록 콜백 전송 성공 - project_id: {project_id}")
    except Exception as e:
        logger.error(f"회의록 콜백 전송 실패 - project_id: {project_id}, 오류: {e}")

# Background task functions
async def process_and_callback(input: WikiInput, wiki_processor):
    """위키 처리 및 콜백 실행"""
    try:
        await wiki_processor.process_diff_files(input)
        await wiki_callback(input.project_id, "completed")
    except ValueError as ve:
        logger.error(f"Invalid URL for project_id={input.project_id}: {ve}")
    except Exception as e:
        logger.error(f"Wiki summarization failed for project_id={input.project_id}: {e}")
        await wiki_callback(input.project_id, "failed")

async def process_web_and_callback(input: WikiInput, wiki_processor):
    try:
        from models.wiki.fetcher.doc_fetcher import DocFetcher
        processor = DocFetcher(input.project_id, input.url)
        
        file_contents = await processor.get_diff_files()
        
        if not file_contents:
            raise Exception("콘텐츠를 가져올 수 없습니다")
        
        for relative_path, content in file_contents.items():
            result_wiki = wiki_processor.process_wiki({
                "project_id": input.project_id,
                "content": content, 
                "url": input.url,
                "updated_at": input.updated_at,
                "document_path": relative_path,
            })
        
        await wiki_callback(input.project_id, "completed")
        
    except Exception as e:
        logger.error(f"웹 콘텐츠 처리 실패 - project_id: {input.project_id}, 오류: {e}")
        await wiki_callback(input.project_id, "failed")


# API endpoints
@app.get("/", status_code=status.HTTP_200_OK)
def read_root(chroma_client=Depends(chroma_dependency)):
    """Health check endpoint that verifies ChromaDB connection."""
    if chroma_client:
        try:
            heartbeat = chroma_client.heartbeat()
            return {"status": "ok", "chroma": heartbeat}
        except Exception as e:
            logger.error(f"ChromaDB heartbeat check failed: {e}")
            return {"status": "partial", "error": str(e)}
    else:
        logger.warning("ChromaDB client not available")
        return {"status": "degraded", "message": "ChromaDB not connected"}



def is_allowed_domain(url: str) -> bool:
    try:
        if not url.startswith('https://'):
            return False
        
        # URL에서 도메인 추출
        domain_part = url[8:].split('/')[0].split('?')[0].split('#')[0]
        
        allowed_domains = [
            'github.com',
            'notion.com',
            'notion.so', 
            'notion.site',
            'docs.google.com',
        ]
        
        for domain in allowed_domains:
            if domain_part == domain or domain_part.endswith('.' + domain):
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"URL 검증 실패: {url}, 오류: {e}")
        return False



    
@app.post("/ai/wiki", status_code=status.HTTP_202_ACCEPTED)
async def summarize_wiki(
    input: WikiInput,
    background_tasks: BackgroundTasks,
    chroma_client=Depends(chroma_dependency),
    wiki_processor=Depends(wiki_processor_dependency)
):
    """위키 및 일반 웹사이트 처리"""
    if not chroma_client:
        raise_chroma_unavailable()
    
    url_type = detect_url_type(input.url)
    
    if url_type == "github_wiki":
        logger.info(f"GitHub Wiki 감지 - URL: {input.url}")
        if not await validate_github_url_http(input.url):
            raise_invalid_wiki_url()
        
        background_tasks.add_task(process_and_callback, input, wiki_processor)
        
    else:
        logger.info(f"일반 웹사이트 감지 - URL: {input.url}")


        if not is_allowed_domain(input.url):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="해당 도메인은 지원하지 않습니다. 지원 사이트: GitHub, Notion, Google Docs"
            ) 

        if not await validate_general_url(input.url):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="접근할 수 없는 URL입니다"
            )
        
        background_tasks.add_task(process_web_and_callback, input, wiki_processor)
    
    return {"message": "콘텐츠 처리가 시작되었습니다"}



@app.delete("/projects/{project_id}/wiki", status_code=status.HTTP_200_OK)
async def delete_project(
    project_id: int,
    background_tasks: BackgroundTasks,
    chroma_client=Depends(chroma_dependency)
):
    if not chroma_client:
        raise_chroma_unavailable()
    
    background_tasks.add_task(delete_project_data, project_id)
    return {
        "message": "S3_deleted",
        "data": None
    }

async def delete_project_data(project_id: int):
    try:
        # S3 삭제
        from models.wiki.fetcher.wiki_fetcher import WikiFetcher
        fetcher = WikiFetcher(project_id, "")
        s3_result = fetcher.delete_project_data()
        
        # ChromaDB 삭제
        from vectordb.chroma_store import ChromaDBManager
        chroma_manager = ChromaDBManager()
        chroma_result = chroma_manager.delete_by_project_id(project_id)
        
        if s3_result and chroma_result:
            logger.info(f"프로젝트 {project_id} 전체 AI 데이터 삭제 완료 (S3 + ChromaDB)")
        else:
            logger.warning(f"프로젝트 {project_id} 일부 삭제 실패 - S3: {s3_result}, ChromaDB: {chroma_result}")
            
    except Exception as e:
        logger.error(f"프로젝트 {project_id} 삭제 중 오류: {e}")


        
@app.post("/projects/{project_id}/notes", status_code=status.HTTP_202_ACCEPTED)
async def receive_meeting_note(
    project_id: int,
    input: MeetingNote,
    task_parser=Depends(meeting_workflow_dependency),
    db_engine=Depends(database_dependency)
):
    """회의록을 받아서 태스크로 분해하고 백엔드에 콜백 전송"""
    if project_id != input.project_id:
        raise_project_id_mismatch()

    # 동기적으로 처리하고 결과 반환
    result = await process_meeting_note_sync(input, project_id, task_parser, db_engine)
    logger.info(f"회의록 처리 완료 - project_id: {project_id}, result: {result}")
    return result



async def process_meeting_note_sync(input: MeetingNote, project_id: int, task_parser, db_engine):
    logger.info(f"회의록 처리 시작 - project_id: {project_id}, content: {input}")
    try:
        # 회의록에서 태스크 추출
        result = await task_parser.arun(
            meeting_note=input.content,
            project_id=project_id,
            position=input.position,
            audio_file_path="dummy_path"  # 오디오 파일 경로는 나중에 처리
        )
        # logger.info(f"result : {result}")
        # 응답 데이터 구성 - 모든 포지션의 태스크를 하나의 배열로 합치기
        response_data = { "message": "subtasks_created", "detail": []}
        for position in result['project_position']:
                response_data["detail"].extend(result[position])
        
        # 사용자 입출력 데이터 DB 저장
        if db_engine:
            try:
                with db_engine.begin() as connection:
                    query = text("""
                        INSERT INTO user_io (user_input, user_output,project_id)
                        VALUES (:user_input, :user_output, :project_id)
                    """)
                    connection.execute(query, {
                        "user_input": input.content,
                        "user_output": json.dumps(response_data, ensure_ascii=False),
                        "project_id" : input.project_id
                    })
                    

                    logger.info("user_io 테이블에 데이터 정상 삽입 완료")
            except SQLAlchemyError as e:
                logger.error(f"user_io 테이블 삽입 실패: {str(e)}")
        
        return(response_data)
        
    except Exception as e:
        logger.error(f"회의록 DB 저장 처리 실패 - project_id: {project_id}, 오류: {e}")
        return {
            "message": "failed",
            "detail": []
        }

@app.post("/ai/audio")
async def audio_upload(file: UploadFile = File(...), project_id: int = Form(...), background_tasks: BackgroundTasks = None):
    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".mp4", ".aac", ".flac", ".m4a", ".mpga", ".mpeg", ".opus", ".pcm", ".webm"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        raise_unsupported_audio_extension(ext, SUPPORTED_EXTENSIONS)
    try:
        # 업로드된 파일을 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 성공 메시지 반환
        background_tasks.add_task(process_audio_and_send_note, tmp_path, project_id)
        return {"message": "audio_file_success"}
    except Exception as e:
        raise_audio_file_save_error(e)

async def process_audio_and_send_note(tmp_path, project_id):
    try:
        stt = GeminiSTT()
        stt_result = stt.run_stt_and_return_text(tmp_path, project_id)
        text = stt_result.get("text", "")
        if text.strip():
            note_payload = {
                "project_id": project_id,
                "content": text,
                "position": ["all"]
            }
            async with httpx.AsyncClient() as client:
                notes_url = AI_NOTES_URL
                await client.post(notes_url, json=note_payload)
        os.remove(tmp_path)
    except Exception as e:
        logger.error(f"Error for file {tmp_path}, project_id {project_id}: {e}", exc_info=True)

@app.get("/warmup", status_code=200)
def warmup():
    from app.dependencies import get_chroma_client
    from vectordb.embed_model import CustomEmbeddingFunction

    logger.info("[WARMUP] 시작")

    # ChromaDB 연결 시도
    try:
        client = get_chroma_client()
        if client:
            client.heartbeat()
            logger.info("[WARMUP] ChromaDB 연결 성공")
        else:
            logger.warning("[WARMUP] ChromaDB 클라이언트 없음")
    except Exception as e:
        logger.warning(f"[WARMUP] ChromaDB 예열 실패: {e}")

    # 모델 로딩 시도
    try:
        embedder = CustomEmbeddingFunction()
        embedder(["warmup"])  # dummy call
        logger.info("[WARMUP] Hugging Face 모델 로딩 성공")
    except Exception as e:
        logger.warning(f"[WARMUP] 모델 예열 실패: {e}")

    return {"status": "warmup complete"}


@app.post("/projects/{project_id}/insert", status_code=status.HTTP_200_OK)
async def insert_user_info(
    project_id: int,
    input: InsertInfo,
    background_tasks: BackgroundTasks,
    db_engine=Depends(database_dependency)
):
    if db_engine:
        try:
            with db_engine.begin() as connection:
                # 1. 먼저 해당 project_id의 최대 id 조회
                max_id_query = text("""
                    SELECT MAX(id) as max_id
                    FROM user_io 
                    WHERE project_id = :project_id
                """)
                
                result = connection.execute(max_id_query, {"project_id": project_id})
                max_id_row = result.fetchone()
                
                if not max_id_row or max_id_row.max_id is None:
                    logger.warning(f"project_id {project_id}에 해당하는 레코드를 찾을 수 없습니다")
                    return {"status": "error", "message": f"project_id {project_id}에 해당하는 레코드가 없습니다"}
                
                max_id = max_id_row.max_id
                
                
                # 2. 해당 id로 업데이트
                update_query = text("""
                    UPDATE user_io 
                    SET user_choice = :user_choice
                    WHERE id = :max_id
                """)
                
                update_result = connection.execute(update_query, {
                    "user_choice": json.dumps(input.content, ensure_ascii=False),
                    "max_id": max_id
                })
                
                if update_result.rowcount == 0:
                    return {"status": "error", "message": "업데이트 실패"}
                
                logger.info(f"user_io 테이블 ID {max_id}에 데이터 정상 업데이트 완료")
                return {
                    "status": "success", 
                    "message": "데이터가 업데이트되었습니다",
                    
                }
                
        except SQLAlchemyError as e:
            logger.error(f"user_io 테이블 업데이트 실패: {str(e)}")
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": "데이터베이스 연결이 없습니다"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
