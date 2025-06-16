from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status, Response
from contextlib import asynccontextmanager
import logging
import json
import httpx
import subprocess
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .config import API_TITLE, API_DESCRIPTION, BE_URL
from .middleware import setup_middleware
from .dependencies import (
    chroma_dependency,
    database_dependency, 
    wiki_summarizer_dependency,
    meeting_workflow_dependency
)
from .exceptions import (
    raise_chroma_unavailable,
    raise_invalid_wiki_url,
    raise_project_id_mismatch
)
from schemas.main_schema import WikiInput, MeetingNote

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

# Utility functions
def validate_github_url(url: str) -> bool:
    """GitHub 위키 URL 검증"""
    if not url.endswith("/wiki"):
        return False
    try:
        base_url = url[:-5]
        git_url = f"{base_url}.wiki.git"
        subprocess.run(
            ["git", "ls-remote", git_url], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
        return True
    except subprocess.CalledProcessError:
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
            logger.info(f"SEND RESPONSE: {response.json()}")
            logger.info(f"회의록 콜백 전송 성공 - project_id: {project_id}")
    except Exception as e:
        logger.error(f"회의록 콜백 전송 실패 - project_id: {project_id}, 오류: {e}")

# Background task functions
async def process_and_callback(input: WikiInput, wiki_chain):
    """위키 처리 및 콜백 실행"""
    try:
        await wiki_chain.summarize_diff_files(input)
        await wiki_callback(input.project_id, "completed")
    except ValueError as ve:
        logger.error(f"Invalid URL for project_id={input.project_id}: {ve}")
    except Exception as e:
        logger.error(f"Wiki summarization failed for project_id={input.project_id}: {e}")
        await wiki_callback(input.project_id, "failed")

async def process_meeting_note_and_callback(input: MeetingNote, project_id: int, task_parser, db_engine):
    """회의록 처리 및 콜백 실행"""
    try:
        # 회의록에서 태스크 추출
        result = task_parser.run(
            meeting_notes=input.content,
            project_id=project_id,
            position=input.position,
        )
        
        # 응답 데이터 구성 - 모든 포지션의 태스크를 하나의 배열로 합치기
        response_data = {"message": "subtasks_created", "detail": []}
        for position in input.position:
            if result.get(position):  # 해당 포지션에 결과가 있는 경우만
                response_data["detail"].extend(result[position])
        
        # 사용자 입출력 데이터 DB 저장
        if db_engine:
            try:
                with db_engine.begin() as connection:
                    query = text("""
                        INSERT INTO user_io (user_input, user_output)
                        VALUES (:user_input, :user_output)
                    """)
                    connection.execute(query, {
                        "user_input": input.content,
                        "user_output": json.dumps(response_data, ensure_ascii=False)
                    })
                    logger.info("user_io 테이블에 데이터 정상 삽입 완료")
            except SQLAlchemyError as e:
                logger.error(f"user_io 테이블 삽입 실패: {str(e)}")
        
        # 성공 콜백 전송
        await meeting_note_callback(project_id, "completed", response_data)
        
    except Exception as e:
        logger.error(f"회의록 처리 실패 - project_id: {project_id}, 오류: {e}")
        # 실패 콜백 전송
        await meeting_note_callback(project_id, "failed")

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

@app.post("/ai/wiki", status_code=status.HTTP_202_ACCEPTED)
async def summarize_wiki(
    input: WikiInput,
    background_tasks: BackgroundTasks,
    chroma_client=Depends(chroma_dependency),
    wiki_chain=Depends(wiki_summarizer_dependency)
):
    """위키 요약 처리"""
    if not chroma_client:
        raise_chroma_unavailable()
    
    # URL 검증
    if not validate_github_url(input.url):
        raise_invalid_wiki_url()
    
    # 백그라운드 실행
    background_tasks.add_task(process_and_callback, input, wiki_chain)
    return {"message": "Wiki summarization started"}

@app.post("/projects/{project_id}/notes", status_code=status.HTTP_202_ACCEPTED)
async def receive_meeting_note(
    project_id: int,
    input: MeetingNote,
    background_tasks: BackgroundTasks,
    task_parser=Depends(meeting_workflow_dependency),
    db_engine=Depends(database_dependency)
):
    """회의록을 받아서 태스크로 분해하고 백엔드에 콜백 전송"""
    if project_id != input.project_id:
        raise_project_id_mismatch()

    # 백그라운드에서 회의록 처리
    background_tasks.add_task(
        process_meeting_note_and_callback, 
        input, 
        project_id, 
        task_parser, 
        db_engine
    )
    
    return {
        "message": "회의록 처리가 시작되었습니다",
        "project_id": project_id
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)