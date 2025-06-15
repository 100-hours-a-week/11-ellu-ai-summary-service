from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import logging
import chromadb
import json
from typing import List, Dict, Any
from functools import lru_cache
from wiki.wiki_chain import WikiSummarizer
from llm.graph import MeetingWorkflow
from config import CHROMA_HOST, CHROMA_PORT
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text 
from sqlalchemy.exc import SQLAlchemyError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import httpx
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
BE_URL = os.getenv("BE_URL")
if not BE_URL:
    logger.error("BE_URL environment variable not set")
    raise ValueError("BE_URL environment variable is required")

# Initialize chains
wiki_chain = WikiSummarizer()
task_parser = MeetingWorkflow()

# ChromaDB dependency
@lru_cache()
def get_chroma_client():
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # Test connection
        client.heartbeat()
        logger.info("Connected to ChromaDB successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        return None

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
    title="AI Processing API",
    description="API for processing wiki summaries and meeting notes",
    lifespan=lifespan,
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class WikiInput(BaseModel):
    project_id: int
    url: str
    updated_at: str
    
    # @validator('content')
    # def validate_content_not_empty(cls, v):
    #     if not v.strip():
    #         raise ValueError("Content cannot be empty")
    #     return v

class MeetingNote(BaseModel):
    project_id: int
    content: str
    position: List[str] = Field(..., description="List of positions to parse tasks for")
    
    @validator('content')
    def validate_content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v
        
    @validator('position')
    def validate_positions(cls, v):
        if not v:
            raise ValueError("At least one position must be provided")
        return v

# API endpoints
@app.get("/", status_code=status.HTTP_200_OK)
def read_root(chroma_client=Depends(get_chroma_client)):
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


# 위키 콜백 함수
async def wiki_callback(project_id: int, status: str):
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

async def process_and_callback(input: WikiInput):
    try:
        await wiki_chain.summarize_diff_files(input)
        await wiki_callback(input.project_id, "completed")
    except ValueError as ve:
        logger.error(f"Invalid URL for project_id={input.project_id}: {ve}")
    except Exception as e:
        logger.error(f"Wiki summarization failed for project_id={input.project_id}: {e}")
        await wiki_callback(input.project_id, "failed")

def validate_github_url(url: str) -> bool:
    if not url.endswith("/wiki"):
        return False
    try:
        base_url = url[:-5]
        git_url = f"{base_url}.wiki.git"
        subprocess.run(["git", "ls-remote", git_url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    
@app.post("/ai/wiki", status_code=status.HTTP_202_ACCEPTED)
async def summarize_wiki(
    input: WikiInput,
    background_tasks: BackgroundTasks,
    chroma_client=Depends(get_chroma_client)
):
    if not chroma_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChromaDB service is currently unavailable"
        )
    # url 검증
    if not validate_github_url(input.url):
        raise HTTPException(status_code=400, 
                            detail="유효하지 않은 URL 입니다. wiki url을 그대로 올려주세요. ('/wiki'로 끝나야 합니다.)")
    # 백그라운드 실행
    background_tasks.add_task(process_and_callback, input)
    return {"message": "Wiki summarization started"}


# 회의록 콜백 함수
async def meeting_note_callback(project_id: int, status: str, task_data: dict = None):
    """회의록 처리 완료 후 백엔드에 콜백 전송"""
    try:
        backend_callback_url = f"{BE_URL}/ai-callback/projects/{project_id}/preview"
        callback_payload = {
            "status": status
        }
        
        # 성공 시 태스크 데이터도 함께 전송
        if status == "completed" and task_data:
            callback_payload["tasks"] = task_data
            
        async with httpx.AsyncClient() as client:
            response = await client.post(backend_callback_url, json=callback_payload)
            response.raise_for_status()
            logger.info(f"회의록 콜백 전송 성공 - project_id: {project_id}")
    except Exception as e:
        logger.error(f"회의록 콜백 전송 실패 - project_id: {project_id}, 오류: {e}")

# 회의록 처리 및 콜백 함수
async def process_meeting_note_and_callback(input: MeetingNote, project_id: int):
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
        DB_URL = os.getenv("User_info_db")
        if DB_URL:
            engine = create_engine(DB_URL)
            try:
                with engine.begin() as connection:
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

@app.post("/projects/{project_id}/notes", status_code=status.HTTP_202_ACCEPTED)
async def receive_meeting_note(
    project_id: int,
    input: MeetingNote,
    background_tasks: BackgroundTasks,
):
    """회의록을 받아서 태스크로 분해하고 백엔드에 콜백 전송"""
    if project_id != input.project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL의 프로젝트 ID와 요청 본문의 프로젝트 ID가 일치하지 않습니다"
        )

    # 백그라운드에서 회의록 처리
    background_tasks.add_task(process_meeting_note_and_callback, input, project_id)
    
    return {
        "message": "회의록 처리가 시작되었습니다",
        "project_id": project_id
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)