from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager
import logging
import chromadb
import httpx
import asyncio
from typing import List, Optional, Dict, Any
from functools import lru_cache
from llm.wiki_chain import WikiSummarizer
from llm.meeting_chain import MeetingTaskParser
from config import CHROMA_HOST, CHROMA_PORT
from dotenv import load_dotenv
import os

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
task_parser = MeetingTaskParser()

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
    content: str
    updated_at: str
    
    @validator('content')
    def validate_content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v

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

class TaskResult(BaseModel):
    position: str
    tasks: List[Dict[str, Any]]
    is_last: bool

class CallbackPayload(BaseModel):
    message: str
    detail: TaskResult

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

@app.post("/ai/wiki", status_code=status.HTTP_202_ACCEPTED)
async def summarize_wiki(
    input: WikiInput, 
    background_tasks: BackgroundTasks,
    chroma_client=Depends(get_chroma_client)
):
    """Process and summarize wiki content in the background."""
    if not chroma_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ChromaDB service is currently unavailable"
        )
    
    # Add task to background processing
    background_tasks.add_task(wiki_chain.summarize_wiki, input)
    
    return {
        "message": "Wiki summarization started",
        "project_id": input.project_id
    }

@app.post("/projects/{project_id}/notes", status_code=status.HTTP_202_ACCEPTED)
async def receive_meeting_note(
    project_id: int, 
    input: MeetingNote,
    background_tasks: BackgroundTasks
):
    """Process meeting notes and extract tasks for different positions."""
    if project_id != input.project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Project ID in URL does not match project ID in request body"
        )
    
    # Add task to background processing
    background_tasks.add_task(
        process_meeting_note, 
        project_id=project_id,
        content=input.content,
        positions=input.position
    )
    
    return {
        "message": "Meeting note processing started",
        "project_id": project_id,
        "positions": input.position
    }

# Background processing functions
async def process_meeting_note(project_id: int, content: str, positions: List[str]):
    """Process meeting note content for multiple positions concurrently."""
    try:
        # Create tasks for each position to process concurrently
        tasks = []
        for idx, position in enumerate(positions):
            is_last = idx == len(positions) - 1
            tasks.append(
                process_position(
                    project_id=project_id,
                    content=content,
                    position=position,
                    is_last=is_last
                )
            )
        
        # Run tasks concurrently
        await asyncio.gather(*tasks)
        logger.info(f"Completed processing meeting note for project {project_id}")
    
    except Exception as e:
        logger.error(f"Error processing meeting note for project {project_id}: {str(e)}")
        # Could implement a notification system here for failed jobs

async def process_position(project_id: int, content: str, position: str, is_last: bool):
    """Process meeting note for a specific position and send results to backend."""
    try:
        # Process the position
        result = task_parser.summarize_and_generate_tasks(
            project_id=project_id,
            meeting_note=content,
            position=position
        )
        
        # Prepare callback payload
        payload = {
            "message": "subtasks_created",
            "detail": {
                "position": position,
                "tasks": result.get("tasks", []),
                "is_last": is_last
            }
        }
        
        # Send results to backend with retry
        await send_result_to_backend(project_id, payload, max_retries=3)
        logger.info(f"Completed processing position {position} for project {project_id}")
    
    except Exception as e:
        logger.error(f"Error processing position {position} for project {project_id}: {str(e)}")

async def send_result_to_backend(project_id: int, result: dict, max_retries: int = 3):
    """Send results to backend with retry logic."""
    backend_callback_url = f"{BE_URL}/ai-callback/projects/{project_id}/preview"
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(backend_callback_url, json=result)
                response.raise_for_status()
                logger.info(f"Successfully sent callback to backend for project {project_id}")
                return
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during callback for project {project_id}: {e.response.status_code} - {e.response.text}")
            # Don't retry for 4xx errors (except 429)
            if e.response.status_code < 500 and e.response.status_code != 429:
                break
            retry_count += 1
        
        except Exception as e:
            logger.error(f"Error sending callback for project {project_id}: {str(e)}")
            retry_count += 1
        
        # Exponential backoff
        if retry_count < max_retries:
            wait_time = 2 ** retry_count
            logger.info(f"Retrying callback in {wait_time} seconds (attempt {retry_count+1}/{max_retries})")
            await asyncio.sleep(wait_time)
    
    logger.error(f"Failed to send callback to backend after {max_retries} attempts for project {project_id}")