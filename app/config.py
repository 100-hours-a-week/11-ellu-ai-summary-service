# config.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 설정
BE_URL = os.getenv("BE_URL", "http://localhost:8000")
VLLM_URL = os.getenv("VLLM_URL", "http://vllm-server:8000")

# LLM 설정
GPT_MODEL = "gpt-4o"
TEMPERATURE = 0.1
MODEL_KWARGS = {"response_format": {"type": "json_object"}}

# ChromaDB 설정
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
Hugging_FACE_KEY = os.getenv("HUGGINGFACE_API_KEY")

# CORS Configuration
CORS_ORIGINS = ["*"]  # Replace with specific origins in production
CORS_CREDENTIALS = True
CORS_METHODS = ["*"]
CORS_HEADERS = ["*"]

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API Configuration
API_TITLE = "AI Processing API"
API_DESCRIPTION = "API for processing wiki summaries and meeting notes"
API_VERSION = "1.0.0"

USER_INFO_DB_URL=os.getenv("User_info_db")

# AWS S3
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")


# Pinecone 설정
RETRIEVER_CLASS = "BasicRetriever" 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "tech-blogs-en")
