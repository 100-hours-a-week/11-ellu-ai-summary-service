import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from vectordb.chroma_store import ChromaDBManager
from models.wiki.wiki_fetcher import WikiFetcher
import torch 
import logging
import time
import numpy as np
from datetime import datetime
import pytz

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class WikiSummarizer:
    def __init__(
        self,
        model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        embed_func=ChromaDBManager()
    ):
        logger.info("Initializing WikiSummarizer...")
        self.embed_func = embed_func.embed_and_store
        logger.info("WikiSummarizer initialization complete")

    def summarize_wiki(self, state: dict) -> dict:
        logger.info(f"Starting wiki summarization for project_id: {state.get('project_id')}")
        content = state.get("content")
        
        if not content:
            logger.error("'content' key missing in input state")
            raise KeyError("'content' 키가 없습니다.")

        logger.info("Generating summary...")

        embedding_start_time = time.time()
        
        max_chunk_size = 2000 
        
        if len(content) > max_chunk_size:
            print(f"Content length {len(content)} chars exceeds max {max_chunk_size}. Chunking activated.")
            
            # 문자 기준 청킹 (토큰 오버헤드 제거)
            chunks = [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}...")
                
                # 각 청크별로 별도 문서로 저장
                chunk_metadata = {
                    "project_id": state.get("project_id"),
                    "repo_url": state["url"], 
                    "document_path": f"{state.get('document_path', 'unknown')}_chunk_{i+1}",
                    "updated_at": state.get("updated_at")
                }
                
                print(f"Storing chunk {i+1} with metadata: {chunk_metadata}")
                self.embed_func(chunk, chunk_metadata)
        else:
            metadata = {
                "project_id": state.get("project_id"),
                "repo_url": state["url"], 
                "document_path": state.get("document_path", "unknown"),
                "updated_at": state.get("updated_at")
            }
            
            print(f"Storing content with metadata: {metadata}")
            self.embed_func(content, metadata)

        embedding_generation_time = time.time() - embedding_start_time
        
        print(f"Wiki processing completed successfully in {embedding_generation_time:.2f} seconds")
        return {"message": "wiki_saved"}


    async def summarize_diff_files(self, state: dict) -> dict:
        logger.info(f"Starting wiki summarization for project_id: {state.project_id}")
        fetcher = WikiFetcher(state.project_id, state.url)
        
        file_contents = fetcher.get_diff_files()
        logger.info(f"Summarizing {len(file_contents)} files")

        all_embedding_log_ids = []

        for relative_path, content in file_contents.items():
            result = self.summarize_wiki({
                "project_id": state.project_id,
                "content": content, 
                "url": state.url,
                "updated_at": datetime.now(pytz.timezone("Asia/Seoul")).isoformat(),
                "document_path": relative_path,
            })
            if result and result.get("embedding_log_id"):
                all_embedding_log_ids.append(result["embedding_log_id"])

        return {
            "message": f"wiki: {len(file_contents)} files summarized",
            "embedding_log_ids_processed": all_embedding_log_ids 
        }
