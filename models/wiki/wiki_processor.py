import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from vectordb.chroma_store import ChromaDBManager
from models.wiki.fetcher.wiki_fetcher import WikiFetcher
import logging
import time
import numpy as np
from datetime import datetime
import pytz

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class WikiProcessor:
    def __init__(
        self,
        model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
        embed_func=None
    ):
        if embed_func is None:
            embed_func = ChromaDBManager()
        logger.info("Initializing WikiProcessor...")
        self.embed_func = embed_func.embed_and_store
        logger.info("WikiProcessor initialization complete")
        
    def process_wiki(self, state: dict) -> dict:
        logger.info(f"Starting wiki processing for project_id: {state.get('project_id')}")
        content = state.get("content")
        
        if not content:
            logger.error("'content' key missing in input state")
            raise KeyError("'content' 키가 없습니다.")

        logger.info("Generating processing...")

        embedding_start_time = time.time()
        
        max_chunk_size = 2000 
        
        if len(content) > max_chunk_size:
            print(f"Content length {len(content)} chars exceeds max {max_chunk_size}. Chunking activated.")
            
            chunks = [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}...")
                
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


    async def process_diff_files(self, state: dict) -> dict:
        logger.info(f"Starting wiki processing for project_id: {state.project_id}")
        fetcher = WikiFetcher(state.project_id, state.url)
        
        file_contents = fetcher.get_diff_files()
        logger.info(f"Processing {len(file_contents)} files")

        all_embedding_log_ids = []

        for relative_path, content in file_contents.items():
            result = self.process_wiki({
                "project_id": state.project_id,
                "content": content, 
                "url": state.url,
                "updated_at": datetime.now(pytz.timezone("Asia/Seoul")).isoformat(),
                "document_path": relative_path,
            })
            if result and result.get("embedding_log_id"):
                all_embedding_log_ids.append(result["embedding_log_id"])

        return {
            "message": f"wiki: {len(file_contents)} files processed",
            "embedding_log_ids_processed": all_embedding_log_ids 
        }
