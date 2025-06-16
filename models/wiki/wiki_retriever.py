from vectordb.embed_model import CustomEmbeddingFunction
import chromadb
import logging
import sqlite3
from datetime import datetime
import numpy as np
import time
from app.config import CHROMA_HOST, CHROMA_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_fn = CustomEmbeddingFunction()


class WikiRetriever:
    def __init__(self):
        self.embedding_fn = CustomEmbeddingFunction()
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # collection 생성
        self.collection = self.chroma_client.get_or_create_collection(name="wiki_summaries")


    def retrieve_wiki_context(self, task: str, project_id: int, k: int = 3) -> dict:
        logger.info(f"Retrieving wiki context for task: '{task}' and project_id: {project_id}")
        
        query_embedding_start_time = time.time()
        query_embedding = self.embedding_fn.embed_query(task)
        query_embedding_time = time.time() - query_embedding_start_time
        
        retrieval_start_time = time.time()
        results = None
        retrieval_log_id = None 

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"project_id": project_id}
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return {task: "", "retrieval_log_id": retrieval_log_id} 
        
        documents = results.get("documents", [[]])
        retrieved_docs = len(documents[0]) if documents and documents[0] else 0
        
        avg_retrieved_doc_length = 0
        if documents and documents[0]:
            doc_lengths = [len(self.embedding_fn.tokenizer.encode(doc)) if hasattr(self.embedding_fn, 'tokenizer') else len(doc) for doc in documents[0]]
            avg_retrieved_doc_length = np.mean(doc_lengths) if doc_lengths else 0


        if not documents or not documents[0]:
            logger.warning(f"No documents found for task: {task}")
