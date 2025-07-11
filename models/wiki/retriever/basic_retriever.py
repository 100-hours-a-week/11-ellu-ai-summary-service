from vectordb.embed_model import CustomEmbeddingFunction
import chromadb
import logging
import time 
from app.config import CHROMA_HOST, CHROMA_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_fn = CustomEmbeddingFunction()


class BasicRetriever:
    def __init__(self):
        self.embedding_fn = CustomEmbeddingFunction()
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.collection = self.chroma_client.get_or_create_collection(name="wiki_summaries")


    def retrieve_wiki_context(self, task: str, project_id: int, k: int = 1) -> dict:
        logger.info(f"Retrieving wiki context for task: '{task}' and project_id: {project_id}")
        
        try:
            query_embedding_start_time = time.time()
            query_embedding = self.embedding_fn.embed_query(task)
            # query_embedding_time = time.time() - query_embedding_start_time
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"project_id": project_id}
            )
                        
            documents = results.get("documents", [[]])
            
            if not documents or not documents[0]:
                logger.warning(f"No documents found for task: {task}, project_id: {project_id}")
                return {task: ""}
            
            # 검색된 문서들을 결합
            context = "\n".join(documents[0])
            logger.info(f"Retrieved {len(documents[0])} documents for task: {task}")

            return {task: context}
            
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return {task: ""}
