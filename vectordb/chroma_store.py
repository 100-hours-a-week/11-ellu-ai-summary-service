import chromadb
from .embed_model import CustomEmbeddingFunction
from app.config import CHROMA_HOST, CHROMA_PORT
import logging

logger = logging.getLogger(__name__)

class ChromaDBManager:
    def __init__(self, collection_name="wiki_summaries", host=CHROMA_HOST, port=CHROMA_PORT):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_function = CustomEmbeddingFunction()
    
    def embed_and_store(self, summary: str, metadata: dict):
        doc_id = f"{metadata['project_id']}_{metadata.get('document_path', 'unknown')}_{metadata.get('updated_at', 'unknown')}"
        
        try:
            self.collection.delete(ids=[doc_id])
        except:
            pass  # 기존 문서가 없으면 무시
        
        # 새 문서 추가
        embedding = self.embedding_function([summary])[0]
        self.collection.add(
            ids=[doc_id], 
            documents=[summary], 
            embeddings=[embedding], 
            metadatas=[metadata]
        )
        
        print(f"문서 저장 완료: {doc_id}")
        return doc_id
    
    def search(self, query_text, n_results=5, where_filter=None):
        query_embedding = self.embedding_function([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        return results
    
    def get_by_project_id(self, project_id):
        return self.collection.get(where={"project_id": project_id})
    
    def delete_by_project_id(self, project_id):
        try:
            existing_docs = self.collection.get(where={"project_id": project_id})
            doc_count = len(existing_docs['ids']) if existing_docs['ids'] else 0
            
            if doc_count == 0:
                logger.info(f"프로젝트 {project_id}: 삭제할 ChromaDB 문서가 없습니다")
                return True
            
            self.collection.delete(where={"project_id": project_id})
            logger.info(f"프로젝트 {project_id} ChromaDB 데이터 삭제 완료 ({doc_count}개 문서)")
            return True
            
        except Exception as e:
            logger.error(f"프로젝트 {project_id} ChromaDB 삭제 실패: {e}")
            return False


default_db_manager = ChromaDBManager()
