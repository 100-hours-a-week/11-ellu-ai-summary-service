import chromadb
from .embed_model import CustomEmbeddingFunction
from app.config import CHROMA_HOST, CHROMA_PORT

class ChromaDBManager:
    def __init__(self, collection_name="wiki_summaries", host=CHROMA_HOST, port=CHROMA_PORT):
        """Initialize ChromaDB connection and collection."""
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_function = CustomEmbeddingFunction()
    
    def embed_and_store(self, summary: str, metadata: dict):
        """Embed and store document in ChromaDB."""
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
        """Search for similar documents."""
        query_embedding = self.embedding_function([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        return results
    
    def get_by_project_id(self, project_id):
        """Retrieve documents by project ID."""
        return self.collection.get(where={"project_id": project_id})
    
    def delete_by_project_id(self, project_id):
        """Delete documents by project ID."""
        return self.collection.delete(where={"project_id": project_id})


default_db_manager = ChromaDBManager()
