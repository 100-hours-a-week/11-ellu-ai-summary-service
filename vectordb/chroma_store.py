import os
import time
import chromadb
from .embed_model import CustomEmbeddingFunction
from app.config import CHROMA_HOST, CHROMA_PORT
import logging

try:
    from opentelemetry.trace import get_tracer, SpanKind
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    get_tracer = lambda name: None
    SpanKind = type('SpanKind', (), {'CLIENT': None})

logger = logging.getLogger(__name__)

class ChromaDBManager:
    def __init__(self, collection_name="wiki_summaries", host=CHROMA_HOST, port=CHROMA_PORT):
        self._host = os.getenv("CHROMA_HOST", host)
        self._port = int(os.getenv("CHROMA_PORT", port))
        self.collection_name = collection_name
        self.embedding_function = CustomEmbeddingFunction()

        self._client = None
        self._collection = None
        self.tracer = get_tracer(__name__) if OTEL_AVAILABLE else None

    def _init_client(self):
        retries = 5
        delay = 2
        for i in range(retries):
            try:
                return chromadb.HttpClient(host=self._host, port=self._port)
            except Exception as e:
                logger.warning(f"ChromaDB 연결 실패 ({i+1}/{retries}): {e}")
                time.sleep(delay)
        raise ConnectionError(f"ChromaDB 연결 실패: {self._host}:{self._port}")

    def get_client(self):
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def get_collection(self):
        if self._collection is None:
            if OTEL_AVAILABLE and self.tracer:
                with self.tracer.start_as_current_span("chromadb_get_collection", kind=SpanKind.CLIENT) as span:
                    span.set_attribute("peer.service", "chromadb")
                    span.set_attribute("db.operation", "get_or_create_collection")
                    span.set_attribute("db.collection.name", self.collection_name)
                    self._collection = self.get_client().get_or_create_collection(name=self.collection_name)
            else:
                self._collection = self.get_client().get_or_create_collection(name=self.collection_name)
        return self._collection

    def embed_and_store(self, summary: str, metadata: dict):
        doc_id = f"{metadata['project_id']}_{metadata.get('document_path', 'unknown')}_{metadata.get('updated_at', 'unknown')}"
        
        if OTEL_AVAILABLE and self.tracer:
            with self.tracer.start_as_current_span("chromadb_embed_and_store", kind=SpanKind.CLIENT) as span:
                span.set_attribute("peer.service", "chromadb")
                span.set_attribute("db.operation", "upsert")
                span.set_attribute("db.collection.name", self.collection_name)
                span.set_attribute("document.id", doc_id)
                
                try:
                    self.get_collection().delete(ids=[doc_id])
                except:
                    pass  # 기존 문서가 없으면 무시

                embedding = self.embedding_function([summary])[0]
                self.get_collection().add(
                    ids=[doc_id], 
                    documents=[summary], 
                    embeddings=[embedding], 
                    metadatas=[metadata]
                )

                print(f"문서 저장 완료: {doc_id}")
                return doc_id
        else:
            try:
                self.get_collection().delete(ids=[doc_id])
            except:
                pass  # 기존 문서가 없으면 무시

            embedding = self.embedding_function([summary])[0]
            self.get_collection().add(
                ids=[doc_id], 
                documents=[summary], 
                embeddings=[embedding], 
                metadatas=[metadata]
            )

            print(f"문서 저장 완료: {doc_id}")
            return doc_id

    def search(self, query_text, n_results=5, where_filter=None):
        if OTEL_AVAILABLE and self.tracer:
            with self.tracer.start_as_current_span("chromadb_search", kind=SpanKind.CLIENT) as span:
                span.set_attribute("peer.service", "chromadb")
                span.set_attribute("db.operation", "query")
                span.set_attribute("db.collection.name", self.collection_name)
                span.set_attribute("search.results_limit", n_results)
                query_embedding = self.embedding_function([query_text])[0]
                return self.get_collection().query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filter
                )
        else:
            query_embedding = self.embedding_function([query_text])[0]
            return self.get_collection().query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )

    def get_by_project_id(self, project_id):
        if OTEL_AVAILABLE and self.tracer:
            with self.tracer.start_as_current_span("chromadb_get_by_project", kind=SpanKind.CLIENT) as span:
                span.set_attribute("peer.service", "chromadb")
                span.set_attribute("db.operation", "get")
                span.set_attribute("db.collection.name", self.collection_name)
                span.set_attribute("project.id", str(project_id))
                return self.get_collection().get(where={"project_id": project_id})
        else:
            return self.get_collection().get(where={"project_id": project_id})

    def delete_by_project_id(self, project_id):
        if OTEL_AVAILABLE and self.tracer:
            with self.tracer.start_as_current_span("chromadb_delete_by_project", kind=SpanKind.CLIENT) as span:
                span.set_attribute("peer.service", "chromadb")
                span.set_attribute("db.operation", "delete")
                span.set_attribute("db.collection.name", self.collection_name)
                span.set_attribute("project.id", str(project_id))
                
                try:
                    existing_docs = self.get_collection().get(where={"project_id": project_id})
                    doc_count = len(existing_docs['ids']) if existing_docs['ids'] else 0
                    span.set_attribute("documents.count", doc_count)

                    if doc_count == 0:
                        logger.info(f"프로젝트 {project_id}: 삭제할 ChromaDB 문서가 없습니다")
                        return True

                    self.get_collection().delete(where={"project_id": project_id})
                    logger.info(f"프로젝트 {project_id} ChromaDB 데이터 삭제 완료 ({doc_count}개 문서)")
                    return True

                except Exception as e:
                    logger.error(f"프로젝트 {project_id} ChromaDB 삭제 실패: {e}")
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                    return False
        else:
            try:
                existing_docs = self.get_collection().get(where={"project_id": project_id})
                doc_count = len(existing_docs['ids']) if existing_docs['ids'] else 0

                if doc_count == 0:
                    logger.info(f"프로젝트 {project_id}: 삭제할 ChromaDB 문서가 없습니다")
                    return True

                self.get_collection().delete(where={"project_id": project_id})
                logger.info(f"프로젝트 {project_id} ChromaDB 데이터 삭제 완료 ({doc_count}개 문서)")
                return True

            except Exception as e:
                logger.error(f"프로젝트 {project_id} ChromaDB 삭제 실패: {e}")
                return False