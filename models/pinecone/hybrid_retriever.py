import os
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
load_dotenv()

logger = logging.getLogger(__name__)

class PineconeHybridRetriever:    
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "tech-blogs-en")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY 환경변수가 설정되지 않았습니다.")
        
        # Pinecone 초기화
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
        logger.info("PineconeHybridRetriever 초기화 완료")
    
    def embed_query(self, text: str) -> List[float]:
        return self.embedding_model.encode([text])[0].tolist()
    
    def search_by_type(self, query: str, source_type: str, data_type: str = None, top_k: int = 5) -> List[Dict]:
        try:
            query_vector = self.embed_query(query)
            
            # 필터 구성
            filter_dict = {"source_type": source_type}
            if data_type:
                filter_dict["data_type"] = data_type
            
            results = self.index.query(
                vector=query_vector,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return [{"score": match.score, "metadata": match.metadata} 
                   for match in results.matches]
            
        except Exception as e:
            logger.error(f"Pinecone 검색 오류 ({source_type}): {e}")
            return []
    
    def search_blog_content(self, query: str, top_k: int = 3) -> List[Dict]: #blog
        return self.search_by_type(query, "official_blog", top_k=top_k)
    
    def search_kg_entities(self, query: str, top_k: int = 5) -> List[Dict]: #entity
        return self.search_by_type(query, "knowledge_graph", "entity", top_k)
    
    def search_kg_relationships(self, query: str, top_k: int = 3) -> List[Dict]: #relation
        return self.search_by_type(query, "knowledge_graph", "relationship", top_k)
    
    def get_enhanced_subtask_context(self, task: str, position: str) -> str:
        try:
            blog_results = self.search_blog_content(f"{task} {position}", top_k=3)
            
            entity_results = self.search_kg_entities(task, top_k=5)
            
            rel_results = self.search_kg_relationships(task, top_k=3)
                
            context_parts = []
            
            if blog_results:
                context_parts.append("=== 실무 경험 및 사례 ===")
                for result in blog_results:
                    meta = result['metadata']
                    tech = meta.get('tech_name', '')
                    title = meta.get('title', '')
                    content = meta.get('text', '')[:300]
                    context_parts.append(f"[{tech}] {title}: {content}...")
            
            if entity_results:
                context_parts.append("\n=== 관련 기술 및 개념 ===")
                for result in entity_results:
                    meta = result['metadata']
                    entity_name = meta.get('entity_name', '')
                    entity_type = meta.get('entity_type', '')
                    context_parts.append(f"- {entity_name} ({entity_type})")
            
            if rel_results:
                context_parts.append("\n=== 기술 간 관계 ===")
                for result in rel_results:
                    meta = result['metadata']
                    source = meta.get('source_entity', '')
                    target = meta.get('target_entity', '')
                    rel_type = meta.get('relationship_type', '').replace('_', ' ').lower()
                    context_parts.append(f"- {source} {rel_type} {target}")
            
            final_context = "\n".join(context_parts)
            
            if not final_context.strip():
                return f"{task}에 대한 {position} 관점에서의 구체적인 접근이 필요합니다."
            
            return final_context
            
        except Exception as e:
            logger.error(f"서브태스크 컨텍스트 생성 오류: {e}")
            return f"{task}에 대한 {position} 관점에서의 접근이 필요합니다."
