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

    def get_position_definition(self, position: str) -> str:
        try:
            logger.info(f"포지션 정의 검색 시작: {position}")
            
            job_desc_results = self.search_by_type(f"{position} role definition", "job_description", top_k=5)
            logger.info(f"job_description 검색 결과: {len(job_desc_results)}개")
            
            for result in job_desc_results:
                meta = result['metadata']
                position_name = meta.get('position', '').upper()
                content = meta.get('content', '') 
                
                logger.info(f"검색된 포지션: {position_name}, 내용 길이: {len(content)}")
                
                if position_name == position.upper() and len(content) > 100:
                    logger.info(f"매칭된 역할 정의 발견: {position}")
                    return content
            
            logger.info(f"job_description에서 찾지 못함, KG 검색 시도: {position}")
            
            # 해당 포지션과 관련된 엔티티 검색
            entities = self.search_kg_entities(f"{position} engineer role", top_k=10)
            
            # 포지션과 관련된 기술 검색
            tech_query = f"{position} technology stack"
            tech_results = self.search_kg_entities(tech_query, top_k=10)
            
            role_parts = [f"**{position} Engineer 역할:**"]
            
            technologies = []
            for result in entities + tech_results:
                meta = result['metadata']
                entity_name = meta.get('entity_name', '')
                entity_type = meta.get('entity_type', '')
                
                if entity_type == 'Technology' and result['score'] > 0.7:
                    technologies.append(entity_name)
            
            if technologies:
                unique_techs = list(set(technologies))[:8]  # 중복 제거, 상위 8개
                role_parts.append(f"- 주요 기술: {', '.join(unique_techs)}")
            
            tasks = []
            for result in entities:
                meta = result['metadata']
                entity_name = meta.get('entity_name', '')
                entity_type = meta.get('entity_type', '')
                
                if entity_type in ['Method', 'Concept'] and result['score'] > 0.6:
                    tasks.append(entity_name)
            
            if tasks:
                unique_tasks = list(set(tasks))[:5]
                role_parts.append(f"- 주요 업무: {', '.join(unique_tasks)}")
            
            kg_result = "\n".join(role_parts)
            logger.info(f"KG 기반 역할 정의 생성: {len(kg_result)} 문자")
            return kg_result
            
        except Exception as e:
            logger.error(f"포지션 정의 검색 오류 ({position}): {e}")
            logger.info(f"검색 오류, 기본값 사용: {position}")
            # 기본 정의 반환
            default_roles = {
                "AI": "AI/ML 모델 개발, 데이터 분석, 머신러닝 파이프라인 구축",
                "BE": "서버 개발, API 설계, 데이터베이스 관리, 비즈니스 로직 구현",
                "FE": "사용자 인터페이스 개발, 웹/앱 화면 구현, 사용자 경험 개선",
                "CLOUD": "인프라 구축, 배포 자동화, 모니터링, DevOps"
            }
            return f"**{position} Engineer 역할:** {default_roles.get(position, '개발 업무')}"
