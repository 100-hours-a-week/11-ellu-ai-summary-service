import httpx
import logging
from typing import Dict
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

class JinaProcessor:    
    def __init__(self, project_id: int, url: str):
        self.project_id = project_id
        self.url = url
        
    async def get_diff_files(self) -> Dict[str, str]:
        try:
            jina_url = f"https://r.jina.ai/{self.url}"
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(jina_url)
                
                if response.status_code != 200:
                    raise Exception(f"Jina Reader 오류: {response.status_code}")
                
                content = response.text.strip()
                if not content:
                    raise Exception("콘텐츠를 찾을 수 없습니다")
                
                logger.info(f"Jina Reader 성공 - URL: {self.url}, 길이: {len(content)}자")
                
                filename = f"webpage_{self.project_id}.md"
                return {filename: content}
                
        except Exception as e:
            logger.error(f"Jina Reader 실패 - URL: {self.url}, 오류: {e}")
            return {}
    
    def delete_project_data(self):
        logger.info(f"단일 페이지 데이터는 별도 삭제 불가  - project_id: {self.project_id}")
        return True