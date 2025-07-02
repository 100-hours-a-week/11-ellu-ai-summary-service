import httpx
import logging
from typing import Dict
from models.wiki.wiki_fetcher import WikiFetcher

logger = logging.getLogger(__name__)

class DocFetcher(WikiFetcher):
    def __init__(self, project_id: int, url: str):
        super().__init__(project_id, None)
        self.web_url = url
        self.s3_prefix = f"uploads/web/{project_id}/"
    
    async def get_diff_files(self) -> Dict[str, str]:
        try:
            jina_url = f"https://r.jina.ai/{self.web_url}"
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(jina_url, headers={
                    'Accept': 'text/plain; charset=utf-8',
                    'Accept-Charset': 'utf-8'
                })                
                if response.status_code != 200:
                    raise Exception(f"Jina Reader 오류: {response.status_code}")
                
                try:
                    content = response.content.decode('utf-8')
                except UnicodeDecodeError:
                    content = response.content.decode('utf-8', errors='ignore')
                
                if not content:
                    raise Exception("콘텐츠를 찾을 수 없습니다")
                
                filename = f"webpage_{self.project_id}.md"
                await self._save_content_to_s3(filename, content)
                
                logger.info(f"웹 크롤링 + S3 저장 성공 - URL: {self.web_url}")
                
                return {filename: content}
                
        except Exception as e:
            logger.error(f"웹 크롤링 실패 - URL: {self.web_url}, 오류: {e}")
            return {}
    
    async def _save_content_to_s3(self, filename: str, content: str):
        self._clear_s3_folder()
        
        s3_key = f"{self.s3_prefix}{filename}"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=content.encode('utf-8'),
            ContentType='text/markdown'
        )
        
        logger.info(f"S3 업로드 성공: {s3_key}")
