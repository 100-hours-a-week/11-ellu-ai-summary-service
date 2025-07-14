import os
import json
import re
import logging
import httpx


from models.llm.json_fixer import JsonFixer
from utils.valid import valid_json

from prompts.prompt import MeetingPromptManager
from app.config import VLLM_URL, TEMPERATURE, MODEL_KWARGS
# ────────────────────────────────────────────────────────
# 설정 및 로깅
# ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# 메인 클래스 정의
# ────────────────────────────────────────────────────────
class Generate_llm_response:
    def __init__(self):
        logger.info("Initializing Generate_llm_response...")
        
        self.prompt = MeetingPromptManager()
        self.json_fixer = JsonFixer()
        self.valid=valid_json()
        self.vllm_url = VLLM_URL
        
    def parse_response(self, content: str, where: str, task: str = None, position: str = None) -> dict:
        """응답 파싱"""
        try:
            cleaned = re.sub(r"```json|```", "", content).strip()
            parsed = json.loads(cleaned)
            
            if where == "main":
                return parsed
            else:  # sub task 처리
                # 다양한 형태의 응답 처리
                if isinstance(parsed, list):
                    # 이미 리스트인 경우
                    return {"세부 단계": parsed}
                elif isinstance(parsed, dict):
                    # 딕셔너리인 경우
                    if "세부 단계" in parsed:
                        return parsed
                    else:
                        # 키-값 쌍을 값들만 추출하여 리스트로 변환
                        values = list(parsed.values())
                        return {"세부 단계": values}
                else:
                    return {"세부 단계": []}
                    
        except json.JSONDecodeError as e:
            if where == "main":
                logger.error(f"Main JSON 파싱 오류: {str(e)}")
                return self.json_fixer.fix_json(content)
            else:
                logger.error(f"Sub JSON 파싱 오류: {str(e)}")
                return self.json_fixer.fix_subtask_json(content, position, task)
        except Exception as e:
            logger.error(f"응답 파싱 중 오류: {str(e)}")
            return {"error": str(e)}
    # ────────────────────────────────────────────────────────
    # 모델 실행 및 JSON 파싱
    # ────────────────────────────────────────────────────────
    async def run_model_and_parse(self, chat: list,where:str,task: str=None, position :str=None) -> dict:
        """LLM 모델 실행 및 결과 파싱"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    json={
                        "model": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
                        "messages": chat,
                        "temperature": TEMPERATURE,
                        **MODEL_KWARGS
                    },
                    timeout=None
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"vllm 생성한 content : {content}")

                return self.parse_response(content, where, task, position)


        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM 서버 요청 실패: {e.response.status_code}")
            logger.error(f"vLLM 서버 응답: {e.response.text}")
            return {"error": f"Failed to request vLLM server: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"모델 실행 중 오류 발생: {str(e)}")
            return {"error": str(e)}



   

 







