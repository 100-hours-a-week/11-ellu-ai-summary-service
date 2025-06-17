import os
import json
import re
import logging
import torch


from transformers import AutoModelForCausalLM, AutoTokenizer
from models.llm.json_fixer import JsonFixer

from prompts.prompt import MeetingPromptManager
from langchain_openai import ChatOpenAI
from app.config import MODEL_NAME, Hugging_FACE_KEY, GPT_MODEL, TEMPERATURE, MODEL_KWARGS
# ────────────────────────────────────────────────────────
# 설정 및 로깅
# ────────────────────────────────────────────────────────
torch.set_printoptions(profile="full")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# 메인 클래스 정의
# ────────────────────────────────────────────────────────
class Generate_llm_response:
    def __init__(self):
        logger.info("Initializing Generate_llm_response...")
        
        self._token = Hugging_FACE_KEY
        if not self._token:
            logger.warning("HUGGINGFACE_API_KEY not found in environment variables!")
        # model_name = "mistralai/Ministral-8B-Instruct-2410"
        self.prompt = MeetingPromptManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=self._token).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=self._token)
        self.json_fixer = JsonFixer()
        self.llm = ChatOpenAI(
            model=GPT_MODEL,
            temperature=TEMPERATURE,
            model_kwargs=MODEL_KWARGS
        )
        

    # ────────────────────────────────────────────────────────
    # 모델 실행 및 JSON 파싱
    # ────────────────────────────────────────────────────────
    def run_model_and_parse(self, chat: list,where:str) -> dict:
        """LLM 모델 실행 및 결과 파싱"""
        try:
            # LLM 모델 실행
            response = self.llm.invoke(chat)
            
            # 응답 파싱
            return self.parse_response(response.content,where)
        except Exception as e:
            logger.error(f"모델 실행 중 오류 발생: {str(e)}")
            return {"error": str(e)}

    def parse_response(self, content: str,where:str) -> dict:
        """응답 파싱"""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            if where =="main":
                logger.error(f"Main JSON 파싱 오류: {str(e)}")
                return self.json_fixer.fix_json(content)
            else :
                logger.error(f"Sub JSON 파싱 오류: {str(e)}")
                return self.json_fixer.fix_subtask_json(content)                
        except Exception as e:
            logger.error(f"응답 파싱 중 오류: {str(e)}")
            return {"error": str(e)}

   

 







