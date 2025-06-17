import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


import logging
from  app.config import  OPENAI_API_KEY,GPT_MODEL
from prompts.parsing_prompt import Json_Parsing_Prompts 
from utils.valid import valid_json

logger = logging.getLogger(__name__)

class JsonFixer:
    def __init__(self, model_name=GPT_MODEL):
        load_dotenv()
        
        self.llm = ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=0)
        self.valid = valid_json()
        self.prompt=Json_Parsing_Prompts()
        
    def fix_json(self, raw_text: str) -> dict:
        """메인 태스크 JSON 수정"""
        prompts = self.prompt.get_main_json_prompt(raw_text)
        system_prompt =prompts[0]
        user_prompt =prompts[1]

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            validated_result = self.valid.validate_main_task_json(response)
            logger.info("메인 태스크 JSON 수정 완료")
            return validated_result
            
        except Exception as e:
            logger.error(f"json_fixer 파일에서 main task JSON 수정 실패: {str(e)}")
            # 기본 구조 반환
            return {"AI": [], "BE": [], "FE": [], "CL": []}

    def fix_subtask_json(self, raw_text, position, tasks) -> list:
        """서브 태스크 JSON 수정"""
        prompts=self.prompt.get_general_json_prompt(raw_text, position, tasks)

        system_prompt = prompts[0]
        user_prompt =prompts[1]

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)

            
            # 결과 검증 및 보완
            validated_result = self.valid.validate_subtask_json(response)
            logger.info("서브 태스크 JSON 수정 완료")
            return validated_result
            
        except Exception as e:
            logger.error(f"json_fixer 파일에서 서브태스크 JSON 수정 실패: {str(e)}")
            # 기본 구조 반환
            return []



    # def fix_general_json(self, raw_text: str) -> dict:
    #     """일반적인 JSON 수정 (기존 fix_json과 호환)"""
    #     return self.fix_json(raw_text)