import os
import json
import re
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm.wiki_retriever import retrieve_wiki_context
from llm.json_fixer import JsonFixer
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeetingTaskParser:
    def __init__(self):
        logger.info("Initializing MeetingTaskParser...")
        load_dotenv()
        token = os.getenv("HUGGINGFACE_API_KEY")
        if not token:
            logger.warning("HUGGINGFACE_API_KEY not found in environment variables!")

        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        logger.info(f"Using model: {model_name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token
        ).to(self.device)
        logger.info("Model loaded successfully")

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token
        )
        logger.info("Tokenizer loaded successfully")

        self.json_fixer=JsonFixer()

    def generate_response(self, chat: list) -> str:
        inputs = self.tokenizer.apply_chat_template(
            chat,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id
        )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def clean_json_codeblock(self, text: str) -> str:
        cleaned = re.sub(r"```json|```", "", text).strip()
        if cleaned.count("{") > cleaned.count("}"):
            cleaned += "}"
        return cleaned

    def summarize_and_generate_tasks(self, meeting_note: str, project_id: int, position: str):
        logger.info(f"Processing meeting note for project_id: {project_id}, position: {position}")

        # Step 1: 포지션에 맞는 핵심 업무 요약
        system_prompt = {
            "role": "system",
            "content": (
                f"너는 회의록에서 '{position}' 파트의 전체 업무 내용을 정리하고, 주요 태스크를 요약하는 전문가야. "
                "입력된 회의록은 표 형식이 아닌 자유로운 자연어 문장으로 구성되어 있어. "
                "출력은 절대로 설명 없이, ','로 구분된 핵심 업무 키워드 목록 한 줄만 생성해. "
                "다른 포지션의 내용은 무시하고, '{position}' 파트의 실질적인 작업만 추출해줘."
            )
        }

        user_prompt = {
            "role": "user",
            "content": f"""
회의록:
{meeting_note}

'{position}' 파트에서 오늘 해야 할 핵심 작업만 요약해줘.  
다른 파트는 무시해도 돼.

**출력은 오직 콤마(,)로 구분된 한 줄 요약만 해줘.**
"""
        }

        summary = self.generate_response([system_prompt, user_prompt])
        task_candidates = summary.split(',')
        logger.info(f"Extracted {len(task_candidates)} task candidates")

        # Step 2: 각 업무 키워드에 대해 세부 작업 분해
        parsed_results = []
        for task in task_candidates:
            task = task.strip()
            if not task:
                continue

            logger.info(f"Processing task: {task}")
            wiki_context = retrieve_wiki_context(task, project_id)
            logger.info(f"Retrieved wiki for Project ID {project_id}")

            langchain_definition = """
LangChain은 LLM을 기반으로 LLM, Chain 등의 모듈을 연결하여
사용자 질의에 대해 복합적인 처리를 수행할 수 있도록 돕는 프레임워크입니다.
주로 체인 구성, 프롬프트 설정, 응답 처리, Tool 연동 등의 작업이 필요합니다.
"""

            task_chat = [
                {
                    "role": "system",
                    "content": f"""
Wiki Context: {wiki_context}

'{position}' 포지션의 작업 '{task}'를 2~4개의 구체적인 세부 작업(subtasks)으로 나눠줘.

조건:
- "task" 항목은 반드시 "{task}"를 그대로 사용
- "subtasks"는 동사+명사 위주 간단 표현 (예: "API 설계", "DB 스키마 작성")
- 출력은 JSON 하나로만, 설명 없이 아래와 같이

출력 예시:
{{
"task": "{task}",
"subtasks": [
    "세부 작업 1",
    "세부 작업 2",
    "세부 작업 3"
]
}}
"""
                }
            ]

            response = self.generate_response(task_chat)
            logger.info(f"Raw model response: {response}")

            try:
                # 1차 모델 응답 파싱 시도
                parsed = json.loads(self.clean_json_codeblock(response))
                parsed_results.append({
                    "keyword": parsed["task"],
                    "subtasks": parsed.get("subtasks", []),
                    "position" : position
                })

            except Exception as e:
                logger.error(f"[Parse Fail] task '{task}' → {e}")
                logger.error(f"Raw response: {response}")

                try:
                    # JSON 파서 실패 시 → JsonFixer로 보정 시도
                    fixed_response = self.json_fixer.fix_json(response)
                    logger.info(f"[Fix Attempted] Fixed response: {fixed_response}")

                    # 고친 응답 다시 파싱
                    parsed = json.loads(self.clean_json_codeblock(fixed_response))
                    parsed_results.append({
                        "keyword": parsed["task"],
                        "subtasks": parsed.get("subtasks", [])
                    })

                except Exception as fix_err:
                    logger.error(f"[Fix Fail] task '{task}' → {fix_err}")
                    logger.error(f"Fixed response: {fixed_response if 'fixed_response' in locals() else 'N/A'}")
                    continue

        return {"message": "subtasks_created", "detail": parsed_results}


