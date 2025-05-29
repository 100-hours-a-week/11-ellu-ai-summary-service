import os
import json
import re
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm.wiki_retriever import retrieve_wiki_context
from llm.json_fixer import JsonFixer
import torch
from typing import TypedDict


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskState(TypedDict):
    meeting_note: str
    project_id: int
    position: list[str]
    prompt: dict        
    main_task: dict | None      
    AI: dict | None 
    BE: dict | None 
    FE: dict | None 
    CL: dict | None    


# ────────────────────────────────────────────────────────
# 메인 클래스
# ────────────────────────────────────────────────────────
class MeetingTaskParser:
    def __init__(self):
        logger.info("Initializing MeetingTaskParser...")
        load_dotenv()
        self._token = os.getenv("HUGGINGFACE_API_KEY")
        if not self._token:
            logger.warning("HUGGINGFACE_API_KEY not found in environment variables!")

        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=self._token).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self._token)
        self.json_fixer = JsonFixer()

    # ────────────────────────────────────────────────────────
    # 공통 모델 실행 및 파싱 함수
    # ────────────────────────────────────────────────────────
    def run_model_and_parse(self, chat: list) -> list[dict]:
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
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        try:
            cleaned = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = self.json_fixer.fix_json(raw)
        return parsed

    # ────────────────────────────────────────────────────────
    # 회의록에서 핵심 태스크 추출
    # ────────────────────────────────────────────────────────
    def extract_core_tasks(self, state: TaskState) -> dict:
        meeting_note = state["meeting_note"]



        system_prompt = {
            "role": "system",
            "content": """
너는 팀 회의록에서 포지션별 할 일을 뽑아주는 전문가야.

- 포지션 정의:
• AI: 인공지능 개발 파트
• BE: 백엔드 개발 파트
• FE: 프론트엔드 개발 파트
• CL: 클라우드 개발 파트

- 회의록에 나온 문구를 그대로 키워드로 사용하고, 의역 금지야.
- 출력은 반드시 아래 JSON 템플릿 형식과 키를 **모두** 포함해야 해.
- 각 포지션 키는 무조건 출력에 포함되며, 할 일이 없으면 빈 배열([])로 채워야 해.
- 출력 키워드는 반드시 짧고 간결한 **명사구** 또는 **동사+명사** 형태여야 해.
- 조사·종결어미, 마침표(.)는 절대 쓰지 마.

템플릿:
{
  "AI": [],
  "BE": [],
  "FE": [],
  "CL": []
}
"""
        }
        user_prompt = {
            "role": "user",
            "content": f"""
회의록:
'{meeting_note}'



목표: 각 포지션 별로 오늘 할 일을 식별해서 JSON 템플릿에 맞게 작성해줘.
"""
        }
        return {'prompt': {'main_task': [system_prompt, user_prompt]}}

    # ────────────────────────────────────────────────────────
    # 프롬프트 기반 응답 생성
    # ────────────────────────────────────────────────────────
    def generate_response(self, state: TaskState) -> dict:
        chat = state['prompt']['main_task']
        parsed = self.run_model_and_parse(chat)
        return {'main_task': parsed}

    # ────────────────────────────────────────────────────────
    # 다음 분기 노드 결정
    # ────────────────────────────────────────────────────────
    def route_to_subtasks(self, state: TaskState) -> list[str]:
        mapping = {
            "ai": "generate_AI_subtasks",
            "be": "generate_BE_subtasks",
            "fe": "generate_FE_subtasks",
            "cl": "generate_Cloud_subtasks",
        }
        return [mapping[p.lower()] for p in state['position'] if p.lower() in mapping]

    # ────────────────────────────────────────────────────────
    # 포지션별 세부 태스크 생성
    # ────────────────────────────────────────────────────────
    def generate_position_response(self, state: TaskState, key: str) -> dict:
        tasks = state['main_task'][key]
        chat = [{
            "role": "system",
            "content": f"""
다음 지침에 따라 **{key} 포지션 작업 목록**을 세부 작업으로 분해하라.

🔹 입력
- `tasks`는 여러 개의 작업을 담은 배열이다.

🔹 출력 예시

            [
            {{  "position": "{key}",
                "task": "<원본 작업>",
                "subtasks": ["세부 1", "세부 2"]
            }}
            ]


🔹 규칙
- "task"는 입력값 그대로
- "subtasks"는 2~4개, 동사+명사 형태로 작성
- 마침표, 설명, 코드블럭 없이 JSON 배열 하나만 출력
입력 작업 목록:
{tasks}
"""
        }]
        parsed = self.run_model_and_parse(chat)
        return {key: parsed}

    def generate_AI_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "AI")

    def generate_BE_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "BE")

    def generate_FE_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "FE")

    def generate_Cloud_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "CL")
