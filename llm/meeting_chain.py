import os
import json
import re
import logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Literal, TypedDict
import concurrent.futures
from wiki.wiki_retriever import WikiRetriever
from llm.json_fixer import JsonFixer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ────────────────────────────────────────────────────────
# 설정 및 로깅
# ────────────────────────────────────────────────────────
torch.set_printoptions(profile="full")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# 상태 정의 (입출력 관리용)
# ────────────────────────────────────────────────────────
class TaskState(TypedDict):
    meeting_note: str
    project_id: int
    position: list[str]
    prompt: dict
    main_task: dict
    AI: list | None
    BE: list | None
    FE: list | None
    CL: list | None
    validation_result: str
    feedback: str | None


# ────────────────────────────────────────────────────────
# 메인 클래스 정의
# ────────────────────────────────────────────────────────
class MeetingTaskParser:
    def __init__(self):
        logger.info("Initializing MeetingTaskParser...")
        load_dotenv()
        self._token = os.getenv("HUGGINGFACE_API_KEY")
        if not self._token:
            logger.warning("HUGGINGFACE_API_KEY not found in environment variables!")
        # model_name = "mistralai/Ministral-8B-Instruct-2410"
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=self._token).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self._token)
        self.json_fixer = JsonFixer()
        self.wiki_retriever = WikiRetriever()

    # ────────────────────────────────────────────────────────
    # 모델 실행 및 JSON 파싱
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
    # 회의록에서 핵심 업무 태스크 추출
    # ────────────────────────────────────────────────────────
    def extract_core_tasks(self, state: TaskState) -> dict:
        meeting_note = state["meeting_note"]
        
        system_prompt = {
            "role": "system",
            "content": """
    당신은 팀 회의록에서 포지션별 할 일을 정확히 분류하는 전문가입니다.

    🔹 **포지션 정의:**
    • AI: 인공지능, 머신러닝, 자연어처리, 데이터 분석, AI 알고리즘 관련 업무
    • BE: 백엔드, 서버, API, 데이터베이스, 비즈니스 로직, 시스템 아키텍처 관련 업무
    • FE: 프론트엔드, UI/UX, 웹/앱 화면, 사용자 인터페이스, 클라이언트 사이드 관련 업무
    • CL: 클라우드, 인프라, 배포, 모니터링, DevOps, 서버 관리 관련 업무

    🔹 **핵심 규칙:**
    1. **원문 충실성**: 회의록에 나온 표현을 그대로 사용하며, 의역하지 않습니다
    2. **완전성**: 회의록에 언급된 모든 작업을 누락 없이 분류합니다
    3. **정확성**: 각 작업을 가장 적절한 포지션에 분류합니다
    4. **일관성**: 모든 포지션 키(AI, BE, FE, CL)를 반드시 포함합니다

    🔹 **작업명 형식:**
    • 짧고 간결한 **명사구** 또는 **동사+명사** 형태
    • 예시: "사용자 인증 구현", "데이터베이스 최적화", "메인 페이지 개발"
    • 금지: 조사(을/를/이/가), 종결어미(다/요/함), 마침표(.)

    🔹 **포지션별 할당 예시:**
    • "AI 모델 학습" → AI 포지션
    • "로그인 API 개발" → BE 포지션  
    • "대시보드 UI 개선" → FE 포지션
    • "서버 배포 자동화" → CL 포지션

    🔹 **빈 포지션 처리:**
    해당 포지션에 할 일이 없으면 반드시 빈 배열([])로 설정합니다.

    🔹 **출력 형식 (반드시 준수):**
    {
      "AI": ["작업1", "작업2", ...] 또는 [],
      "BE": ["작업1", "작업2", ...] 또는 [],
      "FE": ["작업1", "작업2", ...] 또는 [],
      "CL": ["작업1", "작업2", ...] 또는 []
    }

    **중요**: 4개 포지션 키를 모두 포함하고, 해당 없는 포지션은 빈 배열([])로 설정하세요.
            """
        }
        
        user_prompt = {
            "role": "user", 
            "content": f"""
    다음 회의록을 분석하여 포지션별 할 일을 정확히 분류해주세요.

    **회의록:**
    {meeting_note}

    **분석 지침:**
    1. 회의록에서 언급된 모든 작업을 식별하세요
    2. 각 작업을 가장 적절한 포지션(AI/BE/FE/CL)에 분류하세요
    3. 회의록 원문의 표현을 최대한 그대로 사용하세요
    4. 해당 포지션에 작업이 없으면 빈 배열([])로 설정하세요
    5. 반드시 모든 포지션 키(AI, BE, FE, CL)를 포함하세요

    JSON 형식으로 출력해주세요:
            """
        }
        
        return {'prompt': {'main_task': [system_prompt, user_prompt]}}

    # ────────────────────────────────────────────────────────
    # 핵심 태스크를 바탕으로 JSON 생성
    # ────────────────────────────────────────────────────────
    def generate_response(self, state: TaskState) -> dict:
        chat = state['prompt']['main_task']
        parsed = self.run_model_and_parse(chat)
        return {'main_task': parsed}

    # ────────────────────────────────────────────────────────
    # 포지션에 따라 다음 노드 결정
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

        # 빈 작업 리스트 처리
        if not tasks or tasks == []:
            return {key: []}
        
        outputs = []
        for task in tasks:
            # task별로 wiki 검색
            context_dict = self.wiki_retriever.retrieve_wiki_context(task, state['project_id'])
            wiki_context = context_dict[task]
            # wiki_context =""

            # 세부 작업 분해
            chat = [
                {
                    "role": "system",
                    "content": f"""
    당신은 숙련된 {key} 포지션의 시니어 엔지니어입니다.

    🔹 **포지션별 전문 영역:**
    • AI: 머신러닝 모델링, 데이터 파이프라인, 알고리즘 최적화
    • BE: 서버 아키텍처, API 설계, 데이터베이스 관리  
    • FE: 사용자 인터페이스, 컴포넌트 설계, 사용자 경험
    • CL: 인프라 구축, 배포 자동화, 시스템 모니터링

    🔹 **세부 작업 분해 원칙:**
    1. **실행 가능성**: 실제로 수행할 수 있는 구체적인 단계
    2. **순서성**: 논리적 순서를 고려한 작업 배치
    3. **완성도**: 메인 작업을 완료하기 위한 필수 단계들
    4. **현실성**: 실제 개발 환경에서 수행되는 업무 수준

    🔹 **세부 작업 작성 규칙:**
    • 동사로 시작하는 명확한 액션 아이템
    • {wiki_context}를 참고해서 작성
    • 2-5개의 적절한 단계로 분해
    • 너무 세분화하지 않되, 충분히 구체적으로
    • 각 단계는 독립적으로 수행 가능해야 함

    🔹 **출력 형식:**
    [
      {{
        "position": "{key}",
        "task": "<원본 작업명 그대로>",
        "subtasks": [
          "구체적인 세부 작업 1",
          "구체적인 세부 작업 2", 
          "구체적인 세부 작업 3"
        ]
      }}
    ]

    **중요**: 원본 작업명은 절대 변경하지 말고, 세부 작업만 새로 생성하세요.
                """
            },
            {
                "role": "user",
                "content": f"""
    다음 {key} 포지션의 작업 목록을 세부 단계로 분해해주세요.

    **작업 목록:**
    {tasks}

    **요구사항:**
    1. 각 작업을 실제 개발 프로세스에 맞는 세부 단계로 분해
    2. 원본 작업명은 그대로 유지
    3. 세부 작업은 구체적이고 실행 가능하게 작성
    4. JSON 배열 형식으로 출력

    세부 작업으로 분해해주세요:
                """
            }
        ]
        
            parsed = self.run_model_and_parse(chat)
            outputs.extend(parsed)
        return {key: parsed}
    # ────────────────────────────────────────────────────────
    # LLM 평가 기반 품질 판단 → retry 여부 판단
    # ────────────────────────────────────────────────────────
    def judge_quality_with_json_mode(self, state: TaskState) -> dict:
        """JSON 모드를 사용한 더 간단한 접근법"""
        meeting_note = state["meeting_note"]
        main_task = state["main_task"]

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

        messages = [
            SystemMessage(content="""
   당신은 회의록 분석 결과의 품질을 평가하는 전문 심사관입니다.

    평가 후 반드시 다음 JSON 형식으로만 응답하세요:

    ```json
    {
      "result": "pass" 또는 "fail",
      "failure_reasons": ["실패 이유1", "실패 이유2"],
      "improvement_suggestions": ["개선 방법1", "개선 방법2"]
    }
    ```

    **평가 체크리스트:**

    1. 올바른 포지션 분류 여부
    2. 원문 표현 왜곡 여부


    모든 항목 만족 시 "pass", 하나라도 미달 시 "fail"로 판정하세요.

    fail일 때 개선 방법을 어떤 부분을 어떻게 수정하고 어떤 부분을 어디로 옮겨야하는지 구체적으로 설명해주세요.
            """),
            HumanMessage(content=f"""
    **회의록:**
    {meeting_note}

    **분류 결과:**
    {json.dumps(main_task, ensure_ascii=False, indent=2)}

    위 결과를 평가하고 JSON 형식으로 응답해주세요.
            """)
        ]

        try:
            response = llm.invoke(messages)
            evaluation_data = json.loads(response.content)
            
            # 피드백 문자열 생성
            if evaluation_data["result"] == "pass":
                feedback = ""
            else:
                feedback_parts = []
                if evaluation_data.get("failure_reasons"):
                    feedback_parts.append("실패사유:")
                    for reason in evaluation_data["failure_reasons"]:
                        feedback_parts.append(f"- {reason}")
                
                if evaluation_data.get("improvement_suggestions"):
                    feedback_parts.append("\n개선방향:")
                    for suggestion in evaluation_data["improvement_suggestions"]:
                        feedback_parts.append(f"- {suggestion}")
                
                feedback = "\n".join(feedback_parts)

            return {
                "validation_result": evaluation_data["result"],
                "feedback": feedback
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return {
                "validation_result": "fail",
                "feedback": "평가 결과 파싱 중 오류가 발생했습니다. 전반적인 검토가 필요합니다."
            }
        except Exception as e:
            logger.error(f"judge_quality_with_json_mode 오류: {e}")
            return {
                "validation_result": "fail",
                "feedback": "평가 중 오류가 발생했습니다. 전반적인 검토가 필요합니다."
            }

    # ────────────────────────────────────────────────────────
    # 세부 태스크 병렬 생성 (AI, BE, FE, CL)
    # ────────────────────────────────────────────────────────
    def generate_all_position_responses(self, state: TaskState) -> dict:
        positions = state["position"]
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_key = {
                executor.submit(self.generate_position_response, state, pos): pos
                for pos in positions
            }
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()[key]
                except Exception as exc:
                    logger.warning(f"{key} 처리 중 오류 발생: {exc}")
                    results[key] = None
        return results

    # ────────────────────────────────────────────────────────
    # judge 결과에 따라 분기: retry or 포지션 노드
    # ────────────────────────────────────────────────────────
    def route_after_validation(self, state: TaskState) -> list[str]:
        # if state['validation_result'] == 'fail':
        #     return ["retry_node"]
        # else:
            return self.route_to_subtasks(state)

    # ────────────────────────────────────────────────────────
    # retry_node: JSON 결과 재작성
    # ────────────────────────────────────────────────────────
    def retry(self, state: TaskState) -> dict:
        meeting_note = state["meeting_note"]
        main_task = state["main_task"]
        feedback = state["feedback"]
        
        system_prompt = {
            "role": "system",
            "content": """
    당신은 팀 회의록 분석 전문가로서, 품질 검증에서 실패한 JSON 결과를 개선해야 합니다.

    🔹 **포지션별 업무 정의:**
    • AI: 머신러닝 모델, 자연어처리, 데이터 분석, AI 알고리즘 관련
    • BE: 서버, API, 데이터베이스, 비즈니스 로직, 시스템 아키텍처 관련  
    • FE: 사용자 인터페이스, 웹/앱 화면, UX/UI, 클라이언트 사이드 관련
    • CL: 인프라, 배포, 모니터링, 클라우드 서비스, DevOps 관련


    🔹 **출력 형식:**
    {
      "AI": ["작업1", "작업2", ...],
      "BE": ["작업1", "작업2", ...], 
      "FE": ["작업1", "작업2", ...],
      "CL": ["작업1", "작업2", ...]
    }

    **중요:** 아래 피드백을 반드시 반영하여 개선된 결과를 작성하세요.
    """
        }
        
        user_prompt = {
            "role": "user",
            "content": f"""
    **회의록:**
    {meeting_note}

    **이전 결과 (실패):**
    {json.dumps(main_task, ensure_ascii=False, indent=2)}

    **🚨 품질 검증 피드백:**
    {feedback}

    **지시사항:**
    위 피드백에서 지적된 문제점들을 정확히 파악하고, 개선방향에 따라 완벽한 JSON을 다시 작성해주세요.

    특히 다음 사항을 주의깊게 확인하세요:
    - 누락된 작업이 있다면 반드시 추가
    - 잘못 분류된 작업이 있다면 올바른 포지션으로 이동
    - 모호한 표현이 있다면 구체적으로 수정
    - 회의록 원문의 표현을 최대한 유지

    개선된 JSON을 작성해주세요:
    """
        }
        
        return {'prompt': {'main_task': [system_prompt, user_prompt]}}

    # ────────────────────────────────────────────────────────
    # 포지션별 세부 태스크 분기용 wrapper
    # ────────────────────────────────────────────────────────
    def generate_AI_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "AI")

    def generate_BE_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "BE")

    def generate_FE_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "FE")

    def generate_Cloud_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "CL")
