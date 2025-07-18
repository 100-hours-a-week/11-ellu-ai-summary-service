import json
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage


class MeetingPromptManager:
    """회의록 분석을 위한 프롬프트 관리 클래스"""
    
    def __init__(self, wiki_context: str = ""):
        """
        Args:
            wiki_context: 세부 작업 생성 시 참고할 위키 문서나 컨텍스트
        """
        
    
    def get_main_prompt(self) -> Dict[str, str]:
        """메인 작업 분류를 위한 시스템 프롬프트 반환"""
        return {
            "role": "system",
            "content": """
    당신은 팀 회의록에서 포지션별 할 일을 정확히 분류하는 전문가입니다.

    🔹 **포지션 정의:**
    • AI: 인공지능, 머신러닝, 자연어처리, 데이터 분석, AI 알고리즘 관련 업무
    • BE: 백엔드, 서버, API, 데이터베이스, 비즈니스 로직, 시스템 아키텍처 관련 업무
    • FE: 프론트엔드, UI/UX, 웹/앱 화면, 사용자 인터페이스, 클라이언트 사이드 관련 업무
    • CLOUD: 클라우드, 인프라, 배포, 모니터링, DevOps, 서버 관리 관련 업무, CI/CD

    🔹 **핵심 규칙:**
    1. **원문 충실성**: 회의록에 나온 표현을 그대로 사용
    2. **완전성**: 회의록에 언급된 모든 작업을 누락 없이 분류합니다
    3. **정확성**: 각 작업을 가장 적절한 포지션에 분류합니다
    4. **일관성**:  할 일이 없다면 빈 배열로 나타내세요
    5. **🚫 중복 절대 금지**: 동일한 작업이 여러 포지션에 중복 할당되는 것을 절대 금지합니다. 
      - 각 작업은 반드시 하나의 포지션에만 속해야 합니다
      - 예시: "API 개발"이 BE와 CLOUD에 동시에 있으면 안됩니다
      - 중복 발견 시 가장 적합한 하나의 포지션에만 배치하세요

    🔹 **연결어 처리 규칙 (매우 중요):**
    다음 연결어로 이어진 작업들은 **반드시 개별 작업으로 분리**하세요:
    • "및", "하고", "그리고", "과", "와", "또한", "더불어", "아울러"
    • ",", ";"로 구분된 항목들
    • "A와 B 구현" → ["A 구현", "B 구현"]
    • "데이터베이스 설계 및 API 개발" → ["데이터베이스 설계", "API 개발"]
    • "UI 개선하고 성능 최적화" → ["UI 개선", "성능 최적화"]
    • "모델 학습, 평가, 배포" → ["모델 학습", "모델 평가", "모델 배포"]

    🔹 **작업명 형식:**
    • 짧고 간결한 **명사구** 또는 **동사+명사** 형태
    • 예시: "사용자 인증 구현", "데이터베이스 최적화", "메인 페이지 개발"
    • 금지: 조사(을/를/이/가), 종결어미(다/요/함), 마침표(.)
    • 연결어 제거: "및", "하고", "그리고" 등은 작업명에서 완전히 제거

    🔹 **분리 예시:**
    • "로그인 API 및 회원가입 기능 개발" 
      → ["로그인 API 개발", "회원가입 기능 개발"]
    • "데이터 전처리하고 모델 훈련 및 평가"
      → ["데이터 전처리", "모델 훈련", "모델 평가"]
    • "프론트엔드 컴포넌트 개발과 백엔드 API 연동"
      → FE: ["컴포넌트 개발"], BE: ["API 연동"]

    🔹 **포지션별 할당 예시:**
    • "랭체인" → AI 포지션
    • "랭그래프" → AI 포지션
    • "로그인 API 개발" → BE 포지션
    • "대시보드 UI 개선" → FE 포지션
    • "서버 배포 자동화" → CLOUD 포지션

     **빈 포지션 처리:**
    해당 포지션에 할 일이 없으면 반드시 빼주세요.

     **출력 형식 (반드시 준수):**
    {
      "AI": ["작업1", "작업2", ...] 또는 [],
      "BE": ["작업1", "작업2", ...] 또는 [],
      "FE": ["작업1", "작업2", ...] 또는 [],
      "CLOUD": ["작업1", "작업2", ...] 또는 []
    }

    **중요**: 

    1. 연결어로 묶인 작업들을 반드시 개별적으로 분리하여 각각 적절한 포지션에 할당하세요.
    **2. 포지션들 간의 작업명 중복은 절대 없게 하세요.**
    3. 개발적으로 할 일 외에 홍보,미팅 등은 전부 삭제해주세요
            """
        }
    


    def get_user_prompt(self, meeting_note: str) -> Dict[str, str]:
        """메인 작업 분류를 위한 사용자 프롬프트 반환"""
        return {
            "role": "user",
            "content": f"""
    다음 회의록을 분석하여 포지션별 할 일을 정확히 분류해주세요.

    **회의록:**
    {meeting_note}

    **분석 지침:**
    1. 회의록에서 언급된 모든 작업을 식별하세요
    2. **연결어(및, 하고, 그리고, 와, 과 등)로 이어진 작업들을 개별 작업으로 분리**하세요
    3. 각 작업을 가장 적절한 포지션(AI/BE/FE/CLOUD)에 분류하세요
    4. 연결어는 제거하세요
    5. 해당 포지션에 작업이 없으면 Json에서 빼주세요
    6. 전체 작업들 중 이름의 중복이 없게 해주세요
    7. 개발적으로 할 일 외에 홍보,미팅 등은 전부 삭제해주세요

    **분리 예시:**
    - "데이터베이스 설계 및 API 개발" → ["데이터베이스 설계", "API 개발"]
    - "UI 컴포넌트 개발하고 스타일링" → ["UI 컴포넌트 개발", "스타일링"]
    - "모델 훈련, 검증, 배포" → ["모델 훈련", "모델 검증", "모델 배포"]

    JSON 형식으로 출력해주세요:
            """
        }

    def get_subtask_prompts(self, position: str, tasks: str, wiki_context : str, role :str) -> List[Dict[str, str]]:
        """특정 포지션의 세부 작업 분해를 위한 프롬프트 반환"""
        return [
            {
                "role": "system",
                "content": f"""
    당신은 숙련된 {position} 포지션의 시니어 엔지니어입니다.

    🔹 **포지션별 전문 영역:**
    참고: {role}

    🔹 **세부 작업 분해 원칙:**
    1. **실행 가능성**: 실제로 수행할 수 있는 구체적인 단계
    2. **순서성**: 논리적 순서를 고려한 작업 배치
    3. **완성도**: 메인 작업을 완료하기 위한 필수 단계들
    4. **현실성**: 실제 개발 환경에서 수행되는 업무 수준

    🔹 **세부 작업 작성 규칙:**
    • 동사로 시작하는 명확한 액션 아이템
    
    • 2-5개의 적절한 단계로 분해
    •  충분히 구체적으로
    • 각 단계는 독립적으로 수행 가능해야 함
    • 구체적인 세부 작업들을 {wiki_context}에 있는 내용을 참고하여 작성 
      출력 형식을 반드시 지키세요 리스트 형식으로 답변
      다른 말은 하지 말고 

🔹 **출력 형식 (반드시 준수):**
JSON 형식에서 "세부 단계" 키에 배열 값을 반환:

{{
  "세부 단계": ["구체적인 세부 작업 1", "구체적인 세부 작업 2", "구체적인 세부 작업 3"]
}}

**중요**: 반드시 위 JSON 형식을 지켜주세요!
    
                """
            },
            {
                "role": "user",
                "content": f"""
    다음 {position} 포지션의 작업을 세부 단계로 분해해주세요.

    작업 : {tasks}

    **요구사항:**
    1. 작업을 실제 개발 프로세스에 맞는 세부 단계로 분해
    2. 세부 작업은 구체적이고 실행 가능하게 작성 (최대 5개)
    3. 반드시 JSON 형식으로 출력: {{"세부 단계": ["작업1", "작업2", ...]}}

    JSON으로 응답해주세요:
                """
            }
        ]
    def subtask_position_role(self, position: str): 
        if position == "AI":
            return """
    - **AI/ML Engineer:**
      - 데이터 수집, 전처리, 특성 엔지니어링 및 EDA 수행
      - 머신러닝/딥러닝 모델 설계, 훈련, 평가 및 하이퍼파라미터 튜닝
      - MLOps 파이프라인 구축 (모델 버전 관리, 자동화된 재훈련, 배포)
      - 모델 성능 모니터링, A/B 테스트, 드리프트 감지
      - TensorFlow, PyTorch, Scikit-learn, Kubeflow, MLflow 활용
      - 데이터 파이프라인 설계 (Apache Airflow, Spark, Kafka)
      - LLM 파인튜닝, RAG 시스템, 벡터 데이터베이스 구축
            """
        elif position == "BE":  # return 문 누락
            return """
    - **Backend Engineer:**
      - RESTful/GraphQL API 설계 및 구현
      - 마이크로서비스 아키텍처 설계 및 서비스 간 통신
      - 데이터베이스 설계, 쿼리 최적화, 인덱싱 전략
      - 캐싱 전략 (Redis, Memcached), 세션 관리
      - 인증/인가 시스템 (JWT, OAuth2, RBAC)
      - 비동기 처리 (메시지 큐, 이벤트 드리븐 아키텍처)
      - 서버 성능 튜닝, 로드 밸런싱, 장애 처리
      - Spring Boot, Express.js, Django, FastAPI 등 프레임워크 활용
            """
        elif position == "FE":  # return 문 누락
            return """
    - **Frontend Engineer:**
      - 반응형 웹 디자인 및 크로스 브라우저 호환성 구현
      - 컴포넌트 기반 아키텍처 설계 (재사용성, 확장성 고려)
      - 상태 관리 (Redux, Zustand, Context API) 및 데이터 플로우 설계
      - 웹 성능 최적화 (번들링, 코드 스플리팅, 지연 로딩)
      - 사용자 경험(UX) 개선 (애니메이션, 인터랙션, 접근성)
      - 테스트 자동화 (Jest, Cypress, Testing Library)
      - React, Vue.js, Angular, TypeScript, Webpack, Vite 활용
      - PWA, SEO 최적화, 웹 표준 준수
            """
        elif position == "CLOUD":
            return """
    - **Cloud/DevOps Engineer:**
      - 클라우드 인프라 설계 및 구축 (AWS, GCP, Azure)
      - 컨테이너화 (Docker) 및 오케스트레이션 (Kubernetes)
      - CI/CD 파이프라인 구축 (Jenkins, GitHub Actions, GitLab CI)
      - IaC (Infrastructure as Code) - Terraform, CloudFormation
      - 모니터링 및 로깅 시스템 (Prometheus, Grafana, ELK Stack)
      - 보안 정책 구현 (네트워크 보안, 시크릿 관리, 취약점 스캔)
      - 서버리스 아키텍처 (Lambda, Cloud Functions) 설계
      - 백업/복구 전략, 재해 복구 계획 수립
            """
        else:
            return " "
            
            
    def get_judge_prompts(self, meeting_note: str, main_task: Dict[str, List[str]]) -> List:
        """분류 결과 품질 평가를 위한 프롬프트 반환"""
        return [
            SystemMessage(content="""
당신은 회의록 분석 결과를 검토하는 품질 관리자입니다.

**검증 기준 (관대함):**
1. JSON 형식이 올바른가?
2. 주요 작업들이 대략적으로 적절한 포지션에 분류되었는가?
3. 명백한 오분류가 있는가?
4. 중복되는 작업이 있는가

**통과 조건:**
- JSON 형식이 올바름
- 대부분의 작업이 합리적으로 분류됨
- 심각한 오분류가 없음
- 중복되는 작업이 없어야함
                          
**실패 조건 (이것들만):**
- JSON 형식 오류
- AI 작업을 FE에 할당하는 등 명백한 오분류
- 회의록의 핵심 작업을 완전히 누락 (개발적으로 할 일 외에 홍보,미팅 등은 없도 됩니다.)
- 중복되는 작업이 존재

평가 후 다음 형식으로 응답:
                          
```json
{
  "result": "pass",
  "failure_reasons": [],
  "improvement_suggestions": []
}
```
중요: 80% 이상 괜찮으면 pass로 판정하세요.
                          
improvement_suggestions은 예시로 
                          
오류 분류시 :"AI 파트의 [작업 1] 을 삭제하고 CLOUD의 [작업 1] 추가하세요" 
중복시: " ** 파트의 [** 작업] 을 지우세요"

                          
이런식으로 어떻게 동작 해야하는지 구체적으로  설명해줘  
            """),
            HumanMessage(content=f"""
    **회의록:**
    {meeting_note}

    **분류 결과:**
    {json.dumps(main_task, ensure_ascii=False, indent=2)}

    위 결과를 관대하게 평가해주세요. JSON 형식으로 응답해주세요.
            """)
        ]
    def get_llm_invoke_prompt(self, meeting_note: str)-> List:
 
        
        return [
            SystemMessage(content="""
    당신은 팀 회의록에서 포지션별 할 일을 정확히 분류하는 전문가입니다.


    🔹 핵심 규칙:
    1. 원문 충실성: 회의록에 나온 표현을 그대로 사용하며, 의역하지 않습니다
    2. 완전성: 회의록에 언급된 모든 작업을 누락 없이 분류합니다
    3. 정확성: 각 작업을 가장 적절한 포지션에 분류합니다
    4. 일관성: 모든 포지션 키(AI, BE, FE, CLOUD)를 반드시 포함합니다

    🔹 연결어 처리 규칙 (매우 중요):
    다음 연결어로 이어진 작업들은 반드시 개별 작업으로 분리하세요:
    • "및", "하고", "그리고", "과", "와", "또한", "더불어", "아울러"
    • ",", ";"로 구분된 항목들
    • "A와 B 구현" → ["A 구현", "B 구현"]
    • "데이터베이스 설계 및 API 개발" → ["데이터베이스 설계", "API 개발"]
    • "UI 개선하고 성능 최적화" → ["UI 개선", "성능 최적화"]
    • "모델 학습, 평가, 배포" → ["모델 학습", "모델 평가", "모델 배포"]

    🔹 작업명 형식:
    • 짧고 간결한 명사구 또는 동사+명사 형태
    • 예시: "사용자 인증 구현", "데이터베이스 최적화", "메인 페이지 개발"
    • 금지: 조사(을/를/이/가), 종결어미(다/요/함), 마침표(.)
    • 연결어 제거: "및", "하고", "그리고" 등은 작업명에서 완전히 제거

    🔹 분리 예시:
    • "로그인 API 및 회원가입 기능 개발" 
      → ["로그인 API 개발", "회원가입 기능 개발"]
    • "데이터 전처리하고 모델 훈련 및 평가"
      → ["데이터 전처리", "모델 훈련", "모델 평가"]
    • "프론트엔드 컴포넌트 개발과 백엔드 API 연동"
      → FE: ["컴포넌트 개발"], BE: ["API 연동"]

    🔹 포지션별 할당 예시:
    • "AI 모델 학습" → AI 포지션
    • "로그인 API 개발" → BE 포지션
    • "대시보드 UI 개선" → FE 포지션
    • "서버 배포 자동화" → CLOUD 포지션

    🔹 빈 포지션 처리:
    해당 포지션에 할 일이 없으면 반드시 빈 배열([])로 설정합니다.

    🔹 출력 형식 (반드시 준수):
    {
      "AI": ["작업1", "작업2", ...] 또는 [],
      "BE": ["작업1", "작업2", ...] 또는 [],
      "FE": ["작업1", "작업2", ...] 또는 [],
      "CLOUD": ["작업1", "작업2", ...] 또는 []
    }

    중요: 
    1. 4개 포지션 키를 모두 포함하고, 해당 없는 포지션은 빈 배열([])로 설정하세요.
    2. 연결어로 묶인 작업들을 반드시 개별적으로 분리하여 각각 적절한 포지션에 할당하세요.
            """),
            HumanMessage(content=f"""
    다음 회의록을 분석하여 포지션별 할 일을 정확히 분류해주세요.

    회의록:
    {meeting_note}

    분석 지침:
    1. 회의록에서 언급된 모든 작업을 식별하세요
    2. 연결어(및, 하고, 그리고, 와, 과 등)로 이어진 작업들을 개별 작업으로 분리하세요
    3. 각 작업을 가장 적절한 포지션(AI/BE/FE/CLOUD)에 분류하세요
    4. 회의록 원문의 표현을 최대한 그대로 사용하되, 연결어는 제거하세요
    5. 해당 포지션에 작업이 없으면 빈 배열([])로 설정하세요
    6. 반드시 모든 포지션 키(AI, BE, FE, CLOUD)를 포함하세요
    7. 작업은 중복이 없어야합니다.

    분리 예시:
    - "데이터베이스 설계 및 API 개발" → ["데이터베이스 설계", "API 개발"]
    - "UI 컴포넌트 개발하고 스타일링" → ["UI 컴포넌트 개발", "스타일링"]
    - "모델 훈련, 검증, 배포" → ["모델 훈련", "모델 검증", "모델 배포"]

    JSON 형식으로 출력해주세요:
            """)
        ]
    
    def get_retry_prompts(self, meeting_note: str, main_task: Dict[str, List[str]], feedback: str) -> List[Dict[str, str]]:
        """재시도를 위한 개선 프롬프트 반환"""
        return [
            {
                "role": "system",
                "content": """
    당신은 팀 회의록 분석 전문가로서, 품질 검증에서 실패한 JSON 결과를 개선해야 합니다.


    🔹 **출력 형식:**
    {
      "AI": ["작업1", "작업2", ...],
      "BE": ["작업1", "작업2", ...], 
      "FE": ["작업1", "작업2", ...],
      "CLOUD": ["작업1", "작업2", ...]
    }

    
    """
            },
            {
                "role": "user",
                "content": f"""
    **중요:** 아래 피드백을 반드시 반영하여 개선된 결과를 작성하세요.
    **이전 결과 (실패):**
    {json.dumps(main_task, ensure_ascii=False, indent=2)}


    **🚨 품질 검증 피드백:**
    {feedback}

    **지시사항:**
    위 피드백에서 지적된 문제점들을 그대로 따르세요

    """
            }
        ]

    
 