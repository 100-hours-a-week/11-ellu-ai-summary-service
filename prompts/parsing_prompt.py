import json
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

class Json_Parsing_Prompts:
    def __init__(self, wiki_context: str = ""):
        """
        JSON 파싱 프롬프트 관리 클래스
        
        Args:
            wiki_context: 위키 컨텍스트 (현재 미사용이지만 확장 가능)
        """
        self.wiki_context = wiki_context
    
    def get_main_json_prompt(self, raw_text: str) -> List:
        """
        메인 태스크 JSON 수정을 위한 프롬프트 반환
        
        Args:
            raw_text: 수정할 원본 텍스트
            
        Returns:
            List: [SystemMessage, HumanMessage]
        """
        return [
            SystemMessage(content="""당신은 JSON 형식 오류를 수정하는 전문가입니다.

🔹 **입력 데이터 특성:**
- 회의록 분석 결과로 생성된 포지션별 작업 분류 데이터
- 예상 형식: {"AI": [...], "BE": [...], "FE": [...], "CLOUD": [...]}

🔹 **수정 규칙:**
1. **구조 복원**: 누락된 중괄호, 대괄호, 쉼표 추가
2. **키 표준화**: 포지션 키는 반드시 "AI", "BE", "FE", "CLOUD"로 통일
3. **값 정리**: 배열 내 문자열은 따옴표로 감싸기
4. **불필요한 요소 제거**: 주석, 설명문, 마크다운 코드 블록 제거
5. **완전성 보장**: 4개 포지션 키가 모두 존재하도록 보장

🔹 **일반적인 오류 패턴:**
- 마지막 쉼표 누락: `{"AI": ["작업1"] "BE": ["작업2"]}`
- 따옴표 누락: `{AI: [작업1, 작업2]}`
- 배열 구조 오류: `{"AI": "작업1", "작업2"}`
- 마크다운 블록: ```json {...} ```
- 불완전한 JSON: `{"AI": ["작업1"`

🔹 **출력 요구사항:**
- 유효한 JSON 객체만 반환 (설명 없음)
- 모든 포지션 키 포함 (빈 배열이라도)
- 표준 JSON 형식 준수

🔹 **예시:**
입력: `{AI: [사용자 분석], BE: [API 개발, 데이터베이스 설계}}`
출력: `{"AI": ["사용자 분석"], "BE": ["API 개발", "데이터베이스 설계"], "FE": [], "CLOUD": []}`"""),
            
            HumanMessage(content=f"""다음 텍스트를 올바른 JSON 형식으로 수정해주세요:

**원본 텍스트:**
{raw_text}

**요구사항:**
1. 포지션별 작업 분류 JSON 형태로 수정
2. AI, BE, FE, CLOUD 키 모두 포함
3. 각 키의 값은 문자열 배열
4. 유효한 JSON 형식만 반환 (설명 없음)

수정된 JSON:""")
        ]

    def get_sub_json_prompt(self, raw_text: str) -> List:
        """
        서브 태스크 JSON 수정을 위한 프롬프트 반환
        
        Args:
            raw_text: 수정할 원본 텍스트
            
        Returns:
            List: [SystemMessage, HumanMessage]
        """
        return [
            SystemMessage(content="""당신은 JSON 형식 오류를 수정하는 전문가입니다.

🔹 **입력 데이터 특성:**
- 특정 포지션의 작업을 세부 단계로 분해한 결과
- 예상 형식: [{"position": "AI", "task": "모델 개발", "subtasks": ["데이터 수집", "모델 훈련", "성능 평가"]}]

🔹 **수정 규칙:**
1. **구조 복원**: 배열과 객체 구조 정확히 복원
2. **필수 키 보장**: position, task, subtasks 키 모두 포함
3. **데이터 타입 정확성**: 
   - position: 문자열 (AI/BE/FE/CLOUD)
   - task: 문자열 (원본 작업명)
   - subtasks: 문자열 배열
4. **불필요한 요소 제거**: 설명문, 주석, 마크다운 제거



🔹 **출력 요구사항:**
- 유효한 JSON 배열만 반환
- 각 객체는 position, task, subtasks 키 포함
- 배열 내 모든 요소는 동일한 구조

🔹 **예시:**

출력 예시시: 
        {
          "position": "AI", 
          "task": "모델개발", 
          "subtasks": ["데이터수집", "모델훈련"]
        }
     """),
            
            HumanMessage(content=f"""다음 텍스트를 올바른 JSON 배열 형식으로 수정해주세요:

**원본 텍스트:**
{raw_text}

**요구사항:**
1. 서브태스크 구조 JSON 배열로 수정
2. 각 객체는 position, task, subtasks 키 포함
3. position은 문자열, task는 문자열, subtasks는 문자열 배열
4. 유효한 JSON 형식만 반환 (설명 없음)

수정된 JSON:""")
        ]

    def get_general_json_prompt(self, raw_text: str, position: str, tasks: str) -> List:
        """
        일반적인 JSON 수정을 위한 프롬프트 반환 (서브태스크용)
        
        Args:
            raw_text: 수정할 원본 텍스트
            position: 포지션 정보 (AI/BE/FE/CLOUD)
            tasks: 태스크 정보
            
        Returns:
            List: [SystemMessage, HumanMessage]
        """
        return [
            SystemMessage(content=f"""당신은 배열 형식 오류를 수정하는 전문가입니다.

🔹 **수정 규칙:**
1. **구조 복원**: 누락된 구문 요소 추가
2. **문법 수정**: 올바른 배열 문법으로 변환
3. **데이터 정리**: 불필요한 요소 제거
4. **형식 통일**: 일관된 배열 구조 유지

🔹 **출력 요구사항:**
- 유효한 배열만 반환 (설명 없음)
- 원본 데이터의 의미 보존
- 표준 배열 형식 준수

🔹 **출력 형식:**
```json
["구체적인 세부 작업 1","구체적인 세부 작업 2", "구체적인 세부 작업 3"]
```                         
"""),
            
            HumanMessage(content=f"""다음 텍스트를 올바른 배열 형식으로 수정해주세요:

**원본 텍스트:**
{raw_text}

수정된 JSON:""")
        ]