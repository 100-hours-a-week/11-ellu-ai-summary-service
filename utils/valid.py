import re
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class valid_json :

    def remove_empty_list_keys(dictionary: dict) -> dict:
  
        return {
            key: value for key, value in dictionary.items()
            if not (isinstance(value, list) and len(value) == 0)
        }


    def validate_main_task_json(self, response) -> dict:
        """메인 태스크 JSON 유효성 검사 및 보완 (간결 버전)"""
        
        try:
            # JSON 텍스트 추출 및 파싱
            json_str = re.sub(r'```json\s*|```\s*$', '', response.content.strip())
            parsed = json.loads(json_str)
            
            # 필수 포지션별 데이터 정제
            return {
                position: [str(item).strip() for item in parsed.get(position, []) if item and str(item).strip()]
                if isinstance(parsed.get(position), list) else []
                for position in ["AI", "BE", "FE", "CLOUD"]
            }
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"JSON 검증 오류: {e}")
            return {"AI": [], "BE": [], "FE": [], "CLOUD": []}
        
    

    def validate_subtask_json(self, response) -> list:
        """서브 태스크 JSON 유효성 검사 및 보완"""
        try:
            # 1. 응답에서 JSON 텍스트 추출 및 정리
            json_str = response.content.strip()
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*$', '', json_str)
            
            # 2. JSON 파싱
            parsed_data = json.loads(json_str)
            
            # 3. 리스트 타입 검증
            if not isinstance(parsed_data, list):
                return []
            
            # 4. 각 항목 유효성 검사 및 정제
            return [
                {
                    "position": str(item.get("position", "")),
                    "task": str(item.get("task", "")),
                    "subtasks": self._normalize_subtasks(item.get("subtasks", []))
                }
                for item in parsed_data 
                if isinstance(item, dict)
            ]
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"JSON 검증 오류: {e}")
            return []

    def _normalize_subtasks(self, subtasks) -> list:
        """subtasks를 정규화"""
        if isinstance(subtasks, list):
            return [str(sub) for sub in subtasks if sub]
        elif isinstance(subtasks, str):
            return [subtasks]
        return []