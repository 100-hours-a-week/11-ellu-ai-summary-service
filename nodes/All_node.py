from langchain_openai import ChatOpenAI
from prompts.prompt import MeetingPromptManager
from schemas.task import TaskState
from models.wiki.wiki_retriever import WikiRetriever
from models.llm.task_model import Generate_llm_response
from app.config import GPT_MODEL, TEMPERATURE, MODEL_KWARGS
import json
import logging
import concurrent.futures
from utils.valid import valid_json

logger = logging.getLogger(__name__)

class NodeHandler:
    def __init__(self):
        self.prompt = MeetingPromptManager()
        self.wiki_retriever = WikiRetriever()
        self.task_model = Generate_llm_response()
        self.valid=valid_json()
        self.llm = ChatOpenAI(
            model=GPT_MODEL,
            temperature=TEMPERATURE,
            model_kwargs=MODEL_KWARGS
        )
        logger.info("NodeHandler 초기화 완료")

    def extract_core_tasks(self, state: TaskState) -> dict:
        try:
            meeting_note = state["meeting_note"]
            system_prompt = self.prompt.get_main_prompt()
            user_prompt = self.prompt.get_user_prompt(meeting_note)
            result = {'prompt': {'main_task': [system_prompt, user_prompt]}}
            logger.info(f"핵심 태스크 추출 성공 - 회의록 길이: {len(meeting_note)} 글자")
            return result
        except Exception as e:
            logger.error(f"핵심 태스크 추출 중 오류: {str(e)}")
            return {'error': str(e)}

    def generate_response(self, state: TaskState) -> dict:
        try:
            chat = state['prompt']['main_task']
            parsed = self.task_model.run_model_and_parse(chat,"main")
            logger.info("메인 태스크 응답 생성 성공")
            return {'main_task': parsed}
        except Exception as e:
            logger.error(f"응답 생성 중 오류: {str(e)}")
            return {'error': str(e)}

    def generate_position_response(self, state: TaskState, key: str) -> dict:
        try:
            tasks = state['main_task'][key]
            if not tasks:
                logger.info(f"포지션 {key}: 처리할 태스크가 없음")
                return {key: []}
            
            outputs = []
            for task in tasks:
                logger.info(f"Processing task: {task}")

                try:
                    wiki_result = self.wiki_retriever.retrieve_wiki_context(task, state['project_id'])
                    wiki_context = wiki_result.get(task, "") # 텍스트 추출

                    if wiki_context:
                        logger.info(f"Retrieved wiki for Project ID {state['project_id']}: {len(wiki_context)} chars")
                    else:
                        logger.warning(f"No wiki content found for Project ID {state['project_id']}")
                        wiki_context = ""  
                        
                except Exception as e:
                    logger.error(f"Wiki retrieval failed for Project ID {state['project_id']}: {e}")
                    wiki_context = "" 

                chat = self.prompt.get_subtask_prompts(key, task, wiki_context)
                
                parsed = self.task_model.run_model_and_parse(chat, "sub")
            
                outputs.append(parsed) 
            
            logger.info(f"포지션 {key} 응답 생성 성공 - {len(tasks)}개 태스크 처리, {len(outputs)}개 결과 생성")
            return {key: outputs}
        except Exception as e:
            logger.error(f"포지션별 응답 생성 중 오류: {str(e)}")
            return {'error': str(e)}


    def judge_quality_with_json_mode(self, state: TaskState) -> dict:
        try:
            meeting_note = state["meeting_note"]
            main_task = state["main_task"]
            messages = self.prompt.get_judge_prompts(meeting_note, main_task)
            
            response = self.llm.invoke(messages)
            evaluation_data = json.loads(response.content)
            
            feedback = self._generate_feedback(evaluation_data)
            result = {
                "validation_result": evaluation_data["result"],
                "feedback": feedback
            }
            
            logger.info(f"품질 평가 완료 - 결과: {evaluation_data['result']}")
            if evaluation_data["result"] == "fail":
                logger.warning(f"품질 평가 실패 - 피드백: {feedback}")
                
            return result
        except Exception as e:
            logger.error(f"품질 평가 중 오류: {str(e)}")
            return {
                "validation_result": "fail",
                "feedback": f"평가 중 오류 발생: {str(e)}"
            }

    def _generate_feedback(self, evaluation_data: dict) -> str:
        if evaluation_data["result"] == "pass":
            return ""
        
        feedback_parts = []
        if evaluation_data.get("failure_reasons"):
            feedback_parts.append("실패사유:")
            feedback_parts.extend(f"- {reason}" for reason in evaluation_data["failure_reasons"])
        
        if evaluation_data.get("improvement_suggestions"):
            feedback_parts.append("\n개선방향:")
            feedback_parts.extend(f"- {suggestion}" for suggestion in evaluation_data["improvement_suggestions"])
        
        return "\n".join(feedback_parts)

    def route_to_subtasks(self, state: TaskState) -> list[str]:
        
            mapping = {
                "ai": "generate_AI_subtasks",
                "be": "generate_BE_subtasks",
                "fe": "generate_FE_subtasks",
                "cloud": "generate_Cloud_subtasks",
            }
            if state['validation_result'] == "pass" or state['count'] == 4:
                if state['count'] == 4 :
                    meeting_note = state["meeting_note"]
                    prompt = self.prompt.get_llm_invoke_prompt(meeting_note)
                    
                    response = self.llm.invoke(prompt)
                    parsed = self.valid(response)
                    routes = [mapping[p.lower()] for p in state['position'] if p.lower() in mapping]
                    logger.info(f"서브태스크 라우팅 성공 - {len(routes)}개 경로: {routes}")
                    state['main_task']= parsed
                    return routes
                    

                else:
                    routes = [mapping[p.lower()] for p in state['position'] if p.lower() in mapping]
                    logger.info(f"서브태스크 라우팅 성공 - {len(routes)}개 경로: {routes}")
                    return routes
            else :
                return ["retry_node"]


    def retry(self, state: TaskState) -> dict:
        try:
            meeting_note = state["meeting_note"]
            main_task = state["main_task"]
            feedback = state["feedback"]
            
            prompt_list = self.prompt.get_retry_prompts(meeting_note, main_task, feedback)
            result = {'prompt': {'main_task': prompt_list},'count':state['count']+1}
            logger.info("재시도 프롬프트 생성 성공")
            return result

        except Exception as e:
            logger.error(f"재시도 중 오류: {str(e)}")
            return {'error': str(e)}

    def generate_all_position_responses(self, state: TaskState) -> dict:
        try:
            positions = state["position"]
            logger.info(f"전체 포지션 응답 생성 시작 - 대상 포지션: {positions}")
            
            results = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_key = {
                    executor.submit(self.generate_position_response, state, pos): pos
                    for pos in positions
                }
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        result = future.result()
                        results.update(result)
                        completed_count += 1
                        logger.debug(f"포지션 {key} 처리 완료 ({completed_count}/{len(positions)})")
                    except Exception as e:
                        logger.error(f"포지션 {key} 처리 중 오류: {str(e)}")
                        results[key] = {'error': str(e)}
            
            success_count = sum(1 for v in results.values() if not isinstance(v, dict) or 'error' not in v)
            logger.info(f"전체 포지션 응답 생성 완료 - 성공: {success_count}/{len(positions)}")
            
            return results
        except Exception as e:
            logger.error(f"전체 포지션 응답 생성 중 오류: {str(e)}")
            return {'error': str(e)}
        
    def route_after_validation(self, state: TaskState) -> list[str]:
        if state['validation_result'] == 'fail':
            return ["retry_node"]
        else:
            return self.route_to_subtasks(state)
    
    def generate_AI_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "AI")

    def generate_BE_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "BE")

    def generate_FE_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "FE")

    def generate_Cloud_response(self, state: TaskState) -> dict:
        return self.generate_position_response(state, "CLOUD") 