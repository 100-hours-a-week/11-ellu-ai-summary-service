from langchain_openai import ChatOpenAI
from prompts.prompt import MeetingPromptManager
from schemas.task import TaskState
from models.wiki.basic_retriever import BasicRetriever
from models.pinecone.hybrid_retriever import PineconeHybridRetriever
from models.llm.task_model import Generate_llm_response
from app.config import GPT_MODEL, TEMPERATURE, MODEL_KWARGS
import json
import logging
import asyncio
from utils.valid import valid_json

logger = logging.getLogger(__name__)

class NodeHandler:
    def __init__(self):
        logger.info(" NodeHandler 초기화 시작...")
        logger.info("Pinecone 초기화 시도 중...")
        try:
            self.pinecone_retriever = PineconeHybridRetriever()
            self.prompt = MeetingPromptManager(pinecone_retriever=self.pinecone_retriever)
            logger.info("Pinecone 하이브리드 검색기 초기화 성공!")
        except Exception as e:
            logger.error(f"Pinecone 초기화 실패, Wiki만 사용: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            self.pinecone_retriever = None
            self.prompt = MeetingPromptManager()
        
        self.wiki_retriever = BasicRetriever()

        self.task_model = Generate_llm_response()
        self.valid=valid_json()
        self.llm = ChatOpenAI(
            model=GPT_MODEL,
            temperature=TEMPERATURE,
            model_kwargs=MODEL_KWARGS
        )
        logger.info(f"NodeHandler 초기화 완료 - Pinecone 사용: {self.pinecone_retriever is not None}")

    def extract_core_tasks(self, state: TaskState) -> dict:
        try:
            meeting_note = state["meeting_note"]
            system_prompt = self.prompt.get_main_prompt()
            user_prompt = self.prompt.get_user_prompt(meeting_note)
            result = {'prompt':  [system_prompt, user_prompt]}
            logger.info(f"핵심 태스크 추출 성공 - 회의록 길이: {len(meeting_note)} 글자")
            return result
        except Exception as e:
            logger.error(f"핵심 태스크 추출 중 오류: {str(e)}")
            return {'error': str(e)}

    async def generate_response(self, state: TaskState) -> dict:
        try:
            chat = state['prompt']
            parsed = await self.task_model.run_model_and_parse(chat,"main")
            
             # 빈 배열의 키 제거
            logger.info("메인 태스크 응답 생성 성공")
            for i in list(parsed.keys()):
                if parsed[i] == [] or i not in ["AI","BE","CLOUD","FE"]:
                    del parsed[i]
            logger.info(f"parsed:{parsed}")
            return {'main_task': parsed,'project_position' : list(parsed.keys())} 

        except Exception as e:
            logger.error(f"응답 생성 중 오류: {str(e)}")
            return {'error': str(e)}

    async def generate_position_response(self, state: TaskState, key: str) -> dict:
        try:
            tasks = state['main_task'][key]
            if not tasks:
                logger.info(f"포지션 {key}: 처리할 태스크가 없음")
                return {key: []}
            
            outputs = []
            for task in tasks:
                logger.info(f"Processing task: {task}")

                # Pinecone 검색 우선 사용
                wiki_context = ""
                try:
                    if self.pinecone_retriever:
                        # 1차 검색
                        logger.info(f"Pinecone 검색 시작 for task '{task}', position '{key}'")
                        pinecone_context = self.pinecone_retriever.get_enhanced_subtask_context(task, key)
                        logger.info(f"Pinecone 결과: {len(pinecone_context.strip()) if pinecone_context else 0} 문자")
                        
                        if pinecone_context and len(pinecone_context.strip()) > 50:
                            wiki_context = pinecone_context
                            logger.info(f"Pinecone context 사용 for task '{task}': {len(wiki_context)} 문자")
                            logger.info("=" * 50)
                        else:
                            # 2차 fall back
                            logger.info(f"Pinecone context 부족, Wiki fallback 사용 for task '{task}'")
                            wiki_result = self.wiki_retriever.retrieve_wiki_context(task, state['project_id'])
                            wiki_context = wiki_result.get(task, "")
                    else:
                        # Pinecone 없을시
                        logger.info(f"Pinecone 없음, Wiki만 사용 for task '{task}'")
                        wiki_result = self.wiki_retriever.retrieve_wiki_context(task, state['project_id'])
                        wiki_context = wiki_result.get(task, "")

                    if wiki_context:
                        logger.info(f"Context retrieved for task '{task}': {len(wiki_context)} chars")
                    else:
                        logger.warning(f"No context found for task '{task}', project_id: {state['project_id']}")
                        wiki_context = f"{task}에 대한 {key} 관점에서의 구체적인 접근이 필요합니다."
                        
                except Exception as e:
                    logger.error(f"Context retrieval failed for task '{task}': {e}")
                    wiki_context = f"{task}에 대한 {key} 관점에서의 접근이 필요합니다." 
                # logger.info(f" wiki 내용: {wiki_result}")
                role= self.prompt.subtask_position_role(key)
                chat = self.prompt.get_subtask_prompts(key, task, wiki_context,role)
                

                response = await self.task_model.run_model_and_parse(chat, "sub",task,key)
                logger.info(f" 만들어진 sub task 답변 : {response}")
                if isinstance(response, dict):
                    values = list(response.values())
                    if values and all(isinstance(v, list) for v in values):
                        response = sum(values, [])  # 모든 값들을 하나의 리스트로 합침
                    else:
                        response = []
                else:
                    response = []
                    
                parsed=[{"position": key, "task": task, "subtasks": response}]

                logger.info(f" parsed 내용: {parsed}")

                outputs.extend(parsed) 
            
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

    def get_routes_from_state(self, state: TaskState) -> list[str]:
        return state.get('routes', [])

    def route_to_subtasks(self, state: TaskState) -> dict:
        """서브태스크 라우팅 로직"""
        mapping = {
            "ai": "generate_AI_subtasks",
            "be": "generate_BE_subtasks", 
            "fe": "generate_FE_subtasks",
            "cloud": "generate_Cloud_subtasks",
        }
        
        # 검증 통과 또는 최대 재시도 횟수 도달
        if state['validation_result'] == "pass":
            
            # 최대 재시도 도달 시 강제로 LLM 재생성

            
            # 정상 통과 시
            

                routes = [mapping[p.lower()] for p in state['project_position'] if p.lower() in mapping]

                logger.info(f"검증 통과 - 서브태스크 라우팅: {len(routes)}개 경로")
                return {'routes': routes}
        elif   state['count'] >= 1:
                logger.warning(f"최대 재시도 횟수({state['count']}) 도달 - 강제 LLM 재생성")
                try:
                    meeting_note = state["meeting_note"]
                    prompt = self.prompt.get_llm_invoke_prompt(meeting_note)
                    response = self.llm.invoke(prompt)
                    
                    # 올바른 메서드 호출
                    parsed = self.valid.validate_main_task_json(response)
                    for i in list(parsed.keys()):  # 빈배열의 키 삭제
                        if parsed[i] == []:
                            del parsed[i]
                    position = list(parsed.keys())
                    # 상태 업데이트를 반환값으로 처리
                    routes = [mapping[p.lower()] for p in position if p.lower() in mapping]
                    logger.info(f"강제 재생성 후 라우팅 - {len(routes)}개 경로: {routes} parsed : {parsed}")
                    # logger.info(f"강제 재생성 후 라우팅 - {len(routes)}개 경로: {routes}")
                    # return {'routes': routes}
                    return {
                        'main_task': parsed,  # 상태 업데이트
                        'routes': routes,
                        'project_position' :  position   # 라우팅 정보
 }
                    
                except Exception as e:
                    logger.error(f"강제 재생성 중 오류: {str(e)}")
                    # 오류 발생 시 기존 데이터로 진행
                    routes = [mapping[p.lower()] for p in state['project_position'] if p.lower() in mapping]
                    return {'routes': routes}
        
        # 재시도 필요
        else:
            logger.info(f"검증 실패 - 재시도 진행 (현재 횟수: {state['count']})")
            return {'routes': ["retry_node"]}


    def retry(self, state: TaskState) -> dict:
        try:
            meeting_note = state["meeting_note"]
            main_task = state["main_task"]
            feedback = state["feedback"]
            
            prompt_list = self.prompt.get_retry_prompts(meeting_note, main_task, feedback)
            result = {'prompt':  prompt_list,'count':state['count']+1}
            logger.info("재시도 프롬프트 생성 성공")
            return result

        except Exception as e:
            logger.error(f"재시도 중 오류: {str(e)}")
            return {'error': str(e)}

    async def generate_all_position_responses(self, state: TaskState) -> dict:
        try:
            positions = state["project_position"]
            logger.info(f"전체 포지션 응답 생성 시작 - 대상 포지션: {positions}")
            
            tasks = [self.generate_position_response(state, pos) for pos in positions]
            results_list = await asyncio.gather(*tasks)
            
            results = {}
            for res in results_list:
                results.update(res)
            
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
    
    async def generate_AI_response(self, state: TaskState) -> dict:
        return await self.generate_position_response(state, "AI")

    async def generate_BE_response(self, state: TaskState) -> dict:
        return await self.generate_position_response(state, "BE")

    async def generate_FE_response(self, state: TaskState) -> dict:
        return await self.generate_position_response(state, "FE")

    async def generate_Cloud_response(self, state: TaskState) -> dict:
        return await self.generate_position_response(state, "CLOUD") 
