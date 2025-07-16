from langgraph.graph import StateGraph, START, END
from nodes.All_node import NodeHandler
from schemas.task import TaskState
import logging

logger = logging.getLogger(__name__)

class MeetingWorkflow:
    def __init__(self):
        self.task_handler = NodeHandler()
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            builder = StateGraph(TaskState)

                # 노드 등록
                builder.add_node("audio_to_text", self._audio_to_text_node)
                builder.add_node("extract_core_tasks", self.task_handler.extract_core_tasks)
                builder.add_node("generate_response", self.task_handler.generate_response)
                builder.add_node("generate_AI_subtasks", self.task_handler.generate_AI_response)
                builder.add_node("generate_BE_subtasks", self.task_handler.generate_BE_response)
                builder.add_node("generate_FE_subtasks", self.task_handler.generate_FE_response)
                builder.add_node("generate_Cloud_subtasks", self.task_handler.generate_Cloud_response)
                builder.add_node("retry_node", self.task_handler.retry)
                builder.add_node("judge_quality", self.task_handler.judge_quality_with_json_mode)
                builder.add_node("determine_routes", self.task_handler.route_to_subtasks)
                # 엣지 연결
                builder.add_conditional_edges(
                    START,
                    self._start_branch,
                    {
                        "audio_to_text": "audio_to_text",
                        "extract_core_tasks": "extract_core_tasks"
                    }
                )
                builder.add_edge("audio_to_text", "extract_core_tasks")
                builder.add_edge("extract_core_tasks", "generate_response")
                builder.add_edge("generate_response", "judge_quality")
                builder.add_edge("judge_quality", "determine_routes")
                # 조건 분기
                builder.add_conditional_edges(
                    "determine_routes",
                    self.task_handler.get_routes_from_state,
                    {
                        "generate_AI_subtasks": "generate_AI_subtasks",
                        "generate_BE_subtasks": "generate_BE_subtasks",
                        "generate_FE_subtasks": "generate_FE_subtasks",
                        "generate_Cloud_subtasks": "generate_Cloud_subtasks",
                        "retry_node": "retry_node" 
                    }
                )
                builder.add_edge("retry_node", "generate_response")

                # 포지션별 응답 후 종료
                builder.add_edge("generate_AI_subtasks", END)
                builder.add_edge("generate_BE_subtasks", END)
                builder.add_edge("generate_FE_subtasks", END)
                builder.add_edge("generate_Cloud_subtasks", END)

                return builder.compile()
        except Exception as e:
            logger.error(f"그래프 빌드 중 오류 발생: {str(e)}")
            raise

    def _start_branch(self, state: dict):
        """
        입력 타입에 따라 분기: audio_file_path가 있으면 STT, meeting_note가 있으면 텍스트
        """
        if state.get('audio_file_path'):
            return "audio_to_text"
        elif state.get('meeting_note'):
            return "extract_core_tasks"
        else:
            state['error'] = "No valid input provided"
            return END

    def _audio_to_text_node(self, state: dict):
        audio_file_path = state.get('audio_file_path')
        project_id = state.get('project_id')
        result = self.task_handler.audio_to_text_with_gemini(audio_file_path, project_id)
        state['meeting_note'] = result['text']
        return state

    async def arun(self, *, audio_file_path: str, meeting_note: str, project_id: int, position: list):
        try:
            position=list(set(position))
            init_state = {
                'audio_file_path': audio_file_path,
                'meeting_note': meeting_note,
                'project_id': project_id,
                'position': None,
                'prompt': None,
                'main_task': None,
                'AI': None,
                'BE': None,
                'FE': None,
                'CLOUD': None,
                'error': None,
                'status': 'pending',
                'count' : 0,
                'project_position':None
            }
            result = await self.graph.ainvoke(init_state)
            
            # 결과 상태 업데이트
            if 'error' in result:
                result['status'] = 'failed'
            else:
                result['status'] = 'completed'
                
            return result
        except Exception as e:
            logger.error(f"워크플로우 실행 중 오류 발생: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed'
            }

# if __name__ == "__main__":
#     workflow = MeetingWorkflow()
#     result = workflow.run(
#         meeting_notes="오늘 회의에서는 AI팀이 사용자 분석 기능을 구현해야 한다고 논의했습니다.",
#         project_id=1,
#         position="AI",
#         prompt=None
#     )
#     print(result)
