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
                builder.add_edge(START, "extract_core_tasks")
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

    def run(self, meeting_notes: str, project_id: int, position: list):
        try:
            position=list(set(position))
            init_state = {
                'meeting_note': meeting_notes,
                'project_id': project_id,
                'position': position,
                'prompt': None,
                'main_task': None,     
                'AI': None,
                'BE': None,
                'FE': None,
                'CLOUD': None,
                'error': None,
                'status': 'pending',
                'count' : 0
            }
            result = self.graph.invoke(init_state)
            
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
