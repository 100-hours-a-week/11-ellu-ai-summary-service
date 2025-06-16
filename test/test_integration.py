import unittest
import time
from unittest.mock import patch, MagicMock
from nodes.graph import MeetingWorkflow
from models.wiki.wiki_retriever import WikiRetriever
from app.config import PROJECT_ID

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # ChromaDB 클라이언트 모킹
        self.chroma_patcher = patch('chromadb.HttpClient')
        self.mock_chroma = self.chroma_patcher.start()
        self.mock_chroma.return_value = MagicMock()
        
        # WikiRetriever 모킹
        self.wiki_patcher = patch('models.wiki.wiki_retriever.WikiRetriever')
        self.mock_wiki = self.wiki_patcher.start()
        self.mock_wiki.return_value.retrieve_wiki_context.return_value = {"테스트 작업": "테스트 컨텍스트"}
        
        # Generate_llm_response 모킹
        self.llm_patcher = patch('models.llm.task_model.Generate_llm_response')
        self.mock_llm = self.llm_patcher.start()
        mock_instance = MagicMock()
        mock_instance.run_model_and_parse.return_value = {
            "AI": ["테스트 AI 작업"],
            "BE": ["테스트 BE 작업"],
            "FE": ["테스트 FE 작업"],
            "Cloud": ["테스트 Cloud 작업"]
        }
        self.mock_llm.return_value = mock_instance
        
        self.workflow = MeetingWorkflow()
        self.wiki_retriever = WikiRetriever()
        self.test_meeting_note = """
        회의 내용:
        1. 사용자 인증 시스템 구현
        2. 데이터베이스 최적화 필요
        3. 프론트엔드 UI 개선
        4. API 엔드포인트 설계
        5. 클라우드 배포 자동화
        """
        self.test_project_id = PROJECT_ID
        self.test_positions = ["AI", "BE", "FE", "CL"]

    def tearDown(self):
        self.chroma_patcher.stop()
        self.wiki_patcher.stop()
        self.llm_patcher.stop()

    def test_workflow_with_wiki(self):
        """워크플로우와 위키 검색 통합 테스트"""
        # 워크플로우 실행
        result = self.workflow.run(
            meeting_notes=self.test_meeting_note,
            project_id=self.test_project_id,
            position=self.test_positions
        )
        
        # 결과 검증
        self.assertEqual(result['status'], 'completed')
        self.assertIn('main_task', result)
        
        # 각 포지션의 태스크에 대해 위키 검색 테스트
        for position in self.test_positions:
            tasks = result['main_task'].get(position, [])
            for task in tasks:
                wiki_context = self.wiki_retriever.retrieve_wiki_context(task, self.test_project_id)
                self.assertIsInstance(wiki_context, dict)
                self.assertIn(task, wiki_context)
                self.assertIsInstance(wiki_context[task], str)

    def test_error_handling(self):
        """에러 처리 통합 테스트"""
        # 잘못된 프로젝트 ID로 테스트
        result = self.workflow.run(
            meeting_notes=self.test_meeting_note,
            project_id=-1,
            position=self.test_positions
        )
        self.assertEqual(result['status'], 'failed')
        self.assertIn('error', result)

        # 위키 검색 실패 테스트
        with self.assertRaises(Exception):
            self.wiki_retriever.retrieve_wiki_context("테스트 태스크", -1)

    def test_performance(self):
        """성능 테스트"""
        start_time = time.time()
        
        # 워크플로우 실행
        result = self.workflow.run(
            meeting_notes=self.test_meeting_note,
            project_id=self.test_project_id,
            position=self.test_positions
        )
        
        # 위키 검색 시간 측정
        for position in self.test_positions:
            tasks = result['main_task'].get(position, [])
            for task in tasks:
                self.wiki_retriever.retrieve_wiki_context(task, self.test_project_id)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 성능 검증 (예상 시간은 환경에 따라 조정 필요)
        self.assertLess(execution_time, 30)  # 워크플로우는 30초 이내

if __name__ == '__main__':
    unittest.main() 