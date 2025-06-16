import unittest
from nodes.graph import MeetingWorkflow

class TestMeetingWorkflow(unittest.TestCase):
    def setUp(self):
        self.workflow = MeetingWorkflow()
        self.test_meeting_note = """
        오늘의 회의 내용:
        1. 백엔드 API 개발 시작
        2. 프론트엔드 UI 컴포넌트 구현
        3. 클라우드 서버 배포 환경 구성
        """
        self.test_project_id = 1
        self.test_positions = ["BE", "FE", "CL"]

    def test_workflow_execution(self):
        """전체 워크플로우 실행 테스트"""
        result = self.workflow.run(
            meeting_notes=self.test_meeting_note,
            project_id=self.test_project_id,
            position=self.test_positions
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('main_task', result)
        
        if result['status'] == 'completed':
            main_task = result['main_task']
            self.assertIsInstance(main_task, dict)
            for position in self.test_positions:
                self.assertIn(position, main_task)
                self.assertIsInstance(main_task[position], list)

    def test_empty_meeting_note(self):
        """빈 회의록 처리 테스트"""
        result = self.workflow.run(
            meeting_notes="",
            project_id=self.test_project_id,
            position=self.test_positions
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('error', result)

    def test_invalid_project_id(self):
        """잘못된 프로젝트 ID 처리 테스트"""
        result = self.workflow.run(
            meeting_notes=self.test_meeting_note,
            project_id=-1,
            position=self.test_positions
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('error', result)

    def test_empty_positions(self):
        """빈 포지션 리스트 처리 테스트"""
        result = self.workflow.run(
            meeting_notes=self.test_meeting_note,
            project_id=self.test_project_id,
            position=[]
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('error', result)

    def test_single_position(self):
        """단일 포지션 처리 테스트"""
        result = self.workflow.run(
            meeting_notes=self.test_meeting_note,
            project_id=self.test_project_id,
            position=["BE"]
        )
        
        self.assertIsInstance(result, dict)
        if result['status'] == 'completed':
            main_task = result['main_task']
            self.assertIn('BE', main_task)
            self.assertNotIn('FE', main_task)
            self.assertNotIn('CL', main_task)

if __name__ == '__main__':
    unittest.main() 