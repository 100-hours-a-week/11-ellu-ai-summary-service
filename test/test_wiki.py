import unittest
from models.wiki.wiki_retriever import WikiRetriever
from config import PROJECT_ID

class TestWikiRetriever(unittest.TestCase):
    def setUp(self):
        self.wiki_retriever = WikiRetriever()
        self.test_project_id = PROJECT_ID
        self.test_tasks = [
            "사용자 인증 구현",
            "데이터베이스 최적화",
            "API 엔드포인트 설계",
            "프론트엔드 컴포넌트 개발"
        ]

    def test_retrieve_wiki_context(self):
        """위키 컨텍스트 검색 테스트"""
        for task in self.test_tasks:
            context = self.wiki_retriever.retrieve_wiki_context(task, self.test_project_id)
            self.assertIsInstance(context, dict)
            self.assertIn(task, context)
            self.assertIsInstance(context[task], str)
            self.assertTrue(len(context[task]) > 0)

    def test_empty_task(self):
        """빈 태스크 처리 테스트"""
        context = self.wiki_retriever.retrieve_wiki_context("", self.test_project_id)
        self.assertIsInstance(context, dict)
        self.assertEqual(context[""], "")

    def test_invalid_project_id(self):
        """잘못된 프로젝트 ID 처리 테스트"""
        with self.assertRaises(Exception):
            self.wiki_retriever.retrieve_wiki_context("테스트 태스크", -1)

if __name__ == '__main__':
    unittest.main() 