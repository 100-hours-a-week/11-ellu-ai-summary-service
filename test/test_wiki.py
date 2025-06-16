import unittest
from unittest.mock import patch, MagicMock
from models.wiki.wiki_retriever import WikiRetriever

class TestWikiRetriever(unittest.TestCase):
    def setUp(self):
        # ChromaDB 클라이언트 모킹
        self.chroma_patcher = patch('chromadb.HttpClient')
        self.mock_chroma = self.chroma_patcher.start()
        self.mock_chroma.return_value = MagicMock()
        
        self.wiki_retriever = WikiRetriever()
        self.test_tasks = [
            "테스트 작업 1",
            "테스트 작업 2",
            "테스트 작업 3"
        ]
        self.test_project_id = "test-project"

    def tearDown(self):
        self.chroma_patcher.stop()

    def test_retrieve_wiki_context(self):
        """위키 컨텍스트 검색 테스트"""
        for task in self.test_tasks:
            context = self.wiki_retriever.retrieve_wiki_context(task, self.test_project_id)
            
            self.assertIsInstance(context, dict)
            self.assertIn(task, context)
            self.assertIsInstance(context[task], str)
            self.assertTrue(len(context[task]) > 0)

    def test_empty_task(self):
        """빈 작업 테스트"""
        context = self.wiki_retriever.retrieve_wiki_context("", self.test_project_id)
        
        self.assertIsInstance(context, dict)
        self.assertIn("", context)
        self.assertEqual(context[""], "")

    def test_invalid_project_id(self):
        """잘못된 프로젝트 ID 테스트"""
        with self.assertRaises(Exception):
            self.wiki_retriever.retrieve_wiki_context(self.test_tasks[0], "")

if __name__ == '__main__':
    unittest.main() 