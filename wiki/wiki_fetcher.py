from git import Repo
import os
import logging

logger = logging.getLogger(__name__)

class WikiFetcher:
    def __init__(self, project_id: int, url: str, base_path="./data"):
        self.project_id = project_id
        self.local_path = os.path.join(base_path, f"{project_id}-wiki")
        self.repo = None
        self.first_clone = False

        if url:
            if url.endswith("/wiki"):
                base_url = url[:-5]
                self.url = f"{base_url}.wiki.git"
            elif url.endswith(".wiki.git"):
                self.url = url
            else:
                raise ValueError(f"유효하지 않은 URL: {url}. '/wiki' 혹은 '.wiki.git' 형태의 주소를 넣어주세요.")
            
        logger.info(f"[WikiFetcher] URL: {self.url}")

    def get_diff_files(self):
        # 폴더가 없으면 clone + s전체 파일
        if not os.path.exists(self.local_path):
            logger.info(f"[WikiFetcher] Cloning repo for project {self.project_id}...")
            Repo.clone_from(self.url, self.local_path)
            self.first_clone = True

        # repo 객체 초기화 (clone 이후 or 기존 디렉토리 기준)
        if not self.repo:
            self.repo = Repo(self.local_path)

        # 항상 최신 상태 유지
        logger.info(f"[WikiFetcher] Pulling latest changes...")
        self.repo.remotes.origin.pull()

        if self.first_clone:
            logger.info("[WikiFetcher] First clone: embedding ALL .md files.")
            return [
                os.path.join(root, file)
                for root, _, files in os.walk(self.local_path)
                for file in files if file.endswith(".md")
            ]

        # 이후엔 diff 기준
        changed_files = []
        diffs = self.repo.git.diff('HEAD~1', 'HEAD', name_status=True).splitlines()
        for line in diffs:
            status, filename = line.split('\t')
            if filename.endswith(".md"):
                changed_files.append(os.path.join(self.local_path, filename))

        logger.info(f"[WikiFetcher] Changed files: {changed_files}")
        return changed_files
    