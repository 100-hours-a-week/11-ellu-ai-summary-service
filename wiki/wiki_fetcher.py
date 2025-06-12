from git import Repo
import os
    


class WikiFetcher:
    PROJECT_REPOS = {
        # 1: "https://github.com/100-hours-a-week/1-team-YouTIL-wiki.wiki.git",
        # 2: "https://github.com/100-hours-a-week/2-hertz-wiki.wiki.git",
        # 3: "https://github.com/100-hours-a-week/3-team-YouTIL-wiki.wiki.git",
        # 4: "https://github.com/100-hours-a-week/4-team-YouTIL-wiki.wiki.git",
        # 5: "https://github.com/100-hours-a-week/5-team-YouTIL-wiki.wiki.git",
        # 6: "https://github.com/100-hours-a-week/6-team-YouTIL-wiki.wiki.git",
        # 7: "https://github.com/100-hours-a-week/7-team-YouTIL-wiki.wiki.git",
        # 8: "https://github.com/100-hours-a-week/8-team-YouTIL-wiki.wiki.git",
        # 9: "https://github.com/100-hours-a-week/9-team-YouTIL-wiki.wiki.git",
        # 10: "https://github.com/100-hours-a-week/10-team-YouTIL-wiki.wiki.git",
        11: "https://github.com/100-hours-a-week/11-ellu-wiki.wiki.git",
    }

    def __init__(self, project_id: int, base_path="./data"):
        self.project_id = project_id
        self.url = self.PROJECT_REPOS[project_id]
        self.local_path = os.path.join(base_path, f"{project_id}-wiki")
        self.repo = None
        self.first_clone = False

    def get_diff_files(self):
        # 폴더가 없으면 clone + 전체 파일
        if not os.path.exists(self.local_path):
            print(f"[WikiFetcher] Cloning repo for project {self.project_id}...")
            Repo.clone_from(self.url, self.local_path)
            self.first_clone = True

        # repo 객체 초기화 (clone 이후 or 기존 디렉토리 기준)
        if not self.repo:
            self.repo = Repo(self.local_path)

        # 항상 최신 상태 유지
        print(f"[WikiFetcher] Pulling latest changes...")
        self.repo.remotes.origin.pull()

        if self.first_clone:
            print("[WikiFetcher] First clone: embedding ALL .md files.")
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

        print(f"[WikiFetcher] Changed files: {changed_files}")
        return changed_files
    