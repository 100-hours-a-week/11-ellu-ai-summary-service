import boto3
import zipfile
import tempfile
import shutil
from git import Repo
import os
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class WikiFetcher:
    def __init__(self, project_id: int, url: str):
        github_token = os.getenv("GITHUB_TOKEN")

        self.project_id = project_id
        
        # S3 설정
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        
        # 최적 S3 경로 찾기 (테스트에서 모든 경로가 작동했으므로 uploads 사용)
        self.s3_prefix = f"uploads/wikis/{project_id}/"
        
        # URL 처리
        if url:
            if url.endswith("/wiki"):
                base_url = url[:-5]
                if github_token:
                    parsed_url = base_url.replace("https://github.com/", f"https://{github_token}@github.com/")
                    self.url = f"{parsed_url}.wiki.git"
                else:
                    self.url = f"{base_url}.wiki.git"
            else:
                raise ValueError(f"유효하지 않은 URL: {url}. '/wiki' 형태의 주소를 넣어주세요.")
            
        logger.info(f"[WikiFetcher] URL: {self.url}, S3 Prefix: {self.s3_prefix}")


    def _upload_to_s3(self, repo_path):
        try:
            # 기존 S3 파일들 모두 삭제 (업데이트를 위해)
            self._clear_s3_folder()
            
            uploaded_count = 0
            
            # repo_path의 모든 파일을 순회
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # 로컬 경로에서 S3 키 생성
                    relative_path = os.path.relpath(local_file_path, repo_path)
                    s3_key = f"{self.s3_prefix}{relative_path.replace(os.sep, '/')}"  # Windows 호환성
                    
                    # S3에 파일 업로드
                    self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
                    uploaded_count += 1
            
            logger.info(f"Successfully uploaded {uploaded_count} files to S3")
                
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise


    def _download_from_s3(self, temp_dir):
        try:
            repo_path = os.path.join(temp_dir, f"{self.project_id}-wiki")
            os.makedirs(repo_path, exist_ok=True)
            
            # S3에서 프로젝트 폴더 내 모든 객체 나열
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.s3_prefix
            )
            
            if 'Contents' not in response:
                logger.info("No existing files in S3")
                return False
            
            download_count = 0
            # 각 파일 다운로드
            for obj in response['Contents']:
                s3_key = obj['Key']
                # S3 키에서 로컬 파일 경로 생성
                relative_path = s3_key[len(self.s3_prefix):]
                if not relative_path:  # 빈 키 건너뛰기
                    continue
                    
                local_file_path = os.path.join(repo_path, relative_path)
                
                # 디렉토리 생성
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # 파일 다운로드
                self.s3_client.download_file(self.bucket_name, s3_key, local_file_path)
                download_count += 1
            
            logger.info(f"Downloaded {download_count} files from S3")
            return True
            
        except Exception as e:
            logger.warning(f"S3 download failed: {e}")
            return False

    def _clear_s3_folder(self):
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.s3_prefix
            )
            
            if 'Contents' in response:
                # 삭제할 객체 목록 생성
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                
                # 1000개씩 배치 삭제 (S3 제한)
                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i:i+1000]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': batch}
                    )
                
                logger.info(f"Deleted {len(objects_to_delete)} files from S3")
                
        except Exception as e:
            logger.error(f"Failed to clear S3 folder: {e}")

    def _read_md_files(self, repo_path, file_paths):
        file_contents = {}
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 상대 경로를 키로 사용
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_contents[relative_path] = content
                    logger.debug(f"Read file: {relative_path} ({len(content)} chars)")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                
        return file_contents

    def get_diff_files(self):
            # 임시 디렉토리에서 모든 작업 수행
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = os.path.join(temp_dir, f"{self.project_id}-wiki")
                
                # S3에서 기존 파일들 다운로드 시도
                existing_repo = self._download_from_s3(temp_dir)
                
                if not existing_repo:
                    # clone
                    logger.info(f"[WikiFetcher] First clone for project {self.project_id}")
                    Repo.clone_from(self.url, repo_path)
                    
                    # S3에 개별 파일로 업로드
                    self._upload_to_s3(repo_path)
                    
                    all_md_files = []
                    for root, _, files in os.walk(repo_path):
                        for file in files:
                            if file.endswith(".md"):
                                all_md_files.append(os.path.join(root, file))
                    
                    logger.info(f"First clone: found {len(all_md_files)} .md files")
                    
                    # read
                    file_contents = self._read_md_files(repo_path, all_md_files)
                    return file_contents
                
                # 기존 repo 업데이트
                repo = Repo(repo_path)
                old_head = repo.head.commit.hexsha
                logger.info(f"Old HEAD: {old_head[:8]}")
                
                logger.info(f"[WikiFetcher] Pulling latest changes...")
                repo.remotes.origin.pull()
                
                new_head = repo.head.commit.hexsha
                logger.info(f"New HEAD: {new_head[:8]}")
                
                # 변경사항 있을사 S3 업데이트
                if old_head != new_head:
                    logger.info("Changes detected - updating S3...")
                    
                    try:
                        all_diffs = repo.git.diff(f'{old_head}..{new_head}', name_status=True).splitlines()
                        logger.info(f"Total changed files: {len(all_diffs)}")
                        
                        for diff_line in all_diffs[:10]: 
                            logger.info(f"Changed: {diff_line}")
                        
                        if len(all_diffs) > 10:
                            logger.info(f"... and {len(all_diffs) - 10} more files")
                            
                    except Exception as e:
                        logger.error(f"Failed to get full diff: {e}")
                    
                    self._upload_to_s3(repo_path)
                    
                    changed_files = []
                    try:
                        diffs = repo.git.diff(f'{old_head}..{new_head}', name_status=True).splitlines()
                        
                        for line in diffs:
                            if '\t' in line:
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    status, filename = parts[0], parts[1]
                                    logger.debug(f"Processing diff line: status={status}, filename={filename}")
                                    
                                    if filename.endswith(".md"):
                                        file_path = os.path.join(repo_path, filename)
                                        if os.path.exists(file_path):
                                            changed_files.append(file_path)
                                            logger.info(f"Found changed .md file: {filename}")
                                        else:
                                            logger.warning(f"Changed .md file not found: {filename}")
                                            
                    except Exception as e:
                        logger.error(f"Failed to get diff: {e}")
                        logger.info("Falling back to all .md files")
                        changed_files = []
                        for root, _, files in os.walk(repo_path):
                            for file in files:
                                if file.endswith(".md"):
                                    changed_files.append(os.path.join(root, file))
                    
                    logger.info(f"Found {len(changed_files)} changed .md files")
                    
                    if len(changed_files) == 0:
                        all_md_files = []
                        for root, _, files in os.walk(repo_path):
                            for file in files:
                                if file.endswith(".md"):
                                    all_md_files.append(os.path.join(root, file))
                        
                        logger.warning(f"No .md files changed, but {len(all_md_files)} total .md files exist")
                        
                        if len(all_md_files) > 0:
                            logger.info("Sample .md files in repo:")
                            for md_file in all_md_files[:5]:
                                relative_path = os.path.relpath(md_file, repo_path)
                                logger.info(f"  {relative_path}")
                    
                    file_contents = self._read_md_files(repo_path, changed_files)
                    return file_contents
                else:
                    logger.info("No changes detected")
                    return {}
            

    # def delete_project_data(self):
    #     try:
    #         self._clear_s3_folder()
    #         logger.info(f"Successfully deleted all S3 data for project {self.project_id}")
    #         return True
    #     except Exception as e:
    #         logger.error(f"S3 delete failed: {e}")
    #         return False

    # def get_project_info(self):
    #     try:
    #         # 폴더 내 파일 목록 조회
    #         response = self.s3_client.list_objects_v2(
    #             Bucket=self.bucket_name,
    #             Prefix=self.s3_prefix
    #         )
            
    #         if 'Contents' not in response:
    #             return {
    #                 'project_id': self.project_id,
    #                 'exists': False,
    #                 's3_prefix': self.s3_prefix,
    #                 'file_count': 0,
    #                 'total_size': 0
    #             }
            
    #         file_count = len(response['Contents'])
    #         total_size = sum(obj['Size'] for obj in response['Contents'])
    #         latest_modified = max(obj['LastModified'] for obj in response['Contents'])
            
    #         return {
    #             'project_id': self.project_id,
    #             'exists': True,
    #             's3_prefix': self.s3_prefix,
    #             'file_count': file_count,
    #             'total_size': total_size,
    #             'total_size_mb': round(total_size / (1024 * 1024), 2),
    #             'latest_modified': latest_modified
    #         }
            
    #     except ClientError as e:
    #         logger.error(f"Failed to get project info: {e}")
    #         return {
    #             'project_id': self.project_id,
    #             'exists': False,
    #             'error': str(e)
    #         }

    # def list_project_files(self):
    #     try:
    #         response = self.s3_client.list_objects_v2(
    #             Bucket=self.bucket_name,
    #             Prefix=self.s3_prefix
    #         )
            
    #         if 'Contents' not in response:
    #             return []
            
    #         files = []
    #         for obj in response['Contents']:
    #             relative_path = obj['Key'][len(self.s3_prefix):]
    #             if relative_path:  # 빈 키 제외
    #                 files.append({
    #                     'path': relative_path,
    #                     'size': obj['Size'],
    #                     'modified': obj['LastModified'],
    #                     's3_key': obj['Key']
    #                 })
            
    #         return files
            
    #     except Exception as e:
    #         logger.error(f"Failed to list files: {e}")
    #         return []
