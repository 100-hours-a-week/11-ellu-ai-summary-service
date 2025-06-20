# 1. Python requests 라이브러리 사용
import requests
import json

def test_with_requests():
    """Python requests로 POST 요청"""
    url = "http://localhost:8080/projects/11/notes"  # 또는 실제 서버 주소
    
    payload = {
        "project_id": 11,
        "content": """오늘 회의 결과:
        1. AI팀은 사용자 행동 분석 모델 개발 및 추천 시스템 구현
        2. 백엔드팀은 사용자 인증 API 개발하고 데이터베이스 최적화
        3. 프론트엔드팀은 대시보드 UI 개선 및 모바일 반응형 구현  
        4. 클라우드팀은 서버 배포 자동화와 모니터링 시스템 구축""",
        "position": ["AI", "BE", "FE"]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 202:  # 성공
            result = response.json()
            print(f"✅ 성공: {result}")
        else:
            print(f"❌ 실패: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 오류: {e}")

# 2. httpx 라이브러리 사용 (비동기)
import httpx
import asyncio

async def test_with_httpx():
    """httpx로 비동기 POST 요청"""
    url = "http://localhost:8080/projects/11/notes"
    
    payload = {
        "project_id": 11,
        "content": """오늘 회의 결과:
        1. AI팀은 사용자 행동 분석 모델 개발 및 추천 시스템 구현
        2. 백엔드팀은 사용자 인증 API 개발하고 데이터베이스 최적화
        3. 프론트엔드팀은 대시보드 UI 개선 및 모바일 반응형 구현  
        4. 클라우드팀은 서버 배포 자동화와 모니터링 시스템 구축""",
        "position": ["AI", "BE", "FE"]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 202:
                result = response.json()
                print(f"✅ 성공: {result}")
            else:
                print(f"❌ 실패: {response.status_code}")
                
    except Exception as e:
        print(f"❌ 요청 오류: {e}")

# 3. urllib 사용 (표준 라이브러리)
import urllib.request
import urllib.parse
import json

def test_with_urllib():
    """urllib로 POST 요청"""
    url = "http://localhost:8080/projects/11/notes"
    
    payload = {
        "project_id": 11,
        "content": """오늘 회의 결과:
        1. AI팀은 사용자 행동 분석 모델 개발 및 추천 시스템 구현
        2. 백엔드팀은 사용자 인증 API 개발하고 데이터베이스 최적화
        3. 프론트엔드팀은 대시보드 UI 개선 및 모바일 반응형 구현  
        4. 클라우드팀은 서버 배포 자동화와 모니터링 시스템 구축""",
        "position": ["AI", "BE", "FE"]
    }
    
    try:
        # JSON 데이터 준비
        json_data = json.dumps(payload).encode('utf-8')
        
        # 요청 생성
        req = urllib.request.Request(
            url,
            data=json_data,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            method='POST'
        )
        
        # 요청 보내기
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode('utf-8')
            print(f"Status Code: {response.getcode()}")
            print(f"Response: {response_data}")
            
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP 오류: {e.code} - {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"❌ 요청 오류: {e}")

# 4. FastAPI 테스트 클라이언트 사용
from fastapi.testclient import TestClient
# from app.main import app  # 실제 FastAPI 앱 import

def test_with_fastapi_client():
    """FastAPI TestClient로 테스트"""
    # client = TestClient(app)
    
    payload = {
        "project_id": 11,
        "content": """오늘 회의 결과:
        1. AI팀은 사용자 행동 분석 모델 개발 및 추천 시스템 구현
        2. 백엔드팀은 사용자 인증 API 개발하고 데이터베이스 최적화
        3. 프론트엔드팀은 대시보드 UI 개선 및 모바일 반응형 구현  
        4. 클라우드팀은 서버 배포 자동화와 모니터링 시스템 구축""",
        "position": ["AI", "BE", "FE"]
    }
    
    # response = client.post("/projects/11/notes", json=payload)
    # print(f"Status Code: {response.status_code}")
    # print(f"Response: {response.json()}")

# 5. 실제 테스트 실행 함수
def run_api_test():
    """API 테스트 실행"""
    print("🧪 API 테스트 시작")
    print("="*50)
    
    # 1. requests 테스트
    print("\n1️⃣ requests 라이브러리 테스트:")
    test_with_requests()
    
    # 2. urllib 테스트
    print("\n2️⃣ urllib 테스트:")
    test_with_urllib()
    
    # 3. httpx 비동기 테스트
    print("\n3️⃣ httpx 비동기 테스트:")
    asyncio.run(test_with_httpx())

# curl 명령어 생성 함수
def generate_curl_command():
    """curl 명령어 생성"""
    payload = {
        "project_id": 11,
        "content": """오늘 회의 결과:
        1. AI팀은 사용자 행동 분석 모델 개발 및 추천 시스템 구현
        2. 백엔드팀은 사용자 인증 API 개발하고 데이터베이스 최적화
        3. 프론트엔드팀은 대시보드 UI 개선 및 모바일 반응형 구현  
        4. 클라우드팀은 서버 배포 자동화와 모니터링 시스템 구축""",
        "position": ["AI", "BE", "FE"]
    }
    
    json_str = json.dumps(payload, ensure_ascii=False, indent=2)
    
    curl_command = f"""curl -X POST "http://localhost:8080/projects/11/notes" \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{json_str}'"""
    
    print("📋 curl 명령어:")
    print(curl_command)
    
    # Windows용 (PowerShell)
    powershell_command = f"""Invoke-RestMethod -Uri "http://localhost:800/projects/11/notes" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{json_str}'"""
    
    print("\n📋 PowerShell 명령어:")
    print(powershell_command)

if __name__ == "__main__":
    # curl 명령어 출력
    generate_curl_command()
    
    print("\n" + "="*50)
    
    # 실제 API 테스트 (서버가 실행 중일 때)
    # run_api_test()
    
    print("\n💡 사용법:")
    print("1. 서버를 먼저 실행하세요: uvicorn app.main:app --host 0.0.0.0 --port 8080")
    print("2. run_api_test() 주석을 해제하고 실행하세요")
    print("3. 또는 위의 curl 명령어를 터미널에서 실행하세요")