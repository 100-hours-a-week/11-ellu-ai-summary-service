# test_runpod.py 파일 생성
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="dummy"
)

try:
    print("🔗 Runpod vLLM 서버에 연결 중...")
    
    # 모델 목록 확인
    models = client.models.list()
    print("✅ 연결 성공!")
    print("사용 가능한 모델:", [model.id for model in models.data])
    
    # 채팅 테스트
    response = client.chat.completions.create(
        model="microsoft/DialoGPT-medium",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=50
    )
    print("💬 채팅 응답:", response.choices[0].message.content)
    
except Exception as e:
    print(f"❌ 오류: {e}")
    print("SSH 터널이 실행 중인지 확인하세요!")