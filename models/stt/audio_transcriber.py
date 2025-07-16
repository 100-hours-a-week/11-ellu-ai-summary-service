import requests
import logging
from app.config import GEMINI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiSTT:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-latest:generateContent"

    def run_stt_and_return_text(self, file_path: str, project_id: int) -> dict:
        """주어진 오디오 파일 경로와 프로젝트 아이디를 받아 Gemini 2.5 Flash-Lite API로 변환하여 텍스트와 프로젝트 아이디 반환"""
        try:
            logger.info(f"Transcribing audio file with Gemini: {file_path}")
            with open(file_path, "rb") as f:
                audio_data = f.read()
            import base64
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "audio/mp3",
                                    "data": audio_b64
                                }
                            }
                        ]
                    }
                ]
            }
            headers = {"Content-Type": "application/json"}
            params = {"key": self.api_key}
            response = requests.post(self.api_url, headers=headers, params=params, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"Gemini STT result: {text[:50]}...")
            return {"project_id": project_id, "text": text}
        except Exception as e:
            logger.error(f"Gemini STT 오류: {e}")
            return {"project_id": project_id, "text": ""}
