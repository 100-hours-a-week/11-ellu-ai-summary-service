import logging
from app.config import GEMINI_API_KEY
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiSTT:
    def __init__(self):
        self.api_key = GEMINI_API_KEY

    def run_stt_and_return_text(self, file_path: str, project_id: int) -> dict:
        """주어진 오디오 파일 경로와 프로젝트 아이디를 받아 텍스트와 프로젝트 아이디 반환"""
        try:
            logger.info(f"Transcribing audio file with Gemini: {file_path}")
            with open(file_path, "rb") as f:
                audio_data = f.read()
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = 'Generate a transcript of the speech.'
            response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=[prompt, types.Part.from_bytes(data=audio_data, mime_type='audio/mp3')]
            )
            result = response.candidates[0].content.parts[0].text
            logger.info(f"Gemini STT result: {result[:50]}...")
            return {"project_id": project_id, "text": result}        
        except Exception as e:
            logger.error(f"Gemini STT 오류: {e}")
            return {"project_id": project_id, "text": ""}
