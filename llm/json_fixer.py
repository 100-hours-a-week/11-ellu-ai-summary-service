import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in .env")

llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)


class TaskJson(BaseModel):
    task: str
    subtasks: list[str]


class JsonFixer:
    def __init__(self, model_name="gpt-4o"):
        self.model = llm

    def fix_json(self, raw_text: str) -> TaskJson:
        messages = [
            SystemMessage(content="다음은 JSON 형식이 망가진 응답입니다. 이를 파싱 가능한 JSON으로 수정해서 제공하세요. 설명 없이 JSON만 출력하세요."),
            HumanMessage(content=raw_text)
        ]

        response = self.model.invoke(messages)
        json_str = response.content.strip().removeprefix("```json").removesuffix("```")
        parsed = json.loads(json_str)

        return TaskJson(**parsed)
