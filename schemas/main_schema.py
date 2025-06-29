from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any


class WikiInput(BaseModel):
    """위키 입력 스키마"""
    project_id: int
    url: str
    updated_at: str
    
    # @validator('content')
    # def validate_content_not_empty(cls, v):
    #     if not v.strip():
    #         raise ValueError("Content cannot be empty")
    #     return v


class MeetingNote(BaseModel):
    """회의록 입력 스키마"""
    project_id: int
    content: str
    position: List[str] = Field(..., description="List of positions to parse tasks for")
    
    @validator('content')
    def validate_content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v
        
    @validator('position')
    def validate_positions(cls, v):
        if not v:
            raise ValueError("At least one position must be provided")
        return v


class InsertInfo(BaseModel):
    user_table_id: int
    content: List[Dict[str, Any]]  # 리스트로 변경!

    @validator('content')
    def validate_content_not_empty(cls, v):
        if not v:  # 빈 리스트 체크
            raise ValueError("Content cannot be empty")
        return v

