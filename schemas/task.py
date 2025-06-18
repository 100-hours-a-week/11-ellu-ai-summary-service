from typing import TypedDict, List, Dict, Optional, Union

class TaskState(TypedDict, total=False):
    meeting_note: str
    project_id: int
    position: List[str]
    prompt: Optional[Dict[str, Union[str, List[Dict[str, str]]]]]
    main_task: Optional[Dict[str, List[str]]]
    AI: Optional[List[str]]
    BE: Optional[List[str]]
    FE: Optional[List[str]]
    CLOUD: Optional[List[str]]
    validation_result: Optional[str]
    feedback: Optional[str]
    error: Optional[str]
    status: Optional[str]
    count: int
    routes :Optional[List[str]]