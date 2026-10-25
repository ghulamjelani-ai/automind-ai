from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    business_name: str | None = None
    industry: str | None = None
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str


class ChatMessageResponse(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    conversation_id: str
    title: str
    messages: list[ChatMessageResponse]