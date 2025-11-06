import logging
from pathlib import Path
from typing import Dict, List, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.flow_manager import ConversationManager
from src.llm_client import LLMClient
from src.rag_utils import ensure_vector_store


LOGGER = logging.getLogger("nova.app")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FLOW_PATH = BASE_DIR / "flow" / "week1_flow.json"

app = FastAPI(title="Nova Career Coach")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class Message(BaseModel):
    sender: Literal["Nova", "User"]
    type: Literal["text", "question"]
    text: str


class SessionResponse(BaseModel):
    session_id: str
    messages: List[Message]


class MessageRequest(BaseModel):
    session_id: str
    message: str


class MessageResponse(BaseModel):
    messages: List[Message]


@app.on_event("startup")
async def startup_event() -> None:
    base_path = BASE_DIR
    vector_store = ensure_vector_store(base_path)
    llm_client = LLMClient()
    manager = ConversationManager(FLOW_PATH, vector_store, llm_client)

    app.state.manager = manager
    app.state.sessions: Dict[str, dict] = {}
    LOGGER.info("Application startup complete. Vector store ready with %d chunks.", len(vector_store.documents))


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionResponse)
async def create_session_endpoint() -> SessionResponse:
    manager: ConversationManager = app.state.manager
    session_id = uuid4().hex
    session_state = manager.create_session()
    app.state.sessions[session_id] = session_state
    initial_message = manager.initial_prompt()
    return SessionResponse(
        session_id=session_id,
        messages=[Message(**initial_message)],
    )


@app.post("/api/message", response_model=MessageResponse)
async def message_endpoint(payload: MessageRequest) -> MessageResponse:
    sessions: Dict[str, dict] = app.state.sessions
    manager: ConversationManager = app.state.manager

    session_state = sessions.get(payload.session_id)
    if not session_state:
        raise HTTPException(status_code=404, detail="Session not found")

    replies = manager.handle_user_message(session_state, payload.message)
    return MessageResponse(messages=[Message(**reply) for reply in replies])

