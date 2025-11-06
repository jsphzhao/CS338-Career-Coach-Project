import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from .llm_client import LLMClient
from .rag_utils import RetrievedChunk, VectorStore


LOGGER = logging.getLogger(__name__)


class ConversationManager:
    def __init__(
        self,
        flow_path: Path,
        vector_store: VectorStore,
        llm_client: LLMClient,
    ) -> None:
        self.flow_steps = json.loads(flow_path.read_text())["steps"]
        self.vector_store = vector_store
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Session lifecycle helpers
    # ------------------------------------------------------------------
    def create_session(self) -> Dict[str, Any]:
        return {
            "step_index": 0,
            "awaiting_name": True,
            "profile": {"name": None},
            "answers": {},
            "history": [],
            "pending_question": None,
        }

    def initial_prompt(self) -> Dict[str, str]:
        return {
            "sender": "Nova",
            "type": "question",
            "text": "I'm Nova, your career coach. What name would you like me to use during our chats?",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_user_message(self, session: Dict[str, Any], message: str) -> List[Dict[str, str]]:
        message = message.strip()
        if not message:
            return []

        responses: List[Dict[str, str]] = []

        if session.get("awaiting_name"):
            session["profile"]["name"] = message.strip().split()[0]
            session["awaiting_name"] = False
            session["history"].append({"role": "user", "content": f"My name is {message}."})
            responses.extend(self._advance_flow(session))
            return responses

        pending_question = session.get("pending_question")
        if pending_question:
            question_id = pending_question["id"]
            session["answers"][question_id] = message
            session["history"].append({"role": "user", "content": message})

            coach_reply = self._generate_coach_reply(session, pending_question, message)
            responses.append({"sender": "Nova", "type": "text", "text": coach_reply})
            session["history"].append({"role": "assistant", "content": coach_reply})

            session["pending_question"] = None
            responses.extend(self._advance_flow(session))
            return responses

        # If we reach here, no question was pending; gently remind user.
        responses.append(
            {
                "sender": "Nova",
                "type": "text",
                "text": "Thanks for sharing. I’ll guide you with the next prompt shortly.",
            }
        )
        responses.extend(self._advance_flow(session))
        return responses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _advance_flow(self, session: Dict[str, Any]) -> List[Dict[str, str]]:
        outputs: List[Dict[str, str]] = []
        while session["step_index"] < len(self.flow_steps):
            step = self.flow_steps[session["step_index"]]
            if step["type"] == "print":
                text = self._fill_placeholders(step["text"], session, keep_newlines=True)
                outputs.append({"sender": "Nova", "type": "text", "text": text})
                session["history"].append({"role": "assistant", "content": text})
                session["step_index"] += 1
                continue

            if step["type"] == "question":
                question_text = self._fill_placeholders(step["question"], session)
                question_text = re.sub(r"\s*a\.\s*$", "", question_text)
                outputs.append({"sender": "Nova", "type": "question", "text": question_text})
                session["history"].append({"role": "assistant", "content": question_text})
                session["pending_question"] = step
                session["step_index"] += 1
                break

            session["step_index"] += 1

        return outputs

    def _generate_coach_reply(
        self,
        session: Dict[str, Any],
        question_step: Dict[str, Any],
        user_message: str,
    ) -> str:
        system_prompt = self._build_system_prompt(session, question_step)
        data_required = question_step.get("data_required") or ""
        data_context = self._fill_placeholders(data_required, session, keep_newlines=True)

        retrieval_query = f"{question_step.get('question', '')} {user_message}"
        retrieved_chunks = self.vector_store.retrieve(retrieval_query, k=4)
        context_text = self._format_retrieved_chunks(retrieved_chunks)

        base_instructions = (
            "You are Nova, an empathetic and encouraging career coach supporting adults "
            "who may be experiencing depression or unemployment. Use a validating, strengths-based tone. "
            "Offer concise responses (roughly 2-4 sentences) with practical next steps when helpful. "
            "Do not ask the next scripted question; the system will handle it."
        )

        system_payload = base_instructions + "\n\n" + system_prompt
        if data_context:
            system_payload += "\n\nRefer to these helper notes when relevant:\n" + data_context
        if context_text:
            system_payload += "\n\nCurriculum excerpts:\n" + context_text

        recent_history = session["history"][-8:]
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_payload}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": user_message})

        if not self.llm.is_configured():
            LOGGER.warning("OPENAI_API_KEY not configured; returning fallback response.")
            return (
                "Thanks for walking through the exercise with me. Once an OpenAI API key is configured, "
                "I'll be able to provide tailored coaching feedback here."
            )

        try:
            reply = self.llm.generate(messages)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOGGER.exception("LLM generation failed: %s", exc)
            return (
                "I ran into a technical hiccup generating a detailed response. Could we try again in a moment?"
            )

        return reply.strip()

    def _fill_placeholders(
        self,
        text: str,
        session: Dict[str, Any],
        keep_newlines: bool = False,
    ) -> str:
        replacements = {
            "{name}": session["profile"].get("name") or "friend",
            "{Name}": session["profile"].get("name") or "friend",
            "{Answer to 2 (reason for participating in DRIVEN)}": session["answers"].get("q1", ""),
            "{Response from 2}": session["answers"].get("q1", ""),
            "{Answer to question 4 (goal selected)}": session["answers"].get("q4", ""),
            "{Answer to questions 4 (goal selected)}": session["answers"].get("q4", ""),
            "{Answer to question 6 (steps to achieving goal)}": session["answers"].get("q6", ""),
            "{Answer to questions 6 (steps to achieving goal)}": session["answers"].get("q6", ""),
            "{Answers to questions 7 (expected barriers)}": session["answers"].get("q7", ""),
        }

        for placeholder, value in replacements.items():
            if value:
                text = text.replace(placeholder, value)

        # remove remaining braces but keep descriptive text
        text = re.sub(r"\{([^}]*)\}", r"\1", text)
        if keep_newlines:
            text = re.sub(r"[ \t]{2,}", " ", text)
            return text.strip()
        return re.sub(r"\s+", " ", text).strip()

    def _build_system_prompt(self, session: Dict[str, Any], step: Dict[str, Any]) -> str:
        return self._fill_placeholders(step.get("system_prompt", ""), session, keep_newlines=True)

    @staticmethod
    def _format_retrieved_chunks(chunks: List[RetrievedChunk]) -> str:
        formatted = []
        for chunk in chunks:
            snippet = chunk.content
            formatted.append(f"[Source: {chunk.source}] {snippet}")
        return "\n".join(formatted)

