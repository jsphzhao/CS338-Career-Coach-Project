# Nova Career Coach Chatbot

Nova is a RAG-enabled career coach that guides users through the DRIVEN week 1 curriculum. It combines:

- **Retrieval-Augmented Generation** over the provided PDFs (`All Videos + Exercises.pdf` and `Copy of Week 1 prompts.pdf`).
- A **conversation flow engine** that enforces the scripted prompts and branching rules from the DRIVEN materials.
- A **FastAPI backend** with session management and an **interactive web UI**.

## Prerequisites

- Python 3.12 (or compatible 3.10+ environment).
- The source PDFs present at the repository root (already provided).
- An OpenAI API key (optional, but required for live LLM responses).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 
> ⚠️ `sentence-transformers` pulls in the CPU build of PyTorch automatically. Expect a larger download on first install.
> 

## Running the app

1. **Set your OpenAI API key** (optional but recommended):

   ```bash
   export OPENAI_API_KEY="sk-..."
   # optionally override the default model
   # export OPENAI_MODEL="gpt-4o-mini"
   ```

2. **Start the FastAPI server**:

   ```bash
   uvicorn app:app --reload
   ```

3. Visit the UI at [`http://localhost:8000`](http://localhost:8000).

On first launch the service automatically builds a vector store under `data/vector_store/` by chunking and embedding the PDFs.

### Offline behaviour

If `OPENAI_API_KEY` is not set, Nova will still walk through the scripted prompts and retrieved context, but returns a friendly notice instead of a personalised LLM reply. Once you supply an API key, the same workflow will deliver full coaching responses.

## Project structure

- `app.py` – FastAPI application, session endpoints, and static file serving.
- `src/rag_utils.py` – PDF ingestion, chunking, and vector store utilities.
- `src/flow_manager.py` – Conversation state machine tying flow steps, RAG, and LLM output together.
- `src/llm_client.py` – Thin wrapper around the OpenAI Responses API with compatibility fallback.
- `flow/week1_flow.json` – Structured representation of the DRIVEN prompts parsed from `Copy of Week 1 prompts.pdf`.
- `static/index.html` – Responsive chat UI.

## Development tips

- **Regenerating the vector store**: delete `data/vector_store/` and restart the server, or call `ensure_vector_store(base_dir, rebuild=True)`.
- **Testing**: use FastAPI's `TestClient` to simulate a chat without running the server (see `app.py` usage in the tests we ran during setup).
- **Styling tweaks**: edit `static/index.html` (CSS + vanilla JS). The UI dynamically renders questions vs. statements and indicates Nova's “thinking” state while awaiting responses.

## Future extensions

- Add authentication for multi-user deployments.
- Persist session transcripts (currently in-memory only).
- Extend the flow manager to support additional DRIVEN weeks by adding new flow JSON files and switching based on session state.

Enjoy building with Nova!
