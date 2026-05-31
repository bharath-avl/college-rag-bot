# CollegeBot: Self-Correcting Multi-Agent RAG with LangGraph & FastAPI

A production-grade, stateful college assistant leveraging LangGraph orchestration, Gemini LLMs, and local vector embeddings to actively grade, rewrite, and guarantee context relevance before serving an answer.

## System Architecture: Self-Correcting LangGraph

```text
               [USER QUERY]
                     │
                     ▼
             [intent_router]
           (Gemini 2.0 Flash)
                     │
                     ▼
               [retriever]
        (MMR Search via ChromaDB)
                     │
                     ▼
           [relevance_grader]  <─────────────┐
           (Gemini 2.0 Flash)                │
                     │                       │
      ┌──────────────┴──────────────┐        │ (Needs Rewrite & loop < 2)
      ▼                             ▼        │
[Score < 0.6]                 [Score ≥ 0.6]  │
      │                             │        │
      ▼                             ▼        │
[query_rewriter]               [generator]   │
(Gemini 2.5 Pro)             (Gemini 2.5 Pro)│
      │                             │        │
      └──────────────┬──────────────┘        │
                     └───────────────────────┘
                                             
                                           [END]
```

## Why This Transcends Basic RAG

Standard RAG systems blindly embed, retrieve, and generate. This architecture assumes initial retrieval might fail and introduces algorithmic safety nets:
* **Intent Routing**: Bypasses expensive reasoning for simple queries, classifying intent via lightweight, low-latency models.
* **Algorithmic Hallucination Grading**: Every retrieved chunk is graded against the user's query by a strict evaluator LLM. If the score falls below a 0.6 threshold, the response is blocked from generation.
* **Self-Correcting Rewrite Loops**: Instead of failing silently on poor context, the system routes rejected queries to a semantic rewriter (Gemini 2.5 Pro), appending necessary keywords to optimize the vector search, and loops back into the retriever up to two times.
* **Zero-Cost Local Embeddings**: BAAI embeddings run locally on Metal (MPS), keeping vectorization fast, private, and free.

## Technology Stack

| Tool / Library | Why It Was Chosen | Alternative Considered |
|---|---|---|
| **LangGraph** | Provides cyclical state management required for self-correction loops. | LangChain Chains (lacks cyclical loops and state checkpointer). |
| **FastAPI** | Async-first design effortlessly handles SSE data streaming for token-by-token generation. | Flask (poor native async/streaming support). |
| **Streamlit** | Rapid frontend prototyping with a highly customizable widget ecosystem. | React / Next.js (higher boilerplate for AI prototyping). |
| **ChromaDB** | Memory-efficient local vector DB that runs fully offline via SQLite bindings. | Pinecone (introduces network latency and API costs). |
| **Gemini 2.0/2.5** | Tiered cost-allocation strategy (Flash for routing/grading, Pro for generation/rewriting). | OpenAI GPT-4o (higher blended costs at scale). |

## Quickstart

```bash
git clone https://github.com/bharath-avl/college-rag-bot.git
cd college-rag-bot
uv sync
```

Create a `.env` file and add your keys:
```env
GOOGLE_API_KEY="your_google_ai_key"
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_key"
LANGCHAIN_PROJECT="college-rag"
```

Start the application stack:
```bash
# Terminal 1: Start the backend API
uv run python main.py

# Terminal 2: Start the frontend UI
uv run streamlit run frontend/app.py
```

## Business Capabilities

* **Instant Knowledge Navigation**: Replaces manual searching through multi-hundred-page college syllabi with instantaneous, cited answers.
* **Contextual Persistence**: Maintains session history seamlessly, allowing follow-up questions to refer back to prior documents or intent branches.
* **Guaranteed Sourcing**: Every answer is hard-linked to the exact document and page number it was extracted from, eliminating uncited assumptions.
* **Dynamic Syllabus Management**: Upload, index, and manage PDF curricula through the UI, dynamically updating the chatbot's knowledge base.

## Observability & LangSmith Traces

Full agentic execution traces are exported to LangSmith to monitor latency, LLM routing decisions, and context grading failures.

![LangSmith Trace Dashboard Placeholder](docs/langsmith-placeholder.png)

## What I Built & Learned

* **Stateful Flow Architecture**: Engineered a fully cyclical LangGraph state machine with an `AsyncSqliteSaver` checkpointer, persisting thread IDs across concurrent sessions.
* **Asynchronous Server-Sent Events**: Implemented robust backend streaming via FastAPI `StreamingResponse` that yields multi-stage LangGraph output directly into the Streamlit UI.
* **Cost-Optimized Model Tiering**: Designed a multi-LLM strategy where cheap classification tasks (routing/grading) run on Gemini Flash, while complex query reformulation runs on Gemini Pro.
* **Advanced Document Chunking**: Leveraged LangChain's recursive splitters combined with character-offset mapping to strictly track page boundaries across raw PyMuPDF extractions.
* **Dependency & Environment Management**: Fully adopted `uv` as the exclusive dependency resolver, accelerating local setup and strictly isolating environment dependencies without relying on standard pip.

## Repository Structure

```text
├── backend/
│   ├── api/          # FastAPI routers and streaming endpoints
│   ├── db/           # SQLite schemas and metadata operations
│   ├── graph/        # LangGraph nodes, edges, state definitions
│   ├── rag/          # PyMuPDF extraction, MMR retrieval, chunking
│   └── utils/        # Config handling and central logging
├── data/             # Local database storage (ChromaDB + SQLite)
├── frontend/         # Streamlit UI, custom CSS, sidebar widgets
├── tests/            # 45 rigorous pytest integration & unit tests
├── main.py           # Uvicorn entry point
└── pyproject.toml    # Environment and dependency definitions via uv
```
