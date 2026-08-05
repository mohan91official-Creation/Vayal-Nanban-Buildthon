# Vayal Nanban · வயல் நண்பன் — citation-first RAG

A bilingual, farmer-first RAG chatbot for the Application Buildthon. It is designed for Tamil Nadu farmers who need clear, source-grounded guidance without navigating a complex technical interface.

## RAG architecture

Every online answer follows a retrieval-augmented generation pipeline:

1. The question is enriched with district, crop, growth stage, irrigation method, and language.
2. `text-embedding-3-small` creates a multilingual query embedding.
3. FAISS retrieves the four closest passages from the bundled, human-reviewed Tamil Nadu agriculture corpus.
4. Retrieved passages, source names, and URLs are inserted into the grounding prompt as `[S1]`–`[S4]`.
5. The model must cite retrieved evidence and say when that evidence is insufficient.
6. A deterministic **Retrieved sources** section is appended so citations cannot disappear from the interface.
7. LangSmith records the retriever run, document IDs, source domains, vector-store type, and retrieval count.

```text
Farmer question + field context
            │
            ▼
 OpenAI multilingual embedding
            │
            ▼
     FAISS vector search ─────► top 4 verified passages
            │                              │
            └──────────────┬───────────────┘
                           ▼
              grounded safety prompt
                           │
                           ▼
             answer + inline citations
                           │
                           ▼
          deterministic source list + trace
```

The corpus lives in `knowledge_base/tamil_nadu_agriculture.json`. It contains concise passages linked to TNAU, Tamil Nadu Agriculture, IMD Chennai, Tamil Nadu Agricultural Marketing, and India 112. It deliberately excludes changing market prices, live forecasts, scheme deadlines, and guessed pesticide doses.

## What makes it farmer friendly

- Tamil-first interface with an English switch
- Four-scene hero carousel with three-second transitions, manual controls, and mobile-safe crops
- District and crop context carried into every answer
- Crop stage and irrigation profile for more relevant guidance
- Multilingual semantic retrieval through a FAISS vector store
- Citation-grounded answers from a bundled, verified agriculture corpus
- One-tap questions for pests, irrigation, markets, and schemes
- Text, voice, and crop-photo input from the same chat box
- Useful offline guidance when no API key is configured
- Urgency-aware field action cards: do today, watch next, and when to get help
- Downloadable field notes that farmers can share with an agriculture officer
- Guardrails against invented weather, market prices, scheme rules, and unsafe pesticide doses
- Optional LangSmith tracing with project grouping and non-identifying farm-context metadata
- Direct links to TNAU Agritech, IMD Chennai, Tamil Nadu Agriculture, Tamil Nadu Agricultural Marketing, and India's 112 emergency service

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

## Enable AI answers, voice transcription, and image guidance

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Add your own OpenAI API key.
3. Restart the app.

Never commit `.streamlit/secrets.toml`.

The optional embedding model setting is:

```toml
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

## Enable LangSmith tracing

1. Create a LangSmith API key in **Settings → API Keys** at `smith.langchain.com`.
2. Add the following values to `.streamlit/secrets.toml`:

```toml
LANGSMITH_TRACING = true
LANGSMITH_API_KEY = "your-langsmith-api-key"
LANGSMITH_PROJECT = "vayal-nanban-buildthon"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_WORKSPACE_ID = ""
```

Restart Streamlit, ask one AI-backed question, and open the
`vayal-nanban-buildthon` project in LangSmith's **Tracing** view. The workspace
ID is needed only when the API key can access multiple workspaces. Traces can
contain prompts, responses, and uploaded crop photographs. Do not enter personal
or financial details, and do not upload identifiable people or documents.

When vector retrieval succeeds, LangSmith shows `vayal_nanban_retriever` and
`vayal_nanban_answer`. The answer run is tagged `rag` and `faiss` and includes
the retrieved document IDs and source domains as non-identifying metadata.

## RAG evaluation questions

- `My paddy field has standing water after heavy rain. What should I do today?`
- `A worker feels dizzy after spraying pesticide.`
- `What should I collect before asking about a drip-irrigation subsidy?`
- `How should I compare two markets before selling paddy?`
- `மஞ்சள் இலைகளில் புள்ளிகள் பரவுகிறது. முதலில் என்ன செய்ய வேண்டும்?`

## Run the checks

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m py_compile streamlit_app.py farmer_assistant.py rag_engine.py upload_safety.py
```

For contributors who prefer pytest, install `requirements-dev.txt` and run
`python -m pytest tests -q`.

## Project structure

- `streamlit_app.py` — farmer-facing interface and conversation flow
- `farmer_assistant.py` — grounded generation, safety policy, LangSmith tracing, voice transcription, and offline answers
- `rag_engine.py` — FAISS vector indexing, context-aware retrieval, and citation formatting
- `knowledge_base/tamil_nadu_agriculture.json` — source-labelled agriculture passages
- `upload_safety.py` — crop-photo validation and safe filename handling
- `assets/vayal-nanban-mark.png` — original Vayal Nanban brand mark
- `static/hero/` — deployment-ready carousel artwork
- `.streamlit/config.toml` — accessible emerald, teal, and harvest visual theme
- `tests/` — core intent, localization, and safety checks

The carousel artwork was supplied by the project owner specifically for the
Vayal Nanban Buildthon application; it is not taken from another participant's app.

## Safety note

The app provides general agricultural guidance. It intentionally avoids presenting unverified live information or guessing chemical doses. Farmers should confirm field-specific treatment with a qualified agriculture officer, KVK specialist, or other local expert.

