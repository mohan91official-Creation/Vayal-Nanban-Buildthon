# Vayal Nanban · வயல் நண்பன்

A bilingual, farmer-first chatbot for the Application Buildthon. It is designed for Tamil Nadu farmers who need clear guidance without navigating a complex technical interface.

## What makes it farmer friendly

- Tamil-first interface with an English switch
- Four-scene hero carousel with three-second transitions, manual controls, and mobile-safe crops
- District and crop context carried into every answer
- Crop stage and irrigation profile for more relevant guidance
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

## Run the checks

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python -m py_compile streamlit_app.py farmer_assistant.py upload_safety.py
```

For contributors who prefer pytest, install `requirements-dev.txt` and run
`python -m pytest tests -q`.

## Project structure

- `streamlit_app.py` — farmer-facing interface and conversation flow
- `farmer_assistant.py` — AI prompt, safety policy, voice transcription, and offline answers
- `upload_safety.py` — crop-photo validation and safe filename handling
- `assets/vayal-nanban-mark.png` — original Vayal Nanban brand mark
- `static/hero/` — deployment-ready carousel artwork
- `.streamlit/config.toml` — accessible emerald, teal, and harvest visual theme
- `tests/` — core intent, localization, and safety checks

The carousel artwork was supplied by the project owner specifically for the
Vayal Nanban Buildthon application; it is not taken from another participant's app.

## Safety note

The app provides general agricultural guidance. It intentionally avoids presenting unverified live information or guessing chemical doses. Farmers should confirm field-specific treatment with a qualified agriculture officer, KVK specialist, or other local expert.
