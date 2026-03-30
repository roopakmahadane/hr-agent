# hr-agent

An AI-powered multi-agent HR training assistant built with LangGraph. Given a job description and company profile, it autonomously extracts role competencies, researches L&D best practices via Tavily web search, generates a structured 3-module training curriculum using Claude Sonnet, and scores its own output — looping back to improve until the curriculum reaches a quality threshold or a circuit breaker stops it. Built as a portfolio project targeting Builder PM roles at HR tech companies.

---

## Architecture

Supervisor pattern: one routing agent orchestrates five specialist nodes. Every specialist returns to the supervisor after running; the supervisor reads state and decides the next step.

```
                        ┌─────────────┐
              ┌────────►│ jd_analyzer │  extracts 5-6 competencies from JD
              │         └──────┬──────┘
              │                │
              │         ┌──────▼──────┐
              │         │ld_researcher│  Tavily web search (broad or targeted)
              │         └──────┬──────┘
              │                │
              │      ┌─────────▼────────┐
              │      │curriculum_builder│  Claude Sonnet → 3-module curriculum
              │      └─────────┬────────┘
              │                │
              │       ┌────────▼────────┐
              │       │quality_checker  │  Claude Haiku → score 1-10 + feedback
              │       └────────┬────────┘
              │                │
┌─────────────┴──┐    score < 8 and       score >= 8
│   supervisor   │◄── attempts < 3   ─────────────────► END
└────────────────┘         │
                           │ (routes to ld_researcher first for
                           │  targeted research, then curriculum_builder
                           │  rebuilds with enriched inputs)
```

---

## Key Design Decisions

**Supervisor pattern** — a single routing node reads state and decides the next agent. Keeps routing logic in one place and makes the flow easy to trace and debug.

**Company-specific context** — every node receives both `jd_text` and `company_profile`. Competencies, research queries, curriculum structure, and quality scoring are all grounded in the specific company's values and culture, not generic HR advice.

**Improvement loop** — when `quality_score < 8`, the graph routes to `ld_researcher` first (not `curriculum_builder` directly). `ld_researcher` uses the `feedback` from `quality_checker` to run a targeted Tavily search, accumulating new research into state before the curriculum is rebuilt with richer inputs.

**Circuit breaker** — `attempts` is incremented by `curriculum_builder` on every build. When `attempts >= 3`, the supervisor exits regardless of score. Prevents infinite loops and runaway API costs.

**Haiku vs Sonnet** — Claude Haiku handles cheap, structured tasks (competency extraction, quality scoring). Claude Sonnet is used only for curriculum generation where output quality directly affects the product.

---

## Stack

| Layer | Tool |
|---|---|
| Agent orchestration | LangGraph 1.0 |
| LLM provider | langchain-anthropic |
| Routing & extraction | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Curriculum generation | Claude Sonnet (`claude-sonnet-4-6`) |
| Web search | tavily-python |
| State persistence (dev) | LangGraph MemorySaver |
| Environment variables | python-dotenv |

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/roopakmahadane/hr-agent.git
cd hr-agent
```

**2. Create your `.env` file**
```bash
cp .env.example .env
# then edit .env and add your keys
```

`.env` should contain:
```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

**3. Install dependencies**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install langgraph langchain-anthropic tavily-python python-dotenv
```

**4. Run the smoke test**
```bash
python agent.py
```

The `__main__` block runs a real end-to-end pipeline with an Impresario Entertainment & Hospitality JD. Prints the final curriculum, quality score, and number of attempts.

---

## What's Next

- **FastAPI endpoint** — wrap `run()` in a POST endpoint so the pipeline can be called from a frontend or external system (Phase 2)
- **Connect to SkillPath Supabase** — replace the in-memory `MemorySaver` with `PostgresSaver` to persist runs and enable async job tracking
- **Streaming output** — surface curriculum generation progress in real time using LangGraph's streaming API
