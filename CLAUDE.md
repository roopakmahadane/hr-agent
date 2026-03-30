# hr-agent

AI-powered multi-agent HR training assistant using LangGraph.
Supervisor pattern with specialist agents that generate company-specific L&D curricula from job descriptions.
This is a portfolio project — part of a 90-day learning sprint targeting Builder PM roles at HR tech companies.

## Stack
- Python 3.14
- LangGraph 1.0 — agent orchestration
- langchain-anthropic — Claude Haiku (routing) + Claude Sonnet (generation)
- tavily-python — web search for L&D best practices
- FastAPI — API endpoint (added in Phase 2)
- python-dotenv — environment variables

## Architecture decisions (already locked — do not change)
- Supervisor pattern: one router agent, specialist sub-agents
- State is TypedDict — never mutate directly, always return new values
- Claude Haiku for supervisor routing (fast, cheap)
- Claude Sonnet for curriculum generation (quality matters)
- Quality loop: score < 8 → loop back to curriculum_builder with feedback
- Max 3 attempts guard on quality loop (prevents infinite loops + cost explosion)
- MemorySaver for now (dev) — PostgresSaver when connecting to SkillPath Supabase later

## Agent structure
- supervisor — reads state, decides next agent, routes via conditional edge
- jd_analyzer — extracts competencies from job description text
- ld_researcher — Tavily web search for training best practices
- curriculum_builder — combines competencies + research + company profile → curriculum
- quality_checker — Claude Haiku scores curriculum 1-10 with specific feedback

## State schema (do not add fields without asking)
- task: str — original user request
- jd_text: str — raw job description
- company_profile: dict — company values/culture (passed in, not fetched)
- competencies: list[str] — extracted from JD (accumulator field)
- web_research: list[str] — from Tavily
- curriculum: str — generated learning plan
- quality_score: int — 1-10
- feedback: str — why score < 8, used in improvement loop
- attempts: int — safety counter, max 3
- final_output: str — user-facing result

## Environment variables (in .env, never hardcode)
- ANTHROPIC_API_KEY
- TAVILY_API_KEY

## Commands
- Run: python agent.py
- Install: pip install langgraph langchain-anthropic tavily-python python-dotenv
- Freeze: pip freeze > requirements.txt

## Rules
- Never hardcode API keys
- Always return state updates, never mutate state directly
- Every conditional edge MUST have a path to END
- Add attempts guard before any loop (if attempts >= 3: return "end")
- Use Haiku for cheap decisions, Sonnet only for generation
- One file for now (agent.py) — no premature splitting
- Explain every decision in comments — this is a learning project