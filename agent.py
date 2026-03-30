"""
HR Training Assistant — multi-agent LangGraph pipeline.

Supervisor pattern:
  supervisor → jd_analyzer → ld_researcher → curriculum_builder → quality_checker
                                                       ↑                  |
                                                       └──────────────────┘
                                            (loop back if quality_score < 8, max 3 attempts)
"""

import json
import operator
import os
import re
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph


load_dotenv()  # load ANTHROPIC_API_KEY and TAVILY_API_KEY from .env
print("[startup] ANTHROPIC_API_KEY", "found" if os.getenv("ANTHROPIC_API_KEY") else "MISSING — check .env")

# Haiku for all supervisor routing decisions — fast and cheap.
# Sonnet is reserved for curriculum generation (quality matters there).
_haiku = ChatAnthropic(model="claude-haiku-4-5-20251001")
_sonnet = ChatAnthropic(model="claude-sonnet-4-6")  # generation only — quality matters here


# ---------------------------------------------------------------------------
# State schema
# All agents read from and write to this shared state.
# Never mutate directly — always return a new dict with updated fields.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    task: str                  # original user request
    jd_text: str               # raw job description text
    company_profile: dict      # company values/culture (passed in at invocation)
    competencies: Annotated[list[str], operator.add]    # competencies extracted from JD; accumulates across agent runs
    web_research: Annotated[list[str], operator.add]    # accumulates across agent runs
    curriculum: str            # generated learning plan
    quality_score: int         # 1-10 score from quality_checker
    feedback: str              # why score < 8; fed back into curriculum_builder
    attempts: int              # safety counter — max 3 to prevent cost explosion
    final_output: str          # polished, user-facing result


# ---------------------------------------------------------------------------
# Supervisor node
# Reads state and returns the name of the next node to run.
# LangGraph uses this return value in the conditional edge to route the graph.
#
# Routing logic (evaluated top-to-bottom — first match wins):
#   1. No competencies yet          → jd_analyzer
#   2. No web research yet          → ld_researcher
#   3. No curriculum yet            → curriculum_builder
#   4. quality_score < 8
#        attempts < 3               → ld_researcher  (targeted search from feedback,
#                                                      then graph will re-route here
#                                                      → curriculum_builder next pass)
#        attempts >= 3              → end             (cost/loop guard)
#   5. quality_score >= 8           → end             (accepted quality)
# ---------------------------------------------------------------------------

def supervisor(state: AgentState) -> str:
    """Decide which agent to run next. Returns a node name or 'end'."""

    # Step 1 — extract competencies from the JD first.
    if not state.get("competencies"):
        return "jd_analyzer"

    # Step 2 — gather L&D research before we can build a curriculum.
    if not state.get("web_research"):
        return "ld_researcher"

    # Step 3 — build the curriculum once we have competencies + research.
    if not state.get("curriculum"):
        return "curriculum_builder"

    # Step 4 — score the curriculum if it hasn't been evaluated yet.
    # quality_score is 0 in initial state; quality_checker always sets it >= 1.
    # So 0 means "built but not yet scored" — send to quality_checker first.
    quality_score = state.get("quality_score", 0)
    attempts = state.get("attempts", 0)

    if quality_score == 0:
        return "quality_checker"

    # Step 5 — quality gate (quality_checker has now run at least once).
    # If score is below threshold AND we still have improvement attempts left,
    # route to ld_researcher for targeted research, then rebuild curriculum.
    if quality_score < 8:
        if attempts >= 3:
            # Safety guard — stop here to prevent infinite loops and runaway costs.
            return "end"
        # ld_researcher reads feedback, adds targeted research, clears curriculum
        # so supervisor re-routes to curriculum_builder on the next pass.
        return "ld_researcher"

    # Step 6 — curriculum passed quality check.
    return "end"


# ---------------------------------------------------------------------------
# jd_analyzer node
# Reads jd_text + company_profile from state and extracts 5-6 competencies
# that are specific to both the role requirements and the company's values/
# culture. Using Haiku here — extraction is a cheap structured task.
#
# Returns {"competencies": [...]} — operator.add accumulates so this list
# is appended to state, not overwritten.
# ---------------------------------------------------------------------------

def jd_analyzer(state: AgentState) -> dict:
    """Extract 5-6 role-and-company-specific competencies from the JD."""

    # Flatten company_profile dict into a readable string for the prompt.
    company_profile = state.get("company_profile", {})
    company_context = "\n".join(
        f"- {k}: {v}" for k, v in company_profile.items()
    )

    prompt = f"""You are an L&D specialist. Read the job description and company profile below.
Extract exactly 5-6 competencies required for success in this role at this specific company.

Rules:
- Each competency must be a short, named skill or capability (e.g. "Data-Driven Decision Making")
- Tailor competencies to both the role requirements AND the company's values/culture
- Return ONLY a plain numbered list, one competency per line, no extra commentary

Company profile:
{company_context}

Job description:
{state['jd_text']}"""

    response = _haiku.invoke([HumanMessage(content=prompt)])

    # Parse the numbered list from the response.
    # Each line looks like "1. Competency Name" or "- Competency Name".
    lines = response.content.strip().splitlines()
    competencies = []
    for line in lines:
        # Strip leading list markers: "1.", "2.", "-", "*"
        cleaned = line.strip().lstrip("0123456789.-* ").strip()
        if cleaned:
            competencies.append(cleaned)

    # Trim to 6 max in case the model returns extras.
    return {"competencies": competencies[:6]}


# ---------------------------------------------------------------------------
# ld_researcher node
# Runs a Tavily web search for L&D best practices.
#
# Two modes — detected by whether feedback is set in state:
#   Initial mode  (feedback is empty):  broad search based on competencies.
#   Improvement mode (feedback is set): targeted search based on feedback,
#       i.e. what quality_checker said was missing. Also clears curriculum so
#       supervisor's step 3 fires on the next call and routes to
#       curriculum_builder for a rebuild with enriched inputs.
#
# web_research uses operator.add as its reducer, so returning a list here
# appends to the existing list rather than replacing it.
# ---------------------------------------------------------------------------

def ld_researcher(state: AgentState) -> dict:
    """Search for L&D best practices; in improvement mode, also clear curriculum."""

    is_improvement_run = bool(state.get("feedback"))

    if is_improvement_run:
        # Targeted search: use feedback to find what was flagged as missing.
        # e.g. feedback = "missing onboarding section" → search for that specifically.
        query = f"L&D best practices: {state['feedback']}"
    else:
        # Initial broad search: cover the full set of competencies.
        query = f"L&D training best practices for: {', '.join(state.get('competencies', []))}"

    # TavilyClient reads TAVILY_API_KEY from the environment (set via load_dotenv).
    # Cap at 3 results — enough signal without bloating the state accumulator.
    try:
        client = TavilyClient()
        response = client.search(query, max_results=3)
        new_research = [r["content"] for r in response.get("results", []) if r.get("content")]
        if not new_research:
            new_research = [f"[Tavily returned no results for: {query}]"]
    except Exception as e:
        # Don't crash the graph on a search failure — return a placeholder so
        # curriculum_builder can still run with whatever research is already in state.
        new_research = [f"[Tavily search failed ({e}): {query}]"]

    result: dict = {"web_research": new_research}

    if is_improvement_run:
        # KEY FIX: clear curriculum so supervisor's step 3 (`if not curriculum`)
        # becomes true again on the next call, routing to curriculum_builder.
        # Without this, supervisor skips step 3 (curriculum is still set),
        # hits step 4 (quality_score still < 8), and loops back here forever.
        result["curriculum"] = ""

    return result


# ---------------------------------------------------------------------------
# curriculum_builder node
# Combines competencies + web_research + company_profile → curriculum.
# Also increments attempts on every build so the quality loop's safety
# guard (attempts >= 3 → end) actually advances toward the limit.
#
# attempts lives here, not in quality_checker, because the guard is meant
# to limit rebuilds. Incrementing at build time means the guard fires even
# if quality_checker were to fail or be skipped.
# ---------------------------------------------------------------------------

def curriculum_builder(state: AgentState) -> dict:
    """Generate the learning curriculum; increment attempts counter."""

    company_profile = state.get("company_profile", {})
    company_context = "\n".join(f"- {k}: {v}" for k, v in company_profile.items())
    competencies = "\n".join(f"- {c}" for c in state.get("competencies", []))
    research = "\n\n".join(state.get("web_research", []))

    # Include feedback only on improvement passes so Sonnet knows what to fix.
    feedback = state.get("feedback", "")
    feedback_section = (
        f"\n\nPREVIOUS QUALITY FEEDBACK — you MUST address these gaps:\n{feedback}"
        if feedback else ""
    )

    prompt = f"""You are a senior L&D designer. Create a structured training curriculum for the role described below.

FORMAT — output exactly 3 modules, each with exactly 4 lessons:

Module 1: [Module Title]
  Lesson 1.1: [Title] — [1-sentence description]
  Lesson 1.2: [Title] — [1-sentence description]
  Lesson 1.3: [Title] — [1-sentence description]
  Lesson 1.4: [Title] — [1-sentence description]

(repeat for Module 2 and Module 3)

INPUTS

Company profile:
{company_context}

Role competencies to cover:
{competencies}

L&D research and best practices:
{research}

Job description context:
{state.get('jd_text', '')}
{feedback_section}

Generate the curriculum now. Be specific to this company and role — no generic content."""

    curriculum = _sonnet.invoke([HumanMessage(content=prompt)]).content

    return {
        "curriculum": curriculum,
        # Increment on every build. First build → attempts=1.
        # Third rejection → attempts=3, which triggers the guard in supervisor.
        "attempts": state.get("attempts", 0) + 1,
        # Reset to 0 so supervisor's step 4 (quality_score == 0) always routes
        # to quality_checker after every rebuild — not just the first one.
        "quality_score": 0,
    }


# ---------------------------------------------------------------------------
# quality_checker node
# Scores the curriculum 1-10 and explains what's missing.
# Uses Haiku — evaluation is a cheap structured task, Sonnet is for generation.
#
# Returns only {"quality_score": int, "feedback": str}.
# supervisor reads quality_score to decide whether to loop or end.
# ld_researcher reads feedback to run a targeted improvement search.
# ---------------------------------------------------------------------------

def quality_checker(state: AgentState) -> dict:
    """Score the curriculum 1-10 and return specific feedback on gaps."""

    company_profile = state.get("company_profile", {})
    company_context = "\n".join(f"- {k}: {v}" for k, v in company_profile.items())
    competencies = "\n".join(f"- {c}" for c in state.get("competencies", []))

    prompt = f"""You are an L&D quality reviewer. Evaluate the training curriculum below.

Score it 1-10 based on:
- Coverage of all required competencies
- Alignment with company values and culture
- Practical applicability and structure
- Specificity (generic advice scores low)

Company profile:
{company_context}

Required competencies:
{competencies}

Curriculum to evaluate:
{state['curriculum']}

Respond with ONLY a JSON object, no markdown, no extra text:
{{"quality_score": <integer 1-10>, "feedback": "<specific description of what is missing or weak>"}}

If the curriculum scores 8 or above, set feedback to an empty string."""

    response = _haiku.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Claude sometimes wraps JSON in ```json ... ``` or adds surrounding text.
    # Try three extraction strategies in order of preference.
    parsed = None

    # Strategy 1: response is already clean JSON.
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip markdown code fences (```json ... ``` or ``` ... ```).
    if parsed is None:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Strategy 3: find the first {...} block anywhere in the response.
    if parsed is None:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Fallback: parsing failed entirely — use a safe default so the pipeline
    # doesn't crash. Score 5 routes back into the improvement loop.
    if parsed is None:
        return {"quality_score": 5, "feedback": f"Could not parse quality response: {raw[:200]}"}

    # Coerce quality_score to int in case Claude returns a string or float.
    try:
        score = int(parsed["quality_score"])
    except (KeyError, ValueError, TypeError):
        score = 5

    feedback = parsed.get("feedback", "")

    return {"quality_score": score, "feedback": feedback}


# ---------------------------------------------------------------------------
# Graph wiring
#
# Pattern: every specialist node flows back to supervisor after running.
# Supervisor uses a conditional edge to read its own return value and route
# to the correct next node — or END.
#
# Topology:
#   supervisor ──(conditional)──► jd_analyzer
#                              ├─► ld_researcher
#                              ├─► curriculum_builder
#                              ├─► quality_checker
#                              └─► END
#   jd_analyzer       ──────────► supervisor
#   ld_researcher     ──────────► supervisor
#   curriculum_builder ─────────► supervisor
#   quality_checker   ──────────► supervisor
#
# MemorySaver keeps state in memory for this dev build.
# Swap for PostgresSaver when connecting to SkillPath Supabase.
# ---------------------------------------------------------------------------

def _supervisor_node(_: AgentState) -> dict:
    # Nodes must return a dict. Supervisor makes no state changes — it only
    # routes. The actual routing decision is made by supervisor() below,
    # which is passed as the routing function to add_conditional_edges.
    return {}


_builder = StateGraph(AgentState)

# Register nodes.
_builder.add_node("supervisor", _supervisor_node)
_builder.add_node("jd_analyzer", jd_analyzer)
_builder.add_node("ld_researcher", ld_researcher)
_builder.add_node("curriculum_builder", curriculum_builder)
_builder.add_node("quality_checker", quality_checker)

# Supervisor is the entry point — graph starts here on every invocation.
_builder.set_entry_point("supervisor")

# Conditional edge from supervisor: its return value IS the next node name.
# The path_map tells LangGraph which string values are valid destinations.
_builder.add_conditional_edges(
    "supervisor",
    supervisor,  # called to get the routing decision
    {
        "jd_analyzer": "jd_analyzer",
        "ld_researcher": "ld_researcher",
        "curriculum_builder": "curriculum_builder",
        "quality_checker": "quality_checker",
        "end": END,  # LangGraph's built-in terminal node
    },
)

# Every specialist returns to supervisor after running.
for node in ("jd_analyzer", "ld_researcher", "curriculum_builder", "quality_checker"):
    _builder.add_edge(node, "supervisor")

# Compile with MemorySaver so state persists across streamed steps (dev only).
graph = _builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Invoke helper
# Wraps graph.invoke with a thread_id so MemorySaver can track the run.
# Pass a partial AgentState — missing fields default to empty/zero.
# ---------------------------------------------------------------------------

def run(jd_text: str, company_profile: dict, task: str = "Generate training curriculum") -> dict:
    """Invoke the full pipeline and return the final state."""

    initial_state: AgentState = {
        "task": task,
        "jd_text": jd_text,
        "company_profile": company_profile,
        "competencies": [],
        "web_research": [],
        "curriculum": "",
        "quality_score": 0,
        "feedback": "",
        "attempts": 0,
        "final_output": "",
    }

    # thread_id scopes the MemorySaver checkpoint — unique per call so runs
    # never share or pollute each other's state.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return graph.invoke(initial_state, config=config)


# ---------------------------------------------------------------------------
# Manual smoke test — only runs when executing agent.py directly.
# Replace jd_text and company_profile with real data to test end-to-end.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run(
        jd_text=(
            "We are hiring a Senior HR Business Partner for Impresario Entertainment & Hospitality "
            "(SOCIAL, Smoke House Deli, 60+ outlets across India).\n\n"
            "Responsibilities:\n"
            "- Partner with restaurant GMs on talent strategy\n"
            "- Drive competency-based hiring across all outlets\n"
            "- Design and implement L&D programs for 500+ staff\n"
            "- Lead performance management cycles\n"
            "- Build succession pipelines for key F&B roles\n"
            "- Manage employee relations across multiple locations\n\n"
            "Requirements:\n"
            "- 5+ years HR experience in hospitality or F&B\n"
            "- Strong stakeholder management skills\n"
            "- Experience with high-volume, frontline workforce\n"
            "- Data-driven approach to HR decisions"
        ),
        company_profile={
            "name": "Impresario Entertainment & Hospitality",
            "brands": "SOCIAL, Smoke House Deli",
            "industry": "Restaurant & Hospitality",
            "size": "500+ staff, 60+ outlets across India",
            "values": "Creativity, Warmth, Operational Excellence, Community",
        },
    )
    print("Final curriculum:\n", result.get("curriculum"))
    print("Quality score:", result.get("quality_score"))
    print("Attempts:", result.get("attempts"))
