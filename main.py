"""
FastAPI entrypoint for hr-agent.

Exposes the LangGraph pipeline as an HTTP API.
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import run


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CurriculumRequest(BaseModel):
    jd_text: str
    company_profile: dict
    task: str = "Generate training curriculum"


class CurriculumResponse(BaseModel):
    curriculum: str
    quality_score: int
    attempts: int
    competencies: list[str]
    web_research: list[str]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="hr-agent",
    description="AI-powered HR training curriculum generator using LangGraph.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-curriculum", response_model=CurriculumResponse)
def generate_curriculum(request: CurriculumRequest):
    try:
        result = run(
            jd_text=request.jd_text,
            company_profile=request.company_profile,
            task=request.task,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CurriculumResponse(
        curriculum=result.get("curriculum", ""),
        quality_score=result.get("quality_score", 0),
        attempts=result.get("attempts", 0),
        competencies=result.get("competencies", []),
        web_research=result.get("web_research", []),
    )
