import os
import asyncio
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Import solver components
try:
    from solver import problem_parser, ai_explainer, rule_parser
    from solver.euler_rk4 import solve as euler_rk4_solve
    from solver.planetary import solve as planetary_solve
    from solver.double_pendulum import solve as double_pendulum_solve
    from solver.lagrangian import solve as lagrangian_solve
except ImportError as e:
    logging.error(f"Could not import solver modules: {e}")

router = APIRouter()

class SolveRequest(BaseModel):
    problem: str
    topic: str

class SolveResponse(BaseModel):
    plotly_json: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    parameters: Dict[str, Any] = {}
    steps: List[str] = []
    error: Optional[str] = None
    api_used: bool = False
    api_needed: bool = False

class ApiKeyRequest(BaseModel):
    key: str

async def handle_solve(request: SolveRequest) -> SolveResponse:
    """
    Core logic for parsing, solving, and explaining the mechanics problem.
    """
    params = {}
    steps = []
    api_used_total = False
    api_needed = False
    
    try:
        # 0. Check if API is needed (confidence < 0.7)
        _, confidence = rule_parser.extract(request.problem, request.topic)
        if confidence < 0.7:
            api_needed = True

        # a. Parse problem
        steps.append("Parsing problem input...")
        params, api_used_parser = problem_parser.parse(request.problem, request.topic)
        if api_used_parser:
            api_used_total = True
        steps.append("Problem parsed successfully.")

        # b. Dispatch to correct solver based on topic
        results = None
        if request.topic == 'euler_rk4':
            results = euler_rk4_solve(params)
            steps.append("Euler/RK4 solver completed.")
        elif request.topic == 'planetary':
            results = planetary_solve(params)
            steps.append("Planetary motion solver completed.")
        elif request.topic == 'double_pendulum':
            results = double_pendulum_solve(params)
            steps.append("Double pendulum solver completed.")
        elif request.topic == 'lagrangian':
            results = lagrangian_solve(params)
            steps.append("Lagrangian mechanics solver completed.")
        else:
            return SolveResponse(
                error=f"Unsupported topic: {request.topic}",
                parameters=params,
                steps=steps
            )

        # c. Call ai_explainer.explain(results, topic, params)
        steps.append("Generating AI explanation...")
        explanation, api_used_explainer = ai_explainer.explain(results, request.topic, params)
        if api_used_explainer:
            api_used_total = True
        steps.append("Explanation generated.")

        # d. Return SolveResponse
        return SolveResponse(
            plotly_json=results.get("plotly_json") if results else None,
            explanation=explanation,
            parameters=params,
            steps=steps,
            api_used=api_used_total,
            api_needed=api_needed
        )

    except Exception as e:
        logging.exception("Error during solve process")
        return SolveResponse(
            error=str(e),
            parameters=params,
            steps=steps,
            api_used=api_used_total,
            api_needed=api_needed
        )

@router.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest):
    try:
        # 30 second timeout using asyncio
        return await asyncio.wait_for(handle_solve(request), timeout=30.0)
    except asyncio.TimeoutError:
        return SolveResponse(error="Request timed out after 30 seconds")
    except Exception as e:
        return SolveResponse(error=str(e))

@router.post("/set-api-key")
async def set_api_key(request: ApiKeyRequest):
    """Sets the Anthropic API key for the current process."""
    os.environ['ANTHROPIC_API_KEY'] = request.key
    return {"success": True}
