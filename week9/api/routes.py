import asyncio
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Import solver components
# These are expected to be in the mechanics_solver/solver directory
try:
    from solver import problem_parser, ai_explainer
    from solver.euler_rk4 import solve as euler_rk4_solve
    from solver.planetary import solve as planetary_solve
    from solver.double_pendulum import solve as double_pendulum_solve
    from solver.lagrangian import solve as lagrangian_solve
except ImportError as e:
    logging.error(f"Could not import solver modules: {e}")
    # Define stubs for robustness during development if needed, 
    # but the logic below assumes they exist.

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

async def handle_solve(request: SolveRequest) -> SolveResponse:
    """
    Core logic for parsing, solving, and explaining the mechanics problem.
    """
    params = {}
    steps = []
    
    try:
        # a. Parse problem
        steps.append("Parsing problem input...")
        params = problem_parser.parse(request.problem, request.topic)
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
        explanation = ai_explainer.explain(results, request.topic, params)
        steps.append("Explanation generated.")

        # d. Return SolveResponse
        return SolveResponse(
            plotly_json=results.get("plotly_json") if results else None,
            explanation=explanation,
            parameters=params,
            steps=steps
        )

    except Exception as e:
        logging.exception("Error during solve process")
        return SolveResponse(
            error=str(e),
            parameters=params,
            steps=steps
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
