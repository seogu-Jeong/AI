import os
import logging
from typing import Dict, Any, Tuple

# Try to import anthropic, but don't crash if it's not installed
try:
    import anthropic
except ImportError:
    anthropic = None

from solver.template_explainer import explain as template_explain

DEFAULT_EXPLANATIONS = {
    'euler_rk4': "오일러 및 RK4 방법을 이용한 조화 진동자 시뮬레이션 결과입니다. 수치적 안정성과 에너지 보존 특성을 확인하세요.",
    'planetary': "케플러 궤도 시뮬레이션 결과입니다. 행성의 타원 궤도와 면적 속도 일정의 법칙을 분석합니다.",
    'double_pendulum': "이중 진동자의 혼돈(Chaos) 현상 분석 결과입니다. 초기 조건에 대한 민감성과 비선형 역학의 특징을 보여줍니다.",
    'lagrangian': "라그랑주 역학을 이용한 단진동 분석 결과입니다. 일반화 좌표와 오일러-라그랑주 방정식을 통한 해법을 설명합니다."
}

SYSTEM_PROMPT = r"""
당신은 대한민국 유수의 대학에서 전산물리학을 가르치는 교수입니다. 학부 2학년 수준의 고전역학 지식을 가진 학생들에게 시뮬레이션 결과를 전문적이고 명확하게 설명하는 역할을 수행합니다.

### 지침:
- **전문성**: 물리학 용어를 정확하게 사용하며, 수식은 LaTeX 인라인 형식을 사용합니다 (예: $F = ma$, $\ddot{\theta} = -(g/L)\sin\theta$).
- **언어**: 한국어로 설명합니다. 정중하고 학구적인 어조를 유지합니다.
- **구조**: 반드시 다음 구조를 따릅니다:
  ## [제목]
  **핵심 물리**: 해당 문제의 주요 물리적 원리 설명
  **수치 방법**: 사용된 수치 해석 기법의 특징과 한계
  **결과 분석**: 시뮬레이션 데이터가 보여주는 구체적인 현상 분석
  **물리적 의미**: 이 결과가 시사하는 물리학적 통찰
- **제한**: 400단어 이내로 간결하면서도 깊이 있게 작성합니다.

... (rest of prompt)
"""

def is_api_key_valid(api_key: str) -> bool:
    """Check if the API key looks valid."""
    if not api_key:
        return False
    if api_key == "your_key_here":
        return False
    return api_key.startswith(('sk-', 'ghp')) or len(api_key) > 20

def explain(results: Dict[str, Any], topic: str, params: Dict[str, Any]) -> Tuple[str, bool]:
    """
    Generates a professional Korean physics explanation of computed results.
    Tries template-based explainer first, falls back to Claude API.
    Returns (explanation, api_used).
    """
    # Always try template first (free, instant, consistent quality)
    try:
        explanation = template_explain(results, topic, params)
        if explanation:
            return explanation, False
    except Exception as e:
        logging.warning(f"Template explanation failed: {e}")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    default_text = DEFAULT_EXPLANATIONS.get(topic, "시뮬레이션 분석 결과입니다.")
    
    if not is_api_key_valid(api_key) or anthropic is None:
        return default_text, False

    # Build a concise results summary (max 300 chars)
    numerical_summary = results.get('numerical_summary', {})
    summary_parts = []
    for key, val in list(numerical_summary.items())[:5]: # Take first 5 key values
        if isinstance(val, (int, float)):
            summary_parts.append(f"{key}: {val:.4g}")
        else:
            summary_parts.append(f"{key}: {val}")
    
    results_summary = ", ".join(summary_parts)[:300]

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nParameters: {params}\nResults: {results_summary}\nExplain in Korean."
                }
            ]
        )
        
        return response.content[0].text.strip(), True
        
    except Exception as e:
        logging.warning(f"Claude API explanation failed: {e}")
        # On error, return default Korean explanation string
        return default_text, False
