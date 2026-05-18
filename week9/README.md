# Classical Mechanics Solver — Week 9 과제

## Description
이 프로젝트는 고전 역학의 복잡한 문제들을 수치 해석적으로 해결하고, 그 결과를 시각화하며 AI를 통해 물리적 의미를 설명해주는 통합 솔루션입니다. 4가지 주요 물리 주제에 대해 정밀한 계산을 수행하며, 대화형 Plotly 그래프를 통해 동적인 분석이 가능합니다.

- **4가지 주요 주제**: Euler/RK4 수치 해석, 행성 운동, 이중 진동자, 라그랑주 역학
- **AI 설명 기능**: 수치 데이터와 파라미터를 분석하여 물리적 통찰력을 제공하는 AI Explainer 탑재
- **Plotly 시각화**: 고정밀 데이터를 기반으로 한 인터랙티브 웹 그래프 제공

## Tech Stack
| Category | Technology |
| :--- | :--- |
| **Backend** | FastAPI (Python 3.12) |
| **Frontend** | Vanilla JS, HTML5, CSS3, Plotly.js |
| **Solvers** | NumPy, SciPy |
| **AI Integration** | Anthropic Claude API (AI Explainer) |
| **Package Management** | uv (modern Python packaging) |

## Quick Start
```bash
cd mechanics_solver
cp .env.example .env   # ANTHROPIC_API_KEY 입력
uv sync
uv run uvicorn main:app --reload --port 8000
# → http://localhost:8000 접속
```

## API Usage Example
서버가 실행 중일 때, 다음과 같이 직접 API를 호출하여 결과를 얻을 수 있습니다.

```bash
curl -X POST "http://localhost:8000/solve" \
     -H "Content-Type: application/json" \
     -d '{
       "topic": "euler_rk4",
       "problem": "Solve harmonic oscillator with k=10, m=2, initial x=1"
     }'
```

## Architecture
```text
┌──────────────────────────┐      ┌──────────────────────────┐
│        Frontend          │      │         Backend          │
│ (HTML/JS + Plotly.js)    │ <──> │        (FastAPI)         │
└──────────────────────────┘      └────────────┬─────────────┘
                                               │
                       ┌───────────────────────┴──────────────────────┐
                       │                                              │
           ┌───────────▼───────────┐                      ┌───────────▼───────────┐
           │        Solver         │                      │      AI Explainer     │
           │ (NumPy/SciPy Physics) │                      │   (LLM Integration)   │
           └───────────────────────┘                      └───────────────────────┘
```

## Supported Topics
1. **Euler/RK4 Methods**: 조화 진동자 문제를 통해 오일러 방법과 4차 룬게-쿠타 방법의 정확도 및 에너지 보존 성능을 비교 분석합니다.
2. **Planetary Motion**: 중력 법칙에 따른 행성 및 위성의 궤도 운동을 시각화하고 궤도 이심률 및 주기를 계산합니다.
3. **Double Pendulum**: 초기 조건에 극도로 민감한 카오스 시스템인 이중 진동자의 복잡한 궤적을 시뮬레이션합니다.
4. **Lagrangian Mechanics**: 일반화된 좌표계를 사용하는 라그랑주 역학을 통해 복잡한 제약 조건이 있는 시스템의 운동 방정식을 유도하고 해결합니다.

## Token Efficiency
이 프로젝트는 **"Python-computes-AI-explains"** 접근 방식을 사용합니다. 
- 복잡한 수치 적분과 대량의 시뮬레이션 데이터 생성은 로컬 Python(NumPy) 환경에서 수행됩니다.
- AI(LLM)는 원시 데이터를 모두 받는 대신, 가공된 요약 결과와 물리적 파라미터만을 전달받아 분석합니다.
- 이를 통해 토큰 사용량을 최소화하면서도 정확하고 통찰력 있는 물리적 설명을 제공합니다.

## Assignment Context
- **Course**: PNU 전산물리 (Computational Physics)
- **Period**: Week 9 과제
- **Focus**: AI-Assisted Mechanics Problem Solving & Visualization
