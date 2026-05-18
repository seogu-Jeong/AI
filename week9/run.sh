#!/bin/bash
# Classical Mechanics Solver — 실행 스크립트
# 사용법: bash run.sh  또는 더블클릭

cd "$(dirname "$0")"

# uv 설치 확인
if ! command -v uv &> /dev/null; then
    echo "uv가 없습니다. 설치합니다..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 의존성 설치
echo "의존성 확인 중..."
uv sync --quiet 2>/dev/null || uv sync

# 포트 8000 사용 중이면 종료
lsof -ti:8000 | xargs kill -9 2>/dev/null

echo ""
echo "========================================"
echo "  Classical Mechanics Solver 시작 중..."
echo "  브라우저: http://localhost:8000"
echo "  종료: Ctrl+C"
echo "========================================"
echo ""

# 브라우저 자동 오픈 (2초 후)
(sleep 2 && open http://localhost:8000) &

# 서버 시작
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
