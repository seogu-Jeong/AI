import sys
import os
import subprocess

def ensure_packages():
    """누락된 패키지 자동 설치"""
    required_packages = ["fastapi", "uvicorn", "jinja2"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"📦 패키지 자동 설치 중: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

ensure_packages()

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# 경로 설정 (실행 위치 상관없이 절대 경로 추적)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(title="PNU Physics (Velvet Edition)")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/faculty")
async def faculty(request: Request):
    return templates.TemplateResponse("faculty.html", {"request": request})

@app.get("/research")
async def research(request: Request):
    return templates.TemplateResponse("research.html", {"request": request})

@app.get("/admissions")
async def admissions(request: Request):
    return templates.TemplateResponse("admissions.html", {"request": request})

@app.get("/academics")
async def academics(request: Request):
    return templates.TemplateResponse("admissions.html", {"request": request})

@app.get("/community")
async def community(request: Request):
    return templates.TemplateResponse("admissions.html", {"request": request})

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🍷 고급 벨벳 테마 PNU Physics 웹서버 가동 (week2-1)")
    print("👉 접속 주소: http://127.0.0.1:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
