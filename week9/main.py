import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from api.routes import router

# Load environment variables
load_dotenv()

app = FastAPI(title="Classical Mechanics Solver")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# Using absolute path for robustness or relative if standard
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Include routes
app.include_router(router)

@app.get("/")
async def read_index(request: Request):
    # Ensure index.html exists or handle missing
    return templates.TemplateResponse(request, "index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}
