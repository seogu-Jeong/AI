#!/bin/bash
cd backend && pip install fastapi uvicorn yfinance pandas numpy -q && uvicorn main:app --reload --port 8000 &
cd frontend && npm install && npm run dev
