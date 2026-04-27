import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 1. FastAPI App Setup
app = FastAPI(title="Quantum AI - Data Preprocessing Lab")

# 2. Logic: Normalization (Min-Max Scaling)
class PreprocessingModel:
    def __init__(self):
        self.raw_data = None
        self.norm_data = None
        self.stats = {}

    def process(self, n_samples=50, salary_range=(30, 100), age_range=(20, 60)):
        np.random.seed(42)
        # 단위를 백만원 단위로 조정하여 입력 받음
        salary = np.random.uniform(salary_range[0]*1e6, salary_range[1]*1e6, n_samples)
        age = np.random.uniform(age_range[0], age_range[1], n_samples)
        
        self.raw_data = {"salary": salary.tolist(), "age": age.tolist()}
        
        # Min-Max Scaling
        s_min, s_max = salary.min(), salary.max()
        a_min, a_max = age.min(), age.max()
        
        s_norm = (salary - s_min) / (s_max - s_min)
        a_norm = (age - a_min) / (a_max - a_min)
        
        self.norm_data = {"salary": s_norm.tolist(), "age": a_norm.tolist()}
        self.stats = {
            "s_min": float(s_min), "s_max": float(s_max),
            "a_min": float(a_min), "a_max": float(a_max)
        }
        
        return {
            "raw": self.raw_data,
            "norm": self.norm_data,
            "stats": self.stats
        }

model = PreprocessingModel()

class ProcessRequest(BaseModel):
    n_samples: int = 50
    salary_min: float = 30
    salary_max: float = 100
    age_min: float = 20
    age_max: float = 60

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = r"""
    <!DOCTYPE html>
    <html lang="ko" class="antialiased">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Preprocessing | Min-Max Scaling</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <script>
            window.MathJax = { tex: { inlineMath: [['$', '$']] } };
        </script>
        <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Fira+Code:wght@400;500&family=Noto+Sans+KR:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            :root { --bg: #030712; --panel: rgba(17, 24, 39, 0.7); --accent: #f59e0b; }
            body { font-family: 'Inter', 'Noto Sans KR', sans-serif; background: var(--bg); color: #f3f4f6; }
            .dashboard-panel { background: var(--panel); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); border-radius: 1.5rem; }
            .kpi-card { background: rgba(31, 41, 55, 0.5); border-left: 4px solid var(--accent); transition: all 0.3s; }
            input[type=range] { accent-color: var(--accent); }
        </style>
    </head>
    <body class="min-h-screen flex flex-col p-4 md:p-8">
        <nav class="max-w-7xl mx-auto w-full mb-8 flex items-center justify-between">
            <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-amber-500 rounded-2xl flex items-center justify-center shadow-lg shadow-amber-500/20">
                    <svg class="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                </div>
                <div>
                    <h1 class="text-2xl font-extrabold tracking-tight">Data Preprocessing Hub</h1>
                    <p class="text-xs text-amber-500 font-mono tracking-widest uppercase">Feature Scaling & Normalization</p>
                </div>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto w-full grid grid-cols-1 xl:grid-cols-4 gap-8">
            <!-- Sidebar -->
            <div class="xl:col-span-1 space-y-6">
                <div class="dashboard-panel p-6">
                    <h2 class="text-xs font-black text-slate-500 uppercase tracking-widest mb-8">Data Generator</h2>
                    <div class="space-y-8">
                        <div>
                            <label class="flex justify-between text-xs mb-3"><span>샘플 수 ($N$)</span><span id="v-samples" class="text-amber-500 font-bold">50</span></label>
                            <input type="range" id="p-samples" min="10" max="200" step="10" value="50" class="w-full" oninput="document.getElementById('v-samples').innerText=this.value">
                        </div>
                        <div>
                            <label class="flex justify-between text-xs mb-3"><span>연봉 범위 (백만원)</span><span id="v-salary" class="text-amber-500 font-bold">30~100</span></label>
                            <input type="range" id="p-salary-max" min="100" max="500" step="10" value="100" class="w-full" oninput="updateRangeLabel()">
                        </div>
                        <div>
                            <label class="flex justify-between text-xs mb-3"><span>나이 범위 (세)</span><span id="v-age" class="text-amber-500 font-bold">20~60</span></label>
                            <input type="range" id="p-age-max" min="60" max="100" step="5" value="60" class="w-full" oninput="updateRangeLabel()">
                        </div>
                        <button onclick="runProcessing()" class="w-full bg-amber-500 hover:bg-amber-400 text-white font-bold py-4 rounded-2xl shadow-xl shadow-amber-500/20 active:scale-95 transition-all">데이터 정규화 실행</button>
                    </div>
                </div>
                
                <div class="dashboard-panel p-6 bg-amber-500/5 border-amber-500/20">
                    <h3 class="text-sm font-bold text-amber-500 mb-4 flex items-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        왜 정규화가 필요한가요?
                    </h3>
                    <p class="text-xs text-slate-400 leading-relaxed">
                        연봉($10^7$)과 나이($10^1$)처럼 단위 차이가 극심하면, AI 모델은 값이 큰 연봉 데이터에만 압도되어 나이 데이터를 무시하게 됩니다. 0과 1 사이로 맞추면 모든 특징이 공평하게 학습에 반영됩니다.
                    </p>
                </div>
            </div>

            <!-- Content -->
            <div class="xl:col-span-3 space-y-8">
                <!-- KPI Section -->
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div class="kpi-card p-6 rounded-2xl">
                        <p class="text-xs font-bold text-slate-500 uppercase mb-2">원본 데이터 통계 (Raw Statistics)</p>
                        <div class="grid grid-cols-2 gap-4">
                            <div><p class="text-[10px] text-slate-400">Min Salary</p><p id="stat-s-min" class="text-lg font-mono tracking-tighter">0</p></div>
                            <div><p class="text-[10px] text-slate-400">Max Salary</p><p id="stat-s-max" class="text-lg font-mono tracking-tighter">0</p></div>
                        </div>
                    </div>
                    <div class="kpi-card p-6 rounded-2xl border-l-blue-500">
                        <p class="text-xs font-bold text-slate-500 uppercase mb-2">정규화 결과 (Scaled Range)</p>
                        <div class="grid grid-cols-2 gap-4">
                            <div><p class="text-[10px] text-slate-400">Salary Range</p><p class="text-lg font-mono text-blue-400">0.0 ~ 1.0</p></div>
                            <div><p class="text-[10px] text-slate-400">Age Range</p><p class="text-lg font-mono text-blue-400">0.0 ~ 1.0</p></div>
                        </div>
                    </div>
                </div>

                <!-- Charts -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div class="dashboard-panel p-4 h-[450px]" id="chart-raw"></div>
                    <div class="dashboard-panel p-4 h-[450px]" id="chart-norm"></div>
                </div>

                <!-- Math Section -->
                <section class="dashboard-panel p-10 bg-gradient-to-br from-amber-950/10 to-transparent border-t border-amber-500/20">
                    <h2 class="text-2xl font-black mb-8">Mathematical Theory</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-12">
                        <div class="space-y-4">
                            <h3 class="text-lg font-bold text-amber-500 flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-amber-500"></span> Min-Max Scaling</h3>
                            <p class="text-sm text-slate-400 leading-relaxed">데이터 집합의 모든 값을 0과 1 사이의 값으로 변환합니다. 데이터의 분포 형태는 유지하면서 스케일만 조정합니다.</p>
                            <div class="bg-black/40 p-6 rounded-xl text-center text-xl font-serif">
                                $$ x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} $$
                            </div>
                        </div>
                        <div class="space-y-4">
                            <h3 class="text-lg font-bold text-blue-400 flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-blue-500"></span> 효과 (Impact)</h3>
                            <ul class="list-disc pl-5 text-sm text-slate-400 space-y-2">
                                <li>경사 하강법(Gradient Descent)의 수렴 속도 향상</li>
                                <li>특정 특징(Feature)의 가중치 쏠림 방지</li>
                                <li>거리 기반 알고리즘(K-Means, KNN 등) 필수 단계</li>
                            </ul>
                        </div>
                    </div>
                </section>
            </div>
        </main>

        <script>
            const lay = {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#64748b' }, margin: { t: 60, b: 60, l: 80, r: 40 },
                xaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155' },
                yaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155' }
            };

            Plotly.newPlot('chart-raw', [], {...lay, title: {text:'Raw Data (Wide Scale)', font:{color:'#fff'}}});
            Plotly.newPlot('chart-norm', [], {...lay, title: {text:'Normalized Data (Unit Scale)', font:{color:'#fff'}}});

            function updateRangeLabel() {
                const sMax = document.getElementById('p-salary-max').value;
                const aMax = document.getElementById('p-age-max').value;
                document.getElementById('v-salary').innerText = `30~${sMax}`;
                document.getElementById('v-age').innerText = `20~${aMax}`;
            }

            async function runProcessing() {
                const req = {
                    n_samples: parseInt(document.getElementById('p-samples').value),
                    salary_min: 30, salary_max: parseFloat(document.getElementById('p-salary-max').value),
                    age_min: 20, age_max: parseFloat(document.getElementById('p-age-max').value)
                };

                const res = await fetch('/process', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(req)
                });
                const d = await res.json();

                document.getElementById('stat-s-min').innerText = d.stats.s_min.toLocaleString() + ' ₩';
                document.getElementById('stat-s-max').innerText = d.stats.s_max.toLocaleString() + ' ₩';

                // Raw Chart
                Plotly.react('chart-raw', [{
                    x: d.raw.age, y: d.raw.salary, mode: 'markers',
                    marker: { color: '#f59e0b', size: 8, opacity: 0.7 }
                }], {...lay, title: {text:'Raw Data: Age vs Salary', font:{color:'#fff'}}, xaxis: {title:'Age (Years)'}, yaxis: {title:'Salary (Won)'}});

                // Normalized Chart
                Plotly.react('chart-norm', [{
                    x: d.norm.age, y: d.norm.salary, mode: 'markers',
                    marker: { color: '#3b82f6', size: 8, opacity: 0.7 }
                }], {
                    ...lay, title: {text:'Normalized Data (Square View)', font:{color:'#fff'}},
                    xaxis: {title:'Age (0.0~1.0)', range:[-0.1, 1.1]},
                    yaxis: {title:'Salary (0.0~1.0)', range:[-0.1, 1.1]},
                    width: document.getElementById('chart-norm').offsetWidth,
                    height: document.getElementById('chart-norm').offsetHeight
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/process")
async def process_data(request: ProcessRequest):
    return model.process(
        n_samples=request.n_samples,
        salary_range=(request.salary_min, request.salary_max),
        age_range=(request.age_min, request.age_max)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
