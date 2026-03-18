import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 1. FastAPI App Setup
app = FastAPI(title="Quantum AI - Gradient Descent Visualizer")

# 2. Logic: Gradient Descent Simulation
class OptimizerModel:
    def __init__(self):
        # Default Function: f(x) = x^2, f'(x) = 2x
        self.func = lambda x: x**2
        self.grad = lambda x: 2*x

    def simulate(self, start_x=-4.0, lr=0.1, n_steps=20):
        current_x = start_x
        path = []
        
        for i in range(n_steps):
            current_loss = self.func(current_x)
            path.append({"step": i, "x": float(current_x), "loss": float(current_loss)})
            
            # Update: x = x - lr * grad
            g = self.grad(current_x)
            current_x = current_x - (lr * g)
            
        return {
            "path": path,
            "curve": {
                "x": np.linspace(-5, 5, 100).tolist(),
                "y": self.func(np.linspace(-5, 5, 100)).tolist()
            }
        }

model = OptimizerModel()

class GDRequest(BaseModel):
    start_x: float = -4.0
    lr: float = 0.1
    steps: int = 20

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = r"""
    <!DOCTYPE html>
    <html lang="ko" class="antialiased">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Optimization | Gradient Descent Visualizer</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <script>
            window.MathJax = { tex: { inlineMath: [['$', '$']] } };
        </script>
        <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Fira+Code:wght@400;500&family=Noto+Sans+KR:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            :root { --bg: #050505; --panel: rgba(20, 20, 25, 0.8); --accent: #8b5cf6; }
            body { font-family: 'Inter', 'Noto Sans KR', sans-serif; background: var(--bg); color: #e5e7eb; scroll-behavior: smooth; }
            .glass-panel { background: var(--panel); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.05); border-radius: 1.5rem; }
            input[type=range] { accent-color: var(--accent); }
            .glow { text-shadow: 0 0 15px rgba(139, 92, 246, 0.5); }
        </style>
    </head>
    <body class="min-h-screen flex flex-col p-4 md:p-10">
        
        <nav class="max-w-7xl mx-auto w-full mb-12 flex items-center justify-between">
            <div class="flex items-center gap-5">
                <div class="w-14 h-14 bg-gradient-to-br from-violet-500 to-fuchsia-600 rounded-2xl flex items-center justify-center shadow-2xl shadow-violet-500/20">
                    <svg class="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
                </div>
                <div>
                    <h1 class="text-3xl font-black tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-500">Gradient Descent Lab</h1>
                    <p class="text-xs text-violet-400 font-mono tracking-widest uppercase opacity-80">Hyperparameter Optimization Visualizer</p>
                </div>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
            <!-- Sidebar -->
            <div class="lg:col-span-4 space-y-6">
                <div class="glass-panel p-8">
                    <h2 class="text-xs font-black text-slate-500 uppercase tracking-[0.2em] mb-10 flex items-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path></svg>
                        Optimizer Settings
                    </h2>
                    
                    <div class="space-y-10">
                        <div>
                            <div class="flex justify-between text-sm mb-4">
                                <span class="text-slate-400">학습률 (Learning Rate, $\eta$)</span>
                                <span id="v-lr" class="text-violet-400 font-mono font-bold bg-violet-500/10 px-2 rounded">0.10</span>
                            </div>
                            <input type="range" id="p-lr" min="0.01" max="1.1" step="0.01" value="0.1" class="w-full" oninput="document.getElementById('v-lr').innerText=parseFloat(this.value).toFixed(2)">
                            <div class="flex justify-between text-[10px] text-slate-600 mt-2"><span>Slow & Steady</span><span>Fast & Volatile</span></div>
                        </div>

                        <div>
                            <div class="flex justify-between text-sm mb-4">
                                <span class="text-slate-400">시작 위치 (Start $x$)</span>
                                <span id="v-start" class="text-fuchsia-400 font-mono font-bold bg-fuchsia-500/10 px-2 rounded">-4.0</span>
                            </div>
                            <input type="range" id="p-start" min="-4.5" max="4.5" step="0.1" value="-4.0" class="w-full" oninput="document.getElementById('v-start').innerText=parseFloat(this.value).toFixed(1)">
                        </div>

                        <div>
                            <div class="flex justify-between text-sm mb-4">
                                <span class="text-slate-400">업데이트 횟수 (Steps)</span>
                                <span id="v-steps" class="text-slate-200 font-mono font-bold bg-white/5 px-2 rounded">20</span>
                            </div>
                            <input type="range" id="p-steps" min="5" max="100" step="5" value="20" class="w-full" oninput="document.getElementById('v-steps').innerText=this.value">
                        </div>

                        <button onclick="runSimulation()" class="w-full bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-500 hover:to-fuchsia-500 text-white font-bold py-4 rounded-2xl shadow-xl shadow-violet-900/30 active:scale-95 transition-all">경사 하강 시뮬레이션</button>
                    </div>
                </div>

                <div class="glass-panel p-6 border-l-4 border-l-violet-500">
                    <h3 class="text-sm font-bold text-violet-400 mb-3 uppercase tracking-tighter">Objective Function</h3>
                    <p class="text-2xl font-serif text-center py-4 glow">$f(x) = x^2$</p>
                    <p class="text-xs text-slate-500 leading-relaxed italic">가장 낮은 곳($x=0$)을 찾는 것이 인공지능의 목표입니다.</p>
                </div>
            </div>

            <!-- Visualization -->
            <div class="lg:col-span-8 space-y-8">
                <div class="glass-panel p-6 h-[550px]" id="chart-main"></div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="glass-panel p-10 space-y-6">
                        <h2 class="text-xl font-bold border-b border-white/5 pb-4">수학적 원리</h2>
                        <div class="space-y-6 text-sm text-slate-400 leading-relaxed">
                            <p>경사 하강법은 현재 위치에서 **기울기(Gradient)**를 구한 뒤, 그 **반대 방향**으로 조금씩 이동하여 최솟값에 도달하는 방법입니다.</p>
                            <div class="bg-black/40 p-6 rounded-2xl border border-white/5 text-lg font-serif">
                                $$ x_{new} = x_{old} - \eta \cdot \frac{df}{dx} $$
                            </div>
                            <ul class="space-y-3 pl-4 text-xs">
                                <li>$\eta$: Learning Rate (이동 보폭)</li>
                                <li>$\frac{df}{dx}$: 기울기 (내리막길 방향)</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="glass-panel p-10">
                        <h2 class="text-xl font-bold border-b border-white/5 pb-4 mb-6">Optimization Log</h2>
                        <div class="h-64 overflow-y-auto space-y-2 pr-2 custom-scrollbar text-xs font-mono" id="log-container">
                            <p class="text-slate-600">Waiting for simulation...</p>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <script>
            const lay = {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#64748b' },
                margin: { t: 60, b: 60, l: 60, r: 60 },
                xaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155', range: [-5, 5] },
                yaxis: { gridcolor: '#1e293b', zerolinecolor: '#334155', range: [-2, 25] }
            };

            Plotly.newPlot('chart-main', [], {...lay, title: {text:'Optimization Landscape', font:{color:'#fff', size:18, weight:800}}});

            async function runSimulation() {
                const req = {
                    start_x: parseFloat(document.getElementById('p-start').value),
                    lr: parseFloat(document.getElementById('p-lr').value),
                    steps: parseInt(document.getElementById('p-steps').value)
                };

                const res = await fetch('/simulate', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(req)
                });
                const d = await res.json();

                const curve = d.curve;
                const path = d.path;

                // Trace 1: Static Curve
                const traceCurve = {
                    x: curve.x, y: curve.y, mode: 'lines',
                    line: { color: 'rgba(255,255,255,0.15)', width: 3 }
                };

                // Trace 2: GD Steps
                const tracePath = {
                    x: path.map(p => p.x), y: path.map(p => p.loss),
                    mode: 'lines+markers',
                    line: { color: '#8b5cf6', width: 2, dash: 'dot' },
                    marker: { 
                        color: path.map((_, i) => i),
                        colorscale: 'Viridis',
                        size: 10,
                        line: { color: 'white', width: 1 }
                    }
                };

                Plotly.react('chart-main', [traceCurve, tracePath], lay);

                // Update Log
                const logCont = document.getElementById('log-container');
                logCont.innerHTML = path.map(p => `
                    <div class="flex justify-between border-b border-white/5 py-1">
                        <span>Step ${p.step.toString().padStart(2, '0')}</span>
                        <span class="text-violet-400">x: ${p.x.toFixed(4)}</span>
                        <span class="text-fuchsia-400">Loss: ${p.loss.toFixed(4)}</span>
                    </div>
                `).join('');
            }
        </script>
        <style>
            .custom-scrollbar::-webkit-scrollbar { width: 4px; }
            .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
            .custom-scrollbar::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
        </style>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/simulate")
async def simulate_gd(request: GDRequest):
    return model.simulate(start_x=request.start_x, lr=request.lr, n_steps=request.steps)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
