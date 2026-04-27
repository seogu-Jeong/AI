import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 1. FastAPI App Setup
app = FastAPI(title="Custom Physics AI Laboratory")

# 2. AI Model & Data Logic
class LinearModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.k_true = 0.5  
        self.b_true = 0.0  
        self.training_data = None
        self.metrics = {}

    def generate_data(self, n=100, noise_scale=0.1, k=0.5, b=0.0):
        self.k_true = k
        self.b_true = b
        X = np.linspace(1, 15, n).astype(np.float32)
        noise = np.random.normal(0, noise_scale, n).astype(np.float32)
        y = self.k_true * X + self.b_true + noise
        self.training_data = (X.tolist(), y.tolist())
        return X, y

    def train(self, epochs=150, learning_rate=0.1, noise_scale=0.1, num_samples=100, k=0.5, b=0.0):
        X, y = self.generate_data(num_samples, noise_scale, k, b)
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(units=1, use_bias=True, kernel_initializer='random_normal')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        history = self.model.fit(X, y, epochs=epochs, verbose=0)
        self.history = history.history['loss']
        y_pred = self.model.predict(X, verbose=0).flatten()
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        mae = np.mean(np.abs(y - y_pred))
        weights = self.model.layers[0].get_weights()
        learned_k = float(weights[0][0][0])
        learned_b = float(weights[1][0])
        self.metrics = {
            "r2": float(r2),
            "mae": float(mae),
            "final_loss": float(self.history[-1]),
            "learned_k": learned_k,
            "learned_b": learned_b
        }
        return {
            "history": self.history,
            "metrics": self.metrics,
            "data": {"x": self.training_data[0], "y": self.training_data[1]}
        }

    def predict(self, mass: float):
        if self.model is None:
            return self.k_true * mass + self.b_true, self.k_true * mass + self.b_true
        ai_pred = self.model.predict(np.array([mass], dtype=np.float32), verbose=0)
        return float(ai_pred[0][0]), self.k_true * mass + self.b_true

model_wrapper = LinearModel()

class TrainRequest(BaseModel):
    epochs: int = 150
    learning_rate: float = 0.1
    noise_scale: float = 0.1
    num_samples: int = 100
    k: float = 0.5
    b: float = 0.0

class PredictRequest(BaseModel):
    mass: float

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = r"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hooke's AI Lab - Formula Editor</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <script>
            window.MathJax = {
                tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }
            };
        </script>
        <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=Noto+Sans+KR:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Inter', 'Noto Sans KR', sans-serif; background-color: #0a0f1c; color: #f8fafc; scroll-behavior: smooth; }
            .glass-panel { background: rgba(17, 24, 39, 0.8); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); border-radius: 1rem; }
            .kpi-card { background: rgba(30, 41, 59, 0.5); border-top: 2px solid transparent; transition: all 0.3s; }
            .kpi-card:hover { border-top-color: #3b82f6; transform: translateY(-2px); }
            .spring-transition { transition: height 0.3s cubic-bezier(0.4, 0, 0.2, 1); }
            input[type=range] { accent-color: #3b82f6; }
        </style>
    </head>
    <body class="min-h-screen">
        <nav class="sticky top-0 z-50 bg-black/60 backdrop-blur-md border-b border-white/10">
            <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
                <div class="flex items-center gap-2">
                    <div class="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center font-black">H</div>
                    <span class="font-bold text-lg tracking-tight tracking-widest">Hooke's AI Lab <span class="text-indigo-400 text-xs font-normal ml-2">PRO</span></span>
                </div>
                <div class="flex gap-8 text-sm font-medium text-slate-400">
                    <a href="#dashboard" class="hover:text-white transition">대시보드</a>
                    <a href="#sandbox" class="hover:text-white transition">가상실험</a>
                    <a href="#theory" class="hover:text-white transition">이론설명</a>
                </div>
            </div>
        </nav>

        <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div class="lg:col-span-3 space-y-6">
                <!-- Formula Editor -->
                <div class="glass-panel p-6 border-indigo-500/30 bg-indigo-500/5">
                    <h2 class="text-xs font-bold text-indigo-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
                        물리 수식 편집기
                    </h2>
                    <div class="space-y-6">
                        <div>
                            <div class="flex justify-between text-xs mb-2"><span>상수 ($k$)</span><span id="v-k-true" class="text-indigo-400 font-bold">0.5</span></div>
                            <input type="range" id="p-k-true" min="0.1" max="2.0" step="0.1" value="0.5" class="w-full" oninput="updateFormula()">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs mb-2"><span>편향 ($b$)</span><span id="v-b-true" class="text-indigo-400 font-bold">0.0</span></div>
                            <input type="range" id="p-b-true" min="-5.0" max="5.0" step="0.5" value="0.0" class="w-full" oninput="updateFormula()">
                        </div>
                        <div class="bg-black/40 p-4 rounded-xl text-center border border-white/5">
                            <p class="text-[10px] text-slate-500 mb-1">Target Formula</p>
                            <p id="target-formula-display" class="text-lg font-bold text-white tracking-tighter">$y = 0.5x + 0.0$</p>
                        </div>
                    </div>
                </div>

                <div class="glass-panel p-6">
                    <h2 class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">하이퍼파라미터</h2>
                    <div class="space-y-6">
                        <div>
                            <div class="flex justify-between text-xs mb-2"><span>샘플 수 ($N$)</span><span id="v-samples" class="text-blue-400 font-bold">100</span></div>
                            <input type="range" id="p-samples" min="50" max="500" step="50" value="100" class="w-full" oninput="document.getElementById('v-samples').innerText=this.value">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs mb-2"><span>측정 노이즈 ($\sigma$)</span><span id="v-noise" class="text-blue-400 font-bold">0.10</span></div>
                            <input type="range" id="p-noise" min="0" max="0.5" step="0.05" value="0.1" class="w-full" oninput="document.getElementById('v-noise').innerText=parseFloat(this.value).toFixed(2)">
                        </div>
                        <div>
                            <div class="flex justify-between text-xs mb-2"><span>학습률 ($\alpha$)</span><span id="v-lr" class="text-blue-400 font-bold">0.10</span></div>
                            <input type="range" id="p-lr" min="0.01" max="0.3" step="0.01" value="0.1" class="w-full" oninput="document.getElementById('v-lr').innerText=parseFloat(this.value).toFixed(2)">
                        </div>
                        <button onclick="runTrain()" id="btn-train" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded-xl transition-all shadow-lg shadow-blue-900/20 active:scale-95">모델 학습 실행</button>
                    </div>
                </div>
            </div>

            <div class="lg:col-span-9 space-y-8">
                <section id="dashboard" class="scroll-mt-24 space-y-6">
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div class="glass-panel kpi-card p-5 text-center">
                            <p class="text-[10px] text-slate-500 font-bold mb-1">결정계수 ($R^2$)</p>
                            <p id="kpi-r2" class="text-2xl font-black">0.0000</p>
                        </div>
                        <div class="glass-panel kpi-card p-5 text-center">
                            <p class="text-[10px] text-slate-500 font-bold mb-1">최종 손실 (MSE)</p>
                            <p id="kpi-loss" class="text-2xl font-black">0.0000</p>
                        </div>
                        <div class="glass-panel kpi-card p-5 text-center text-blue-400">
                            <p class="text-[10px] text-slate-500 font-bold mb-1">예측 상수 ($k$)</p>
                            <p id="kpi-k" class="text-2xl font-black">0.000</p>
                        </div>
                        <div class="glass-panel kpi-card p-5 text-center text-purple-400">
                            <p class="text-[10px] text-slate-500 font-bold mb-1">예측 편향 ($b$)</p>
                            <p id="kpi-b" class="text-2xl font-black">0.000</p>
                        </div>
                    </div>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="glass-panel p-4 h-80" id="chart-loss"></div>
                        <div class="glass-panel p-4 h-80" id="chart-reg"></div>
                    </div>
                </section>

                <section id="sandbox" class="scroll-mt-24 glass-panel p-8">
                    <h2 class="text-lg font-bold mb-8 flex items-center gap-2">
                        <span class="w-2 h-6 bg-indigo-600 rounded"></span> 가상 물리 실험실
                    </h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-12">
                        <div class="flex justify-center items-start h-64 relative border-l-2 border-slate-800 ml-12">
                            <div class="absolute left-[-12px] top-0 bottom-0 flex flex-col justify-between text-[10px] text-slate-600 py-2">
                                <span>0</span><span>5</span><span>10</span><span>15</span><span>20</span><span>25+</span>
                            </div>
                            <div class="flex flex-col items-center">
                                <div class="w-32 h-2 bg-slate-700 rounded-full mb-1"></div>
                                <div id="sim-spring" class="w-12 spring-transition overflow-hidden" style="height: 50px;">
                                    <svg width="100%" height="100%" viewBox="0 0 40 100" preserveAspectRatio="none">
                                        <path d="M20,0 L20,10 L5,20 L35,30 L5,40 L35,50 L5,60 L35,70 L5,80 L20,90 L20,100" fill="none" stroke="#94a3b8" stroke-width="2"/>
                                    </svg>
                                </div>
                                <div id="sim-mass" class="w-16 h-16 bg-indigo-600 rounded-xl shadow-xl flex items-center justify-center font-bold text-xs">
                                    <span id="v-inf-mass-label">5.0kg</span>
                                </div>
                            </div>
                        </div>
                        <div class="space-y-8">
                            <div class="bg-slate-900/50 p-6 rounded-2xl">
                                <label class="block text-xs font-bold text-slate-500 mb-4">입력 무게 설정 (kg)</label>
                                <input type="range" id="p-inf-mass" min="1" max="15" step="0.1" value="5" class="w-full" oninput="updateInf(this.value)">
                                <p class="text-center text-xl font-black mt-4 text-indigo-400" id="v-inf-mass">5.0 kg</p>
                            </div>
                            <div class="grid grid-cols-2 gap-4">
                                <div class="bg-blue-600/10 p-4 rounded-xl border border-blue-500/20 text-center">
                                    <p class="text-[10px] text-blue-400 font-bold mb-1">AI 예측 ($\hat{y}$)</p>
                                    <p id="res-ai" class="text-xl font-bold">0.00 cm</p>
                                </div>
                                <div class="bg-indigo-600/10 p-4 rounded-xl border border-indigo-500/20 text-center">
                                    <p class="text-[10px] text-indigo-400 font-bold mb-1">Target ($y$)</p>
                                    <p id="res-true" class="text-xl font-bold">0.00 cm</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                <section id="theory" class="scroll-mt-24 glass-panel p-10">
                    <h2 class="text-2xl font-black mb-8 border-b border-white/5 pb-4">이론적 배경</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-12 text-sm text-slate-400 leading-relaxed">
                        <div class="space-y-4">
                            <h3 class="text-white font-bold text-lg flex items-center gap-2"><span class="text-indigo-500">01.</span> Hooke's Law</h3>
                            <p>실제 세계에서 용수철의 변위 $x$는 힘 $F$에 비례합니다 ($F = kx$). 이 실험에서는 질량을 입력으로, 늘어난 길이를 출력으로 설정하여 AI를 학습시킵니다.</p>
                        </div>
                        <div class="space-y-4">
                            <h3 class="text-white font-bold text-lg flex items-center gap-2"><span class="text-indigo-500">02.</span> AI's Challenge</h3>
                            <p>당신이 편집기에서 설정한 $k$와 $b$는 AI가 도달해야 할 **'정답(Ground Truth)'**입니다. 측정 노이즈가 클수록 AI는 정답을 찾기 어려워집니다.</p>
                        </div>
                    </div>
                </section>
            </div>
        </main>

        <script>
            const lay = {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#64748b', size: 10 },
                margin: { t: 40, b: 40, l: 40, r: 20 },
                xaxis: { gridcolor: '#1e293b' }, yaxis: { gridcolor: '#1e293b' }
            };

            Plotly.newPlot('chart-loss', [{x:[], y:[]}], {...lay, title: {text:'Loss Convergence', font:{color:'#fff'}}});
            Plotly.newPlot('chart-reg', [{x:[], y:[]}], {...lay, title: {text:'Regression Fit', font:{color:'#fff'}}});

            let modelTrained = false;

            function updateFormula() {
                const k = parseFloat(document.getElementById('p-k-true').value).toFixed(1);
                const b = parseFloat(document.getElementById('p-b-true').value).toFixed(1);
                document.getElementById('v-k-true').innerText = k;
                document.getElementById('v-b-true').innerText = b;
                
                const sign = b >= 0 ? '+' : '';
                document.getElementById('target-formula-display').innerText = `$y = ${k}x ${sign} ${b}$`;
                if(window.MathJax) MathJax.typesetPromise([document.getElementById('target-formula-display')]);
            }

            async function runTrain() {
                const btn = document.getElementById('btn-train');
                btn.disabled = true; btn.innerText = "데이터 생성 및 학습 중...";
                
                const req = {
                    num_samples: parseInt(document.getElementById('p-samples').value),
                    noise_scale: parseFloat(document.getElementById('p-noise').value),
                    learning_rate: parseFloat(document.getElementById('p-lr').value),
                    k: parseFloat(document.getElementById('p-k-true').value),
                    b: parseFloat(document.getElementById('p-b-true').value),
                    epochs: 150
                };

                try {
                    const res = await fetch('/train', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(req)
                    });
                    const d = await res.json();
                    
                    document.getElementById('kpi-r2').innerText = d.metrics.r2.toFixed(4);
                    document.getElementById('kpi-loss').innerText = d.metrics.final_loss.toFixed(6);
                    document.getElementById('kpi-k').innerText = d.metrics.learned_k.toFixed(3);
                    document.getElementById('kpi-b').innerText = d.metrics.learned_b.toFixed(3);

                    Plotly.react('chart-loss', [{
                        y: d.history, mode: 'lines', line: {color:'#3b82f6'}, fill: 'tozeroy'
                    }], {...lay, title: {text:'Loss Convergence', font:{color:'#fff'}}});

                    const lx = [0, 15];
                    Plotly.react('chart-reg', [
                        {x: d.data.x, y: d.data.y, mode: 'markers', marker: {color:'rgba(100,116,139,0.3)', size:4}, name: 'Noisy Data'},
                        {x: lx, y: lx.map(x => req.k*x + req.b), mode: 'lines', line: {dash:'dash', color:'#10b981'}, name: 'Target'},
                        {x: lx, y: lx.map(x => d.metrics.learned_k*x + d.metrics.learned_b), mode: 'lines', line: {color:'#3b82f6'}, name: 'AI Pred'}
                    ], {...lay, title: {text:'Regression Fit', font:{color:'#fff'}}});

                    modelTrained = true;
                    updateInf(document.getElementById('p-inf-mass').value);
                } finally {
                    btn.disabled = false; btn.innerText = "모델 학습 실행";
                }
            }

            function updateInf(v) {
                document.getElementById('v-inf-mass').innerText = parseFloat(v).toFixed(1) + ' kg';
                document.getElementById('v-inf-mass-label').innerText = parseFloat(v).toFixed(1) + 'kg';
                predict(v);
            }

            async function predict(v) {
                const res = await fetch('/predict', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({mass: parseFloat(v)})
                });
                const d = await res.json();
                document.getElementById('res-ai').innerText = d.prediction.toFixed(2) + ' cm';
                document.getElementById('res-true').innerText = d.theory.toFixed(2) + ' cm';
                
                // 스프링 시각화 (늘어난 길이 반영)
                const baseHeight = 50;
                const extension = d.prediction * 8; // 스케일 조정
                document.getElementById('sim-spring').style.height = (baseHeight + extension) + 'px';
            }
            
            // 초기 수식 로드
            updateFormula();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/train")
async def train_model(request: TrainRequest):
    return model_wrapper.train(
        epochs=request.epochs, 
        learning_rate=request.learning_rate, 
        noise_scale=request.noise_scale, 
        num_samples=request.num_samples,
        k=request.k,
        b=request.b
    )

@app.post("/predict")
async def predict_extension(request: PredictRequest):
    ai_pred, theory_pred = model_wrapper.predict(request.mass)
    return {"prediction": ai_pred, "theory": theory_pred}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
