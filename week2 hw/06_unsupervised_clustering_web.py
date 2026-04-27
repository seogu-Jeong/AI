import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# 1. FastAPI App Setup
app = FastAPI(title="Quantum AI - Unsupervised Learning Lab")

# 2. AI Model & Data Logic (K-Means Clustering)
class KMeansModel:
    def __init__(self):
        self.k = 3
        self.centers = None
        self.X = None
        self.history_centers = []
        self.history_inertia = []
        self.closest_cluster = None

    def generate_data(self, n=300, k=3, spread=0.5):
        self.k = k
        np.random.seed(42) # 재현성을 위해 시드 고정
        
        # 가상의 데이터 중심점 
        true_centers = [
            [3.0, 3.0],   # 그룹 1 (소액/적은 방문)
            [8.0, 8.0],   # 그룹 2 (고액/많은 방문)
            [8.0, 3.0],   # 그룹 3 (고액/적은 방문)
            [3.0, 8.0],   # 그룹 4 (소액/많은 방문)
            [5.5, 5.5]    # 그룹 5 (중간)
        ]
        
        X_list = []
        samples_per_cluster = n // k
        for i in range(k):
            # 각 중심점 근처에 가우스 분포로 데이터 생성
            center = true_centers[i % len(true_centers)]
            cluster_data = np.random.normal(loc=center, scale=spread, size=(samples_per_cluster, 2))
            X_list.append(cluster_data)
        
        self.X = np.vstack(X_list)
        return self.X

    def train(self, n_samples=300, k_clusters=3, noise_spread=0.5, max_iters=20):
        X = self.generate_data(n_samples, k_clusters, noise_spread)
        
        # 1. 초기화: 랜덤한 k개의 점을 초기 중심점으로 선택
        random_indices = np.random.choice(len(X), k_clusters, replace=False)
        centers = X[random_indices]
        
        self.history_centers = [centers.tolist()]
        self.history_inertia = []
        self.closest_cluster = np.zeros(len(X))
        
        for iteration in range(max_iters):
            # 2. 거리 계산 및 그룹 할당 (거리: 유클리디안 거리)
            # distances shape: (K, N)
            distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
            closest_cluster = np.argmin(distances, axis=0)
            
            # 3. 관성(Inertia, Loss) 계산: 각 점에서 중심까지의 거리 제곱합
            inertia = 0.0
            for j in range(k_clusters):
                cluster_points = X[closest_cluster == j]
                if len(cluster_points) > 0:
                    inertia += np.sum((cluster_points - centers[j])**2)
            self.history_inertia.append(float(inertia))
            
            # 4. 중심점 이동: 각 그룹의 평균 위치로
            new_centers = np.array([X[closest_cluster == j].mean(axis=0) if len(X[closest_cluster == j]) > 0 else centers[j] for j in range(k_clusters)])
            self.history_centers.append(new_centers.tolist())
            
            # 5. 수렴 조건 (중심점이 이동하지 않으면 종료)
            if np.allclose(centers, new_centers, atol=1e-4):
                self.closest_cluster = closest_cluster
                self.centers = new_centers
                break
                
            centers = new_centers
            self.centers = centers
            self.closest_cluster = closest_cluster
            
        return {
            "data": {"x": X[:, 0].tolist(), "y": X[:, 1].tolist()},
            "labels": self.closest_cluster.tolist(),
            "final_centers": self.centers.tolist(),
            "history_centers": self.history_centers,
            "history_inertia": self.history_inertia,
            "metrics": {
                "iterations": len(self.history_inertia),
                "final_inertia": float(self.history_inertia[-1]) if self.history_inertia else 0.0
            }
        }

    def predict(self, x: float, y: float):
        if self.centers is None:
            return 0, 0.0
        point = np.array([x, y])
        distances = np.sqrt(np.sum((self.centers - point)**2, axis=1))
        min_dist = np.min(distances)
        cluster_idx = int(np.argmin(distances))
        return cluster_idx, float(min_dist)

model_wrapper = KMeansModel()

# 3. API Endpoints
class TrainRequest(BaseModel):
    n_samples: int = 300
    k_clusters: int = 3
    noise_spread: float = 0.8

class PredictRequest(BaseModel):
    x: float
    y: float

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = r"""
    <!DOCTYPE html>
    <html lang="ko" class="antialiased">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Unsupervised AI | K-Means Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        
        <!-- MathJax Configuration -->
        <script>
            window.MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true
                },
                svg: { fontCache: 'global' }
            };
        </script>
        <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
        
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #050b14;
                --bg-panel: rgba(13, 20, 35, 0.7);
                --accent-primary: #10b981; /* Emerald for Clustering */
                --text-main: #f8fafc;
            }
            body { 
                font-family: 'Inter', 'Noto Sans KR', sans-serif; 
                background-color: var(--bg-main); 
                color: var(--text-main);
                background-image: 
                    radial-gradient(circle at 80% 20%, rgba(16, 185, 129, 0.08), transparent 30%),
                    radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.08), transparent 30%);
                background-attachment: fixed;
                scroll-behavior: smooth;
            }
            .font-mono { font-family: 'Fira Code', monospace; }
            
            .dashboard-panel {
                background: var(--bg-panel);
                backdrop-filter: blur(16px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
                border-radius: 1.25rem;
            }
            
            .kpi-card {
                background: linear-gradient(145deg, rgba(20, 30, 45, 0.6) 0%, rgba(10, 15, 25, 0.8) 100%);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                transition: transform 0.3s ease, border-color 0.3s ease;
            }
            .kpi-card:hover {
                transform: translateY(-4px);
                border-top-color: var(--accent-primary);
            }

            input[type=range] {
                -webkit-appearance: none;
                width: 100%; background: transparent;
            }
            input[type=range]::-webkit-slider-thumb {
                -webkit-appearance: none;
                height: 18px; width: 18px;
                border-radius: 50%;
                background: var(--accent-primary);
                cursor: pointer;
                margin-top: -7px;
                box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
            }
            input[type=range]::-webkit-slider-runnable-track {
                width: 100%; height: 4px;
                background: #1e293b;
                border-radius: 2px;
            }
            
            .loader {
                border: 3px solid rgba(255,255,255,0.1);
                border-top-color: #fff;
                border-radius: 50%;
                width: 20px; height: 20px;
                animation: spin 1s linear infinite;
            }
            @keyframes spin { 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body class="min-h-screen flex flex-col">
        
        <!-- Navigation -->
        <nav class="border-b border-emerald-500/20 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
            <div class="max-w-[1600px] mx-auto px-6 py-4 flex items-center justify-between">
                <div class="flex items-center gap-4">
                    <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 002-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path></svg>
                    </div>
                    <div>
                        <h1 class="text-xl font-extrabold tracking-tight text-white">Data Clustering AI</h1>
                        <p class="text-[10px] text-emerald-400 font-mono uppercase tracking-[0.2em]">Unsupervised Learning Lab</p>
                    </div>
                </div>
                <div class="hidden md:flex items-center gap-8 text-sm font-semibold text-slate-400">
                    <a href="#dashboard" class="hover:text-emerald-400 transition-colors">📊 Dashboard</a>
                    <a href="#sandbox" class="hover:text-emerald-400 transition-colors">🧪 Sandbox</a>
                    <a href="#theory" class="hover:text-emerald-400 transition-colors">📚 Theory & Math</a>
                </div>
            </div>
        </nav>

        <main class="flex-1 max-w-[1600px] w-full mx-auto px-4 sm:px-6 py-8 grid grid-cols-1 xl:grid-cols-12 gap-8">
            
            <!-- LEFT SIDEBAR: HYPERPARAMETERS -->
            <aside class="xl:col-span-3 space-y-6">
                <div class="dashboard-panel p-6 border-t-2 border-t-emerald-500">
                    <h2 class="text-xs font-black text-slate-400 uppercase tracking-widest mb-8 flex items-center gap-2">
                        <svg class="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
                        학습 파라미터 (Hyperparameters)
                    </h2>
                    
                    <div class="space-y-8">
                        <div>
                            <div class="flex justify-between text-sm mb-3">
                                <span class="text-slate-300">데이터 수 ($N$)</span>
                                <span id="v-samples" class="font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">300</span>
                            </div>
                            <input type="range" id="p-samples" min="30" max="1000" step="30" value="300" oninput="document.getElementById('v-samples').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-sm mb-3">
                                <span class="text-slate-300">그룹 개수 ($K$)</span>
                                <span id="v-k" class="font-mono font-bold text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">3</span>
                            </div>
                            <input type="range" id="p-k" min="2" max="6" step="1" value="3" oninput="document.getElementById('v-k').innerText=this.value">
                        </div>

                        <div>
                            <div class="flex justify-between text-sm mb-3">
                                <span class="text-slate-300">데이터 퍼짐 정도 ($\sigma$)</span>
                                <span id="v-spread" class="font-mono font-bold text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded">0.8</span>
                            </div>
                            <input type="range" id="p-spread" min="0.1" max="2.0" step="0.1" value="0.8" oninput="document.getElementById('v-spread').innerText=parseFloat(this.value).toFixed(1)">
                        </div>

                        <div class="pt-4">
                            <button onclick="runTraining()" id="btn-train" class="w-full bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold py-4 rounded-xl shadow-lg shadow-emerald-600/30 active:scale-95 transition-all flex items-center justify-center gap-3">
                                <span class="text-base">클러스터링 알고리즘 실행</span>
                                <div id="loader" class="loader hidden"></div>
                            </button>
                        </div>
                    </div>
                </div>

                <div class="dashboard-panel p-6 bg-emerald-900/10 border-emerald-500/20">
                    <h3 class="text-xs font-bold text-emerald-400 mb-3 flex items-center gap-2">
                        💡 K-Means 작동 원리
                    </h3>
                    <p class="text-xs text-slate-400 leading-relaxed mb-4">
                        정답(Label)이 없는 데이터들 속에서 AI가 스스로 유사한 특징을 가진 데이터들을 $K$개의 그룹으로 묶어냅니다. 중심점이 더 이상 이동하지 않을 때까지 반복(Iteration)합니다.
                    </p>
                </div>
            </aside>

            <!-- MAIN CONTENT -->
            <div class="xl:col-span-9 space-y-8">
                
                <!-- KPI CARDS -->
                <section id="dashboard" class="scroll-mt-24">
                    <h2 class="text-xs font-black text-slate-500 uppercase tracking-[0.3em] mb-6">Clustering Metrics</h2>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div class="kpi-card rounded-2xl p-6">
                            <p class="text-[10px] text-slate-400 font-bold uppercase mb-2">설정된 군집 수 ($K$)</p>
                            <p class="text-3xl font-mono font-black text-white" id="kpi-k">0</p>
                        </div>
                        <div class="kpi-card rounded-2xl p-6">
                            <p class="text-[10px] text-slate-400 font-bold uppercase mb-2">반복 횟수 (Iterations)</p>
                            <p class="text-3xl font-mono font-black text-blue-400" id="kpi-iters">0</p>
                            <p class="text-[10px] text-slate-500 mt-2">수렴 시 자동 종료</p>
                        </div>
                        <div class="kpi-card rounded-2xl p-6 md:col-span-2">
                            <p class="text-[10px] text-slate-400 font-bold uppercase mb-2">최종 비용 함수 (Inertia, $J$)</p>
                            <p class="text-3xl font-mono font-black text-emerald-400" id="kpi-loss">0.00</p>
                            <p class="text-[10px] text-emerald-500/80 mt-2 flex items-center gap-1">
                                중심점과 소속 데이터 간의 거리 제곱합 (낮을수록 좋음)
                            </p>
                        </div>
                    </div>
                </section>

                <!-- CHARTS -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div class="dashboard-panel p-4 h-[450px]">
                        <div id="chart-loss" class="w-full h-full"></div>
                    </div>
                    <div class="dashboard-panel p-4 h-[450px] relative">
                        <div id="chart-cluster" class="w-full h-full"></div>
                    </div>
                </div>

                <!-- SANDBOX -->
                <section id="sandbox" class="dashboard-panel p-8 scroll-mt-24">
                    <h2 class="text-lg font-bold text-white mb-2 border-b border-slate-700/50 pb-4">
                        신규 고객 분류 가상 실험실
                    </h2>
                    <p class="text-sm text-slate-400 mb-8">학습된 AI 모델에 새로운 고객의 데이터를 입력하여 어떤 군집에 속하는지 예측합니다.</p>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-12">
                        <div class="space-y-8 bg-slate-900/50 p-8 rounded-2xl border border-white/5">
                            <div>
                                <label class="flex justify-between text-xs font-bold text-slate-400 mb-3">
                                    <span>구매 금액 (Feature X1)</span>
                                    <span id="v-inf-x" class="text-blue-400 font-mono text-base">5.0</span>
                                </label>
                                <input type="range" id="p-inf-x" min="0" max="10" step="0.1" value="5.0" class="w-full" oninput="updateInference()">
                            </div>
                            <div>
                                <label class="flex justify-between text-xs font-bold text-slate-400 mb-3">
                                    <span>방문 횟수 (Feature X2)</span>
                                    <span id="v-inf-y" class="text-blue-400 font-mono text-base">5.0</span>
                                </label>
                                <input type="range" id="p-inf-y" min="0" max="10" step="0.1" value="5.0" class="w-full" oninput="updateInference()">
                            </div>
                            <div id="sandbox-warning" class="text-xs text-amber-500 bg-amber-500/10 p-3 rounded-lg border border-amber-500/30 text-center">
                                ⚠️ 상단의 '알고리즘 실행' 버튼을 눌러 먼저 모델을 학습시키세요.
                            </div>
                        </div>

                        <div class="flex flex-col justify-center gap-6">
                            <div class="bg-black/40 border border-emerald-500/30 p-8 rounded-2xl text-center shadow-[0_0_30px_rgba(16,185,129,0.1)]">
                                <p class="text-xs text-emerald-400 uppercase font-black tracking-widest mb-4">AI 분류 결과 (Prediction)</p>
                                <p id="res-cluster" class="text-4xl font-mono font-black text-white mb-2">Group ?</p>
                                <p class="text-xs text-slate-500 mt-4">가장 가까운 중심점까지의 거리: <span id="res-dist" class="text-emerald-300 font-mono">0.00</span></p>
                            </div>
                        </div>
                    </div>
                </section>

                <!-- THEORY SECTION -->
                <section id="theory" class="dashboard-panel p-10 scroll-mt-24 bg-gradient-to-br from-emerald-950/20 to-transparent border-t border-emerald-500/30">
                    <h2 class="text-2xl font-black text-white mb-8 border-b border-white/10 pb-4">Mathematics of K-Means Clustering</h2>
                    
                    <div class="space-y-12">
                        <!-- 1. Definition -->
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-12">
                            <div class="space-y-4">
                                <h3 class="text-lg font-bold text-emerald-400 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
                                    1. 유클리디안 거리 (Euclidean Distance)
                                </h3>
                                <p class="text-slate-400 text-sm leading-relaxed">
                                    K-Means 알고리즘은 각 데이터 포인트 $x_i$와 클러스터 중심점 $\mu_j$ 사이의 거리를 기반으로 작동합니다. 가장 일반적으로 사용되는 유클리디안 거리는 $D$차원 공간에서 두 점 사이의 최단 직선 거리를 의미합니다.
                                </p>
                                <div class="bg-black/50 p-6 rounded-xl text-center border border-white/5">
                                    <span class="text-xl text-white">$$ d(x, \mu) = \sqrt{\sum_{l=1}^{D} (x_l - \mu_l)^2} $$</span>
                                </div>
                            </div>

                            <div class="space-y-4">
                                <h3 class="text-lg font-bold text-blue-400 flex items-center gap-2">
                                    <span class="w-2 h-2 rounded-full bg-blue-500"></span>
                                    2. 목적 함수 (Objective Function / Inertia)
                                </h3>
                                <p class="text-slate-400 text-sm leading-relaxed">
                                    알고리즘의 목표는 **'군집 내 제곱합(Within-Cluster Sum of Squares, WCSS)'**을 최소화하는 것입니다. 이를 관성(Inertia) 또는 비용 함수($J$)라고 부르며, 데이터들이 자신이 속한 중심점에 얼마나 조밀하게 모여있는지를 측정합니다.
                                </p>
                                <div class="bg-black/50 p-6 rounded-xl text-center border border-white/5">
                                    <span class="text-xl text-white">$$ J = \sum_{j=1}^{K} \sum_{x_i \in C_j} || x_i - \mu_j ||^2 $$</span>
                                </div>
                            </div>
                        </div>

                        <!-- 2. Algorithm Steps (EM) -->
                        <div class="bg-slate-900/40 p-8 rounded-2xl border border-white/5">
                            <h3 class="text-lg font-bold text-white mb-6 flex items-center gap-2">
                                <svg class="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                                EM 알고리즘 (Expectation-Maximization)
                            </h3>
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
                                <div class="space-y-3">
                                    <div class="text-emerald-400 font-black text-xs uppercase tracking-tighter">Step 01. Initialization</div>
                                    <p class="text-xs text-slate-400 leading-relaxed">데이터 중 무작위로 $K$개의 점을 선택하여 초기 클러스터 중심($\mu_1, \dots, \mu_K$)으로 설정합니다. 초기값 설정은 최종 결과에 영향을 줄 수 있습니다.</p>
                                </div>
                                <div class="space-y-3">
                                    <div class="text-emerald-400 font-black text-xs uppercase tracking-tighter">Step 02. Assignment (E-step)</div>
                                    <p class="text-xs text-slate-400 leading-relaxed">각 데이터 포인트 $x_i$에 대해 모든 중심점과의 거리를 계산하고, 가장 가까운 중심점이 있는 클러스터 $C_j$에 할당합니다.</p>
                                </div>
                                <div class="space-y-3">
                                    <div class="text-emerald-400 font-black text-xs uppercase tracking-tighter">Step 03. Update (M-step)</div>
                                    <p class="text-xs text-slate-400 leading-relaxed">각 클러스터에 할당된 데이터들의 산술 평균 위치를 계산하여 중심점 $\mu_j$를 새로운 위치로 업데이트합니다.</p>
                                </div>
                            </div>
                            <div class="mt-6 pt-6 border-t border-white/5 text-[10px] text-slate-500 italic">
                                * 위 과정을 중심점이 더 이상 변하지 않거나(수렴), 설정한 최대 반복 횟수에 도달할 때까지 반복합니다.
                            </div>
                        </div>

                        <!-- 3. Applications -->
                        <div class="space-y-6">
                            <h3 class="text-lg font-bold text-white">현실 세계의 활용 사례</h3>
                            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                                <div class="p-4 bg-emerald-500/5 border border-emerald-500/10 rounded-xl">
                                    <div class="font-bold text-emerald-400 mb-1 text-sm">고객 세분화</div>
                                    <p class="text-[11px] text-slate-500">구매 패턴, 방문 빈도에 따라 우수/잠재/이탈 고객군을 자동으로 분류합니다.</p>
                                </div>
                                <div class="p-4 bg-blue-500/5 border border-blue-500/10 rounded-xl">
                                    <div class="font-bold text-blue-400 mb-1 text-sm">이미지 압축</div>
                                    <p class="text-[11px] text-slate-500">유사한 색상들을 하나의 군집으로 묶어 대표색으로 표현함으로써 용량을 줄입니다.</p>
                                </div>
                                <div class="p-4 bg-purple-500/5 border border-purple-500/10 rounded-xl">
                                    <div class="font-bold text-purple-400 mb-1 text-sm">이상치 탐지</div>
                                    <p class="text-[11px] text-slate-500">어떤 군집에도 속하지 않거나 중심에서 너무 먼 데이터를 찾아 부정 결제를 감지합니다.</p>
                                </div>
                                <div class="p-4 bg-amber-500/5 border border-amber-500/10 rounded-xl">
                                    <div class="font-bold text-amber-400 mb-1 text-sm">유전자 분석</div>
                                    <p class="text-[11px] text-slate-500">유사한 발현 양상을 보이는 유전자 그룹을 찾아 질병의 원인을 분석합니다.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </main>

        <script>
            // 공통 차트 레이아웃
            const baseLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Inter', color: '#64748b' },
                margin: { t: 50, b: 50, l: 60, r: 30 },
                xaxis: { gridcolor: 'rgba(51, 65, 85, 0.2)', zerolinecolor: 'rgba(51, 65, 85, 0.5)', tickfont: {family: 'Fira Code', size: 10} },
                yaxis: { gridcolor: 'rgba(51, 65, 85, 0.2)', zerolinecolor: 'rgba(51, 65, 85, 0.5)', tickfont: {family: 'Fira Code', size: 10} }
            };

            const colors = ['#3b82f6', '#10b981', '#f43f5e', '#a855f7', '#f59e0b', '#0ea5e9'];

            // 빈 차트 초기화
            Plotly.newPlot('chart-loss', [{x:[], y:[]}], {
                ...baseLayout, title: {text: 'Convergence: Inertia (Loss) vs Iterations', font:{color:'#f8fafc', size: 14}},
                xaxis: { ...baseLayout.xaxis, title: 'Iteration (Epoch)' },
                yaxis: { ...baseLayout.yaxis, title: 'Inertia (Sum of Squared Distances)' }
            }, {displayModeBar: false, responsive: true});

            Plotly.newPlot('chart-cluster', [], {
                ...baseLayout, title: {text: '2D Data Clustering Result', font:{color:'#f8fafc', size: 14}},
                xaxis: { ...baseLayout.xaxis, title: 'Feature X1 (e.g., Purchase Amount)' },
                yaxis: { ...baseLayout.yaxis, title: 'Feature X2 (e.g., Visit Count)' },
                showlegend: false
            }, {displayModeBar: false, responsive: true});

            let isModelTrained = false;
            let currentK = 3;

            async function runTraining() {
                const btn = document.getElementById('btn-train');
                const loader = document.getElementById('loader');
                
                currentK = parseInt(document.getElementById('p-k').value);
                const req = {
                    n_samples: parseInt(document.getElementById('p-samples').value),
                    k_clusters: currentK,
                    noise_spread: parseFloat(document.getElementById('p-spread').value)
                };

                btn.disabled = true;
                btn.classList.add('opacity-70');
                loader.classList.remove('hidden');

                try {
                    const res = await fetch('/train', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(req)
                    });
                    const data = await res.json();
                    
                    updateDashboard(data);
                    isModelTrained = true;
                    document.getElementById('sandbox-warning').classList.add('hidden');
                    
                    // 학습 직후 현재 슬라이더 값으로 예측 실행
                    runInference(); 
                    
                } catch (e) {
                    console.error(e);
                    alert('Training failed.');
                } finally {
                    btn.disabled = false;
                    btn.classList.remove('opacity-70');
                    loader.classList.add('hidden');
                }
            }

            function updateDashboard(resData) {
                const metrics = resData.metrics;
                const historyInertia = resData.history_inertia;
                const finalCenters = resData.final_centers;
                const pData = resData.data;
                const labels = resData.labels;

                // 1. Update KPIs
                document.getElementById('kpi-k').innerText = currentK;
                document.getElementById('kpi-iters').innerText = metrics.iterations;
                document.getElementById('kpi-loss').innerText = metrics.final_inertia.toFixed(2);

                // 2. Update Loss Chart
                Plotly.react('chart-loss', [{
                    x: historyInertia.map((_, i) => i + 1), 
                    y: historyInertia, 
                    mode: 'lines+markers',
                    line: {color: '#10b981', width: 3, shape: 'spline'},
                    marker: {size: 6, color: '#10b981'}
                }], document.getElementById('chart-loss').layout);

                // 3. Update Cluster Chart
                const traces = [];
                
                // 데이터 산점도 (클러스터별)
                for (let i = 0; i < currentK; i++) {
                    const clusterX = [];
                    const clusterY = [];
                    for(let j=0; j < labels.length; j++) {
                        if(labels[j] === i) {
                            clusterX.push(pData.x[j]);
                            clusterY.push(pData.y[j]);
                        }
                    }
                    traces.push({
                        x: clusterX, y: clusterY, mode: 'markers', name: `Group ${i+1}`,
                        marker: {color: colors[i % colors.length], size: 6, opacity: 0.6}
                    });
                }
                
                // 최종 중심점 (별 모양)
                const centerX = finalCenters.map(c => c[0]);
                const centerY = finalCenters.map(c => c[1]);
                traces.push({
                    x: centerX, y: centerY, mode: 'markers', name: 'Centroids',
                    marker: {symbol: 'star', size: 18, color: '#ffffff', line: {width: 1, color: '#000'}}
                });

                Plotly.react('chart-cluster', traces, document.getElementById('chart-cluster').layout);
            }

            function updateInference() {
                const xVal = parseFloat(document.getElementById('p-inf-x').value).toFixed(1);
                const yVal = parseFloat(document.getElementById('p-inf-y').value).toFixed(1);
                document.getElementById('v-inf-x').innerText = xVal;
                document.getElementById('v-inf-y').innerText = yVal;
                
                if(isModelTrained) {
                    runInference();
                }
            }

            async function runInference() {
                const xVal = parseFloat(document.getElementById('p-inf-x').value);
                const yVal = parseFloat(document.getElementById('p-inf-y').value);
                
                try {
                    const res = await fetch('/predict', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({x: xVal, y: yVal})
                    });
                    const data = await res.json();
                    
                    const groupIdx = data.prediction;
                    const elRes = document.getElementById('res-cluster');
                    elRes.innerText = `Group ${groupIdx + 1}`;
                    elRes.style.color = colors[groupIdx % colors.length];
                    
                    document.getElementById('res-dist').innerText = data.distance.toFixed(3);

                } catch (e) { console.error(e); }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/train")
async def train_model(request: TrainRequest):
    return model_wrapper.train(
        n_samples=request.n_samples,
        k_clusters=request.k_clusters,
        noise_spread=request.noise_spread
    )

@app.post("/predict")
async def predict_extension(request: PredictRequest):
    cluster_idx, distance = model_wrapper.predict(request.x, request.y)
    return {"prediction": cluster_idx, "distance": distance}

if __name__ == "__main__":
    print("Starting Unsupervised Learning Dashboard at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
