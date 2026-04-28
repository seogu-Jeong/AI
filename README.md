# StockSense AI — AI 주식 분석 데스크탑 앱

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PySide6](https://img.shields.io/badge/UI-PySide6-green.svg)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**StockSense AI**는 딥러닝 모델들을 활용하여 주식 및 암호화폐 시장을 분석하고 최적의 포트폴리오 전략을 제안하는 macOS 기반 데스크탑 애플리케이션입니다.

StockSense AI is a macOS-based desktop application that analyzes stock and cryptocurrency markets using deep learning models and suggests optimal portfolio strategies.

---

## 📸 Screenshots
*(이곳에 실행 화면 스크린샷을 추가하세요)*

---

## ✨ Key Features (주요 기능)

- **S&P500 AI Screening**: LSTM, CNN, Transformer, MLP 앙상블 모델을 통해 전 종목을 스캔하고 0~100점 사이의 투자 점수를 산출합니다.
- **Stock Detail & XAI**: 캔들 차트 시각화 및 Attention Heatmap을 통한 설명 가능한 AI(XAI) 기능을 제공하여 모델의 판단 근거를 보여줍니다.
- **Backtest Engine**: 상위 N개 종목을 매월 리밸런싱하는 전략의 수익률을 SPY 벤치마크와 비교 분석합니다.
- **RL Portfolio Optimization**: REINFORCE 알고리즘을 사용하여 수익률 극대화를 위한 최적의 보유 종목 수를 결정합니다.
- **Crypto Analysis**: BTC, ETH, SOL 등 주요 암호화폐에 대한 LSTM 기반 단기 예측 및 가상 투자(Paper Trading) 환경을 제공합니다.

---

## 🚀 Quick Start (빠른 시작)

제공된 `install.sh` 스크립트를 사용하여 의존성 설치부터 앱 실행까지 한 번에 진행할 수 있습니다.

```bash
chmod +x install.sh
./install.sh
```

---

## 🛠 Manual Installation (수동 설치)

스크립트를 사용하지 않고 직접 설치하려면 다음 단계를 따르세요.

1. **Repository Clone**
   ```bash
   git clone https://github.com/seogu-Jeong/AI.git
   cd AI/middleterm/stocksense_ai
   ```

2. **Environment Setup** (Python 3.10+ 권장)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r "중간고사 과제/requirements.txt"
   ```

4. **Run Application**
   ```bash
   python3 "중간고사 과제/main.py"
   ```

---

## 🧠 Tech Stack & Course Concepts

본 프로젝트는 수강한 AI 및 머신러닝 강의의 핵심 개념들을 실제 금융 데이터에 적용하였습니다.

- **Gradient Descent / Backprop**: 모든 딥러닝 모델의 가중치 최적화 및 학습에 기본적으로 적용.
- **MLP + ReLU/Softmax**: 종목별 매수/보유/매도 확률 계산을 위한 다층 퍼셉트론 구조.
- **CNN (Convolutional Neural Networks)**: 주가 캔들 차트의 이미지 패턴 인식을 통한 추세 분석.
- **Transformer + Attention**: 시계열 데이터 내 요소별 중요도를 산출하고 XAI(Heatmap) 시각화에 활용.
- **REINFORCE (RL)**: 보상을 기반으로 포트폴리오 구성 비중과 종목 수를 최적화하는 강화학습 에이전트.

---

## 📄 License

This project is licensed under the MIT License.
