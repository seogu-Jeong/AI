# 디지몬 게임

디지몬 IP를 기반으로 만든 두 가지 버전의 게임 프로젝트입니다.

## 구성

### web/
- 바닐라 JavaScript로 구현한 웹 브라우저용 디지몬 게임
- `index.html` 파일을 브라우저로 열어 바로 실행 가능
- 기술 스택: HTML, CSS, JavaScript

### pyqt/
- PyQt5 기반 데스크탑 디지몬 게임
- GUI 인터페이스 및 게임 로직 구현
- 기술 스택: Python, PyQt5

## 실행 방법

### 웹 버전
```bash
# web 폴더에서
open index.html
```

### PyQt 버전
```bash
cd pyqt
pip install pyqt5
python run_game.py
```
