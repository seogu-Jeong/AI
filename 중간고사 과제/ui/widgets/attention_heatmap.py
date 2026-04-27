from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtWidgets import QToolTip
from PySide6.QtCore import QPoint

FACTOR_TOOLTIPS = {
    'RSI(14)': 'RSI (상대강도지수): 14일간 가격 변동 모멘텀. 70 이상=과매수, 30 이하=과매도 신호',
    'MACD': 'MACD: 12/26일 이동평균 차이. 양수=상승 모멘텀, 음수=하락 모멘텀',
    'BB%B': '볼린저밴드 %B: 현재가가 밴드 내 어디에 위치하는지. 1=상단, 0=하단',
    'ATR': 'ATR (평균진폭): 14일 평균 변동폭. 클수록 변동성 높음',
    'OBV Chg': 'OBV 변화율: 거래량 누적 지표 변화. 양수=매수세 강화',
    'Vol Ratio': '거래량 비율: 현재 거래량 / 20일 평균 거래량. >1=거래 활발',
    'Stoch%K': '스토캐스틱 %K: 최근 14일 가격 범위에서 현재가 위치 (0-100)',
    'MA Cross': 'MA 크로스: EMA5>EMA20이면 +1(골든크로스), 미만이면 -1(데드크로스)',
    'PER': 'PER (주가수익비율): 낮을수록 저평가. 업종 평균 대비 분석 필요',
    'PBR': 'PBR (주가순자산비율): 1 이하=자산가치 대비 저평가',
    'ROE': 'ROE (자기자본이익률): 높을수록 자본 효율성 우수',
    'EPS Growth': 'EPS 성장률: 주당순이익 성장률. 양수=이익 증가 추세',
    'Debt Ratio': '부채비율: 낮을수록 재무 안정성 높음',
    'Op Margin': '영업이익률: 매출 대비 영업이익 비율. 높을수록 수익성 우수',
    'Mkt Cap': '시가총액(log): 기업 규모. 클수록 대형주',
    '52W High%': '52주 고가 비율: 현재가/52주 최고가. 1에 가까울수록 신고가 근접',
    '1M Return': '1개월 수익률: 직전 21거래일 가격 변화율',
    '3M Return': '3개월 수익률: 직전 63거래일 가격 변화율. 중기 모멘텀 지표',
    'USD/KRW': 'USD/KRW 변화율: 원달러 환율 변동. 수출주에 영향',
    'VIX': 'VIX (공포지수): 시장 변동성 지표. 20 이상=불안 국면',
    'News Sent': '뉴스 감성: FinBERT 분석 당일 뉴스 감성 점수',
    'News 5MA': '뉴스 감성 5일 이동평균: 단기 뉴스 트렌드',
    'Div Yield': '배당수익률: 연간 배당금/현재가. 높을수록 배당 매력적',
    'EPS Surp': 'EPS 서프라이즈: 실제 EPS - 예상 EPS. 양수=어닝 서프라이즈',
}

class AttentionHeatmap(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(7, 9), dpi=90, facecolor='#c0c0c0')
        super().__init__(self.figure)
        self.setParent(parent)
        self._bar_factors = []
        self._bars = []
        self.mpl_connect('motion_notify_event', self._on_hover)

    def _on_hover(self, event):
        if event.inaxes and self._bars:
            for i, bar in enumerate(self._bars):
                if bar.contains(event)[0]:
                    if i < len(self._bar_factors):
                        factor_label = self._bar_factors[i]
                        tip = FACTOR_TOOLTIPS.get(factor_label, factor_label)
                        global_pos = self.mapToGlobal(QPoint(int(event.x), self.height() - int(event.y)))
                        QToolTip.showText(global_pos, tip, self)
                        return
        QToolTip.hideText()

    def plot(self, factors: list, weights: list):
        import matplotlib
        for font in ['AppleGothic', 'NanumGothic', 'DejaVu Sans']:
            try:
                matplotlib.rcParams['font.family'] = font
                break
            except Exception:
                continue
        matplotlib.rcParams['axes.unicode_minus'] = False

        self.figure.clear()

        LABEL_MAP = {
            'RSI(14)': 'RSI(14)', 'MACD': 'MACD', 'BB%B': 'BB%B',
            'ATR': 'ATR', 'OBV변화율': 'OBV Chg', '거래량비율': 'Vol Ratio',
            'Stoch%K': 'Stoch%K', 'MA크로스': 'MA Cross',
            'PER': 'PER', 'PBR': 'PBR', 'ROE': 'ROE',
            'EPS성장률': 'EPS Growth', '부채비율': 'Debt Ratio',
            '영업이익률': 'Op Margin', '시총(log)': 'Mkt Cap',
            '52주고가비율': '52W High%', '1개월수익률': '1M Return',
            '3개월수익률': '3M Return', 'USD/KRW변화율': 'USD/KRW',
            'VIX': 'VIX', '뉴스감성': 'News Sent',
            '뉴스감성5MA': 'News 5MA', '배당수익률': 'Div Yield',
            'EPS서프라이즈': 'EPS Surp'
        }
        labels = [LABEL_MAP.get(f, f) for f in factors]
        self._bar_factors = labels

        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#ffffff')
        self.figure.patch.set_facecolor('#c0c0c0')

        y_pos = np.arange(len(labels))
        w_arr = np.array(weights, dtype=float)
        norm_w = w_arr / (w_arr.max() if w_arr.max() > 0 else 1)

        # Win98 color: navy to green gradient
        colors = []
        for v in norm_w:
            if v > 0.7:
                colors.append('#006600')
            elif v > 0.4:
                colors.append('#008000')
            elif v > 0.2:
                colors.append('#000080')
            else:
                colors.append('#808080')

        self._bars = ax.barh(y_pos, weights, align='center', color=colors,
                              edgecolor='#808080', height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color='#000000', fontsize=9)
        ax.invert_yaxis()

        ax.set_xlabel('Attention Weight', color='#000000', fontsize=9)
        ax.set_title('FACTOR RELEVANCE  (hover for details)',
                     color='#000080', fontsize=10, weight='bold', pad=8)

        for spine in ax.spines.values():
            spine.set_color('#808080')
        ax.tick_params(axis='x', colors='#000000', labelsize=8)
        ax.set_facecolor('#ffffff')

        w_max = w_arr.max() if w_arr.max() > 0 else 1
        for i, bar in enumerate(self._bars):
            width = bar.get_width()
            ax.text(width + w_max * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{weights[i]:.3f}',
                    va='center', ha='left', color='#000080',
                    fontweight='bold', fontsize=8)

        self.figure.subplots_adjust(left=0.22, right=0.88, top=0.93, bottom=0.07)
        self.draw()
