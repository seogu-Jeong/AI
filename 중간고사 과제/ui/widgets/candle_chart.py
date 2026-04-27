import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import mplfinance as mpf
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
from PySide6.QtWidgets import QSizePolicy


class CandleChart(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(facecolor='#c0c0c0')
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(420)
        self.setMaximumHeight(460)

        # State for period switching
        self._full_df = None
        self._ticker = ''
        self._forecast = None
        self._active_period = '1Y'

        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, 0.5, 'Select Ticker to Load Chart', ha='center', va='center',
                color='#808080', fontsize=12)
        ax.set_axis_off()

    def set_period(self, period: str):
        self._active_period = period
        if self._full_df is not None:
            self._render(self._slice_df(period))

    def _slice_df(self, period: str) -> pd.DataFrame:
        df = self._full_df
        if df is None or df.empty:
            return df
        period_days = {
            '1M': 30, '3M': 90, '6M': 180,
            '1Y': 365, '3Y': 365*3, '5Y': 365*5, 'All': None
        }
        days = period_days.get(period)
        if days is None:
            return df
        cutoff = df.index[-1] - pd.Timedelta(days=days)
        sliced = df[df.index >= cutoff]
        return sliced if not sliced.empty else df

    def plot(self, df: pd.DataFrame, ticker: str = '', forecast: dict = None):
        self._full_df = df.copy() if df is not None and not df.empty else None
        self._ticker = ticker
        self._forecast = forecast
        self._render(self._slice_df(self._active_period))

    def _render(self, df: pd.DataFrame):
        ticker = self._ticker
        forecast = self._forecast
        self.figure.clear()
        self.figure.set_facecolor('#c0c0c0')

        if df is None or df.empty:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', color='#808080')
            ax.set_axis_off()
            self.draw()
            return

        # --- Normalize column names ---
        col_map = {}
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in df.columns:
                col_map[c] = c.title()
        if col_map:
            df = df.rename(columns=col_map)

        # --- Resample to weekly if more than 200 rows (> ~10 months) ---
        is_weekly = len(df) > 200
        if is_weekly:
            df_plot = df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            df_plot = df.copy()

        # --- Layout ---
        gs = GridSpec(3, 1, figure=self.figure,
                      height_ratios=[3, 1, 1.2],
                      hspace=0.05,
                      left=0.08, right=0.97, top=0.92, bottom=0.18)
        ax_main = self.figure.add_subplot(gs[0])
        ax_vol  = self.figure.add_subplot(gs[1])
        ax_rsi  = self.figure.add_subplot(gs[2])

        # --- Style ---
        s = mpf.make_mpf_style(
            base_mpf_style='default',
            facecolor='#ffffff',
            edgecolor='#808080',
            gridcolor='#d0d0d0',
            gridstyle='--',
            marketcolors=mpf.make_marketcolors(
                up='#008000', down='#cc0000',
                edge={'up': '#004000', 'down': '#880000'},
                wick={'up': '#004000', 'down': '#880000'},
                volume={'up': '#66BB6A', 'down': '#EF9A9A'},
                inherit=True
            )
        )

        # --- RSI ---
        close = df_plot['Close']
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        ap_rsi   = mpf.make_addplot(rsi,                   ax=ax_rsi, color='#000080', width=1.2)
        ap_rsi70 = mpf.make_addplot([70.0]*len(df_plot),   ax=ax_rsi, color='#cc0000', width=0.8, linestyle='--')
        ap_rsi30 = mpf.make_addplot([30.0]*len(df_plot),   ax=ax_rsi, color='#008000', width=0.8, linestyle='--')

        # --- MA periods: 10/40 for weekly (~50d/200d), 5/20 for daily ---
        mav_periods = (10, 40) if is_weekly else (5, 20)

        try:
            mpf.plot(df_plot, type='candle', mav=mav_periods,
                     ax=ax_main, volume=ax_vol,
                     style=s, addplot=[ap_rsi, ap_rsi70, ap_rsi30])
        except Exception:
            mpf.plot(df_plot, type='candle', mav=mav_periods,
                     ax=ax_main, volume=ax_vol, style=s)

        # --- World Event Markers ---
        MAJOR_EVENTS = [
            ("2015-08-24", "중국\n폭락",     "#e74c3c"),
            ("2016-06-23", "Brexit\n투표",   "#e67e22"),
            ("2016-11-08", "트럼프\n당선",   "#9b59b6"),
            ("2018-12-24", "미중\n무역전쟁", "#e74c3c"),
            ("2019-08-14", "국채\n역전",     "#e67e22"),
            ("2020-02-20", "코로나\n폭락",   "#c0392b"),
            ("2020-03-23", "코로나\n저점",   "#27ae60"),
            ("2020-11-09", "백신\n발표",     "#2ecc71"),
            ("2021-01-27", "GME\n공매도",    "#8e44ad"),
            ("2022-02-24", "우크라\n침공",   "#e74c3c"),
            ("2022-03-16", "FED\n금리인상",  "#e67e22"),
            ("2022-11-30", "ChatGPT\n등장",  "#3498db"),
            ("2023-03-10", "SVB\n파산",      "#e74c3c"),
            ("2024-08-05", "엔캐리\n청산",   "#e74c3c"),
            ("2025-04-02", "트럼프\n관세",   "#c0392b"),
        ]

        dates_list = list(df_plot.index)
        date_start = dates_list[0]
        date_end   = dates_list[-1]
        ymin, ymax = ax_main.get_ylim()

        for event_str, label, color in MAJOR_EVENTS:
            try:
                event_date = pd.Timestamp(event_str)
                if not (date_start <= event_date <= date_end):
                    continue
                # Find nearest index
                idx = min(range(len(dates_list)), key=lambda i: abs((dates_list[i] - event_date).days))
                ax_main.axvline(x=idx, color=color, linestyle=':', linewidth=0.9, alpha=0.75)
                # Stagger text vertically: odd events at 98% of ymax, even at 88%
                event_num = MAJOR_EVENTS.index((event_str, label, color))
                y_text = ymax * (0.99 if event_num % 2 == 0 else 0.88)
                ax_main.text(idx, y_text, label, fontsize=5.5, color=color,
                             ha='center', va='top', rotation=0,
                             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor=color, linewidth=0.5))
            except Exception:
                continue

        # --- AI Forecast Overlay ---
        if forecast:
            try:
                x_last = len(df_plot) - 1
                curr_price = forecast['current_price']
                daily_std  = forecast['daily_std']

                # Convert to weekly sigma if using weekly candles
                sigma = daily_std * np.sqrt(5) if is_weekly else daily_std

                ax_main.axvline(x=x_last, color='#757575', linestyle='--', linewidth=0.8, alpha=0.5)

                # Forecast horizons in candle units (weekly: 1/4/13, daily: 5/20/45)
                if is_weekly:
                    horizons = [(1, forecast['dir_5d'], '5일'), (4, forecast['dir_20d'], '20일'), (13, forecast['dir_60d'], '60일')]
                else:
                    horizons = [(5, forecast['dir_5d'], '5일'), (20, forecast['dir_20d'], '20일'), (45, forecast['dir_60d'], '60일')]

                colors_map = {'UP': '#2E7D32', 'FLAT': '#757575', 'DOWN': '#C62828',
                              'Up': '#2E7D32', 'Side': '#757575', 'Down': '#C62828'}

                forecast_ticks  = []
                forecast_labels = []

                for candles, direction, label in horizons:
                    fx = x_last + candles
                    mult = 1.0 if direction.upper() in ['UP', 'BUY'] else (-1.0 if direction.upper() in ['DOWN', 'SELL'] else 0.0)
                    pct = mult * sigma * np.sqrt(candles)
                    exp_price   = curr_price * (1 + pct)
                    upper_price = curr_price * (1 + 1.5 * sigma * np.sqrt(candles))
                    lower_price = curr_price * (1 - 1.5 * sigma * np.sqrt(candles))
                    color = colors_map.get(direction, '#757575')

                    ax_main.plot([x_last, fx], [curr_price, exp_price], color=color, linestyle='--', linewidth=1.5)
                    ax_main.fill_between([x_last, fx], [curr_price, lower_price], [curr_price, upper_price], color=color, alpha=0.12)
                    ax_main.scatter(fx, exp_price, color=color, s=60, zorder=5)
                    pct_str = f"{pct*100:+.1f}%"
                    ax_main.text(fx, exp_price * 1.01, f"{pct_str}\n({label})",
                                 ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
                    forecast_ticks.append(fx)
                    forecast_labels.append(label)

                ax_main.set_xlim(right=x_last + (18 if is_weekly else 55))
                ax_main.plot([], [], color='#2E7D32', linestyle='--', label='AI 예측')
                ax_main.legend(loc='upper left', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#d0d0d0')

            except Exception as e:
                print(f"Forecast plot error: {e}")
                forecast_ticks, forecast_labels = [], []
        else:
            forecast_ticks, forecast_labels = [], []

        # --- X-Axis: year labels for weekly, date labels for daily ---
        n = len(df_plot)
        ax_main.set_xticklabels([])
        ax_vol.set_xticklabels([])
        ax_vol.set_xlabel('')

        if is_weekly:
            # Show one tick per year
            positions  = []
            date_labels = []
            seen_years = set()
            for i, dt in enumerate(df_plot.index):
                yr = dt.year
                if yr not in seen_years:
                    seen_years.add(yr)
                    positions.append(i)
                    date_labels.append(str(yr))
            # Append forecast ticks
            positions  += forecast_ticks
            date_labels += forecast_labels
        else:
            step = max(1, n // 8)
            positions   = list(range(0, n, step))
            date_labels = [df_plot.index[i].strftime('%Y-%m') for i in positions]
            positions  += forecast_ticks
            date_labels += forecast_labels

        ax_rsi.xaxis.set_major_locator(FixedLocator(positions))
        ax_rsi.xaxis.set_major_formatter(FixedFormatter(date_labels))
        ax_rsi.tick_params(axis='x', rotation=30, labelsize=7)
        ax_rsi.set_xlabel('')

        # --- Panel labels ---
        period_label = f"{df_plot.index[0].strftime('%Y.%m')} ~ {df_plot.index[-1].strftime('%Y.%m')} ({'주봉' if is_weekly else '일봉'})"
        ax_main.set_title(f"{ticker}  {period_label}", fontsize=8, loc='left', color='#333333', pad=4)
        ax_vol.set_ylabel('거래량', fontsize=7, color='#000000')
        ax_rsi.set_ylabel('RSI(14)', fontsize=7, color='#000000')
        ax_rsi.set_ylim(0, 100)

        self.figure.tight_layout(pad=0.5)
        self.draw()

    def show_loading(self):
        self.figure.clear()
        self.figure.set_facecolor('#c0c0c0')
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#f0f0f0')
        ax.text(0.5, 0.5, '데이터 분석 중...', ha='center', va='center',
                color='#555555', fontsize=13, fontweight='bold')
        ax.set_axis_off()
        self.draw()
