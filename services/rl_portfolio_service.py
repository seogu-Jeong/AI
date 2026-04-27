import numpy as np
import pandas as pd
from db.models import AIScore, RawOHLCV


class RLPortfolioAgent:
    ACTION_SIZES = [5, 10, 15, 20, 25, 30]
    N_ACTIONS = 6
    N_FEATURES = 4

    def __init__(self, lr=0.05, gamma=0.95, n_epochs=300, seed=42):
        np.random.seed(seed)
        self.lr = lr
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.W = np.zeros((self.N_FEATURES, self.N_ACTIONS))

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    def get_probs(self, state):
        logits = state @ self.W
        return self._softmax(logits)

    def select_action(self, state, greedy=False):
        probs = self.get_probs(state)
        if greedy:
            return int(np.argmax(probs))
        return int(np.random.choice(self.N_ACTIONS, p=probs))

    def get_top_n(self, state, greedy=True):
        return self.ACTION_SIZES[self.select_action(state, greedy=greedy)]

    def train(self, states, returns_by_action):
        lr = self.lr
        for epoch in range(self.n_epochs):
            for t in range(len(states)):
                state = states[t]
                probs = self.get_probs(state)
                action = int(np.random.choice(self.N_ACTIONS, p=probs))
                reward = returns_by_action[t][action]
                grad = np.outer(state, -probs)
                grad[:, action] += state
                self.W += lr * reward * grad
            lr *= 0.999


class RLPortfolioService:
    @staticmethod
    def run_rl_backtest(db_session, start_date: str, end_date: str, top_n: int = 30) -> dict:
        try:
            from services.backtest_service import BacktestService
            import yfinance as yf

            scores_query = db_session.query(AIScore).filter(
                AIScore.date >= start_date,
                AIScore.date <= end_date
            ).all()

            if not scores_query:
                return {'error': 'DB에 AI Score 데이터가 없습니다', 'rl_mode': True}

            df_scores = pd.DataFrame([{
                'ticker': s.ticker,
                'date': s.date,
                'ensemble_score': s.ensemble_score
            } for s in scores_query])
            df_scores['date'] = pd.to_datetime(df_scores['date'])

            distinct_dates = sorted(df_scores['date'].unique())
            if len(distinct_dates) < 4:
                return {'error': '최소 4개월 데이터가 필요합니다', 'rl_mode': True}

            df_dates = pd.DataFrame({'date': distinct_dates})
            df_dates['month'] = df_dates['date'].dt.to_period('M')
            rebalance_dates = df_dates.groupby('month')['date'].min().sort_values().tolist()

            if len(rebalance_dates) < 4:
                return {'error': '최소 4개 리밸런싱 구간이 필요합니다', 'rl_mode': True}

            action_sizes = RLPortfolioAgent.ACTION_SIZES
            T = len(rebalance_dates) - 1

            spy_prices = {}
            spy_db = db_session.query(RawOHLCV).filter(
                RawOHLCV.ticker == 'SPY',
                RawOHLCV.date >= rebalance_dates[0].date(),
                RawOHLCV.date <= rebalance_dates[-1].date()
            ).all()
            if spy_db:
                spy_prices = {s.date: s.close for s in spy_db}
            else:
                try:
                    spy_df = yf.download('SPY', start=rebalance_dates[0],
                                         end=rebalance_dates[-1] + pd.Timedelta(days=5),
                                         progress=False, auto_adjust=True)
                    if not spy_df.empty:
                        if isinstance(spy_df.columns, pd.MultiIndex):
                            spy_df.columns = spy_df.columns.get_level_values(0)
                        if hasattr(spy_df.index, 'tz') and spy_df.index.tz is not None:
                            spy_df.index = spy_df.index.tz_localize(None)
                        for d, row_data in spy_df.iterrows():
                            price = row_data['Close'] if 'Close' in spy_df.columns else row_data.iloc[0]
                            if isinstance(price, (pd.Series, pd.DataFrame)):
                                price = float(price.iloc[0])
                            spy_prices[d.date()] = float(price)
                except Exception:
                    pass

            states = []
            returns_by_action = []
            benchmark_rets = []
            prev_ret = 0.0

            for i in range(T):
                d_start = rebalance_dates[i]
                d_end = rebalance_dates[i + 1]

                month_scores = df_scores[df_scores['date'] == d_start].sort_values(
                    by='ensemble_score', ascending=False
                )
                all_tickers = month_scores['ticker'].tolist()
                all_scores = month_scores['ensemble_score'].values

                if len(all_scores) >= 30:
                    top30 = all_scores[:30]
                    mean_s = float(np.mean(top30)) / 100.0
                    std_s = float(np.std(top30)) / 100.0
                    top5_avg = float(np.mean(all_scores[:5])) / 100.0
                elif len(all_scores) > 0:
                    mean_s = float(np.mean(all_scores)) / 100.0
                    std_s = float(np.std(all_scores)) / 100.0 if len(all_scores) > 1 else 0.1
                    top5_avg = float(np.mean(all_scores[:min(5, len(all_scores))])) / 100.0
                else:
                    mean_s, std_s, top5_avg = 0.5, 0.1, 0.5

                state = np.array([mean_s, std_s, top5_avg, prev_ret])
                states.append(state)

                max_n = min(max(action_sizes), len(all_tickers))
                tickers_needed = all_tickers[:max_n]

                prices_start = db_session.query(RawOHLCV).filter(
                    RawOHLCV.ticker.in_(tickers_needed),
                    RawOHLCV.date == d_start.date()
                ).all()
                prices_end = db_session.query(RawOHLCV).filter(
                    RawOHLCV.ticker.in_(tickers_needed),
                    RawOHLCV.date == d_end.date()
                ).all()

                p_start_map = {p.ticker: p.close for p in prices_start}
                p_end_map = {p.ticker: p.close for p in prices_end}

                rets_per_ticker = {}
                for t in tickers_needed:
                    if t in p_start_map and t in p_end_map and p_start_map[t] > 0:
                        rets_per_ticker[t] = (p_end_map[t] - p_start_map[t]) / p_start_map[t]

                action_rets = []
                for n in action_sizes:
                    top_tickers = all_tickers[:min(n, len(all_tickers))]
                    rets = [rets_per_ticker[tt] for tt in top_tickers if tt in rets_per_ticker]
                    action_rets.append(float(np.mean(rets)) if rets else 0.0)
                returns_by_action.append(action_rets)

                s_price = spy_prices.get(d_start.date())
                e_price = spy_prices.get(d_end.date())
                bench_ret = (e_price - s_price) / s_price if s_price and e_price and s_price > 0 else 0.0
                benchmark_rets.append(bench_ret)

                prev_ret = action_rets[-1]

            train_size = max(2, int(T * 0.8))
            agent = RLPortfolioAgent(lr=0.05, n_epochs=300)
            agent.train(states[:train_size], returns_by_action[:train_size])

            rl_curve = [1.0]
            eq_curve = [1.0]
            bench_curve = [1.0]
            dates_out = [rebalance_dates[0].strftime('%Y-%m-%d')]
            rl_monthly = []
            eq_monthly = []

            for i in range(T):
                state = states[i]
                action = agent.select_action(state, greedy=True)
                rl_ret = returns_by_action[i][action]
                eq_ret = returns_by_action[i][-1]
                bench_ret = benchmark_rets[i]

                rl_curve.append(rl_curve[-1] * (1 + rl_ret))
                eq_curve.append(eq_curve[-1] * (1 + eq_ret))
                bench_curve.append(bench_curve[-1] * (1 + bench_ret))
                rl_monthly.append(rl_ret)
                eq_monthly.append(eq_ret)
                dates_out.append(rebalance_dates[i + 1].strftime('%Y-%m-%d'))

            action_history = [RLPortfolioAgent.ACTION_SIZES[agent.select_action(s, greedy=True)] for s in states]

            stats_eq = BacktestService._compute_metrics(eq_monthly, eq_curve, bench_curve)
            stats_rl = BacktestService._compute_metrics(rl_monthly, rl_curve, bench_curve)

            return {
                'dates': dates_out,
                'portfolio_curve': [float(v) for v in eq_curve],
                'benchmark_curve': [float(v) for v in bench_curve],
                'rl_curve': [float(v) for v in rl_curve],
                'total_return': float(stats_eq['total_return']),
                'benchmark_return': float(stats_eq['benchmark_return']),
                'max_drawdown': float(stats_eq['max_drawdown']),
                'annualized_return': float(stats_eq['annualized_return']),
                'sharpe': float(stats_eq['sharpe']),
                'var_95': float(stats_eq['var_95']),
                'sortino': float(stats_eq['sortino']),
                'win_rate': float(stats_eq['win_rate']),
                'rolling_sharpe': stats_eq['rolling_sharpe'],
                'n_months': T,
                'rl_total_return': float(stats_rl['total_return']),
                'rl_sharpe': float(stats_rl['sharpe']),
                'rl_max_drawdown': float(stats_rl['max_drawdown']),
                'rl_sortino': float(stats_rl['sortino']),
                'rl_win_rate': float(stats_rl['win_rate']),
                'rl_annualized_return': float(stats_rl['annualized_return']),
                'action_history': action_history,
                'train_size': train_size,
                'error': None,
                'rl_mode': True
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'dates': [], 'portfolio_curve': [], 'benchmark_curve': [], 'rl_curve': [],
                'total_return': 0.0, 'benchmark_return': 0.0, 'max_drawdown': 0.0,
                'annualized_return': 0.0, 'sharpe': 0.0, 'var_95': 0.0, 'sortino': 0.0,
                'win_rate': 0.0, 'rolling_sharpe': [], 'n_months': 0,
                'rl_total_return': 0.0, 'rl_sharpe': 0.0, 'rl_max_drawdown': 0.0,
                'rl_sortino': 0.0, 'rl_win_rate': 0.0, 'rl_annualized_return': 0.0,
                'action_history': [], 'train_size': 0,
                'error': str(e), 'rl_mode': True
            }
