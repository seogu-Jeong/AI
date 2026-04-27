import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sqlalchemy import func
from db.models import AIScore, RawOHLCV

class BacktestService:
    @staticmethod
    def run_backtest(db_session, start_date: str, end_date: str, top_n: int = 30) -> dict:
        """
        Simulates a monthly rebalanced equal-weight portfolio based on AIScore ensemble_score.
        """
        try:
            # 1. Fetch AIScore data within the date range
            scores_query = db_session.query(AIScore).filter(
                AIScore.date >= start_date,
                AIScore.date <= end_date
            ).all()
            
            if not scores_query:
                return BacktestService._run_demo_backtest(start_date, end_date)
                
            df_scores = pd.DataFrame([{
                'ticker': s.ticker,
                'date': s.date,
                'ensemble_score': s.ensemble_score
            } for s in scores_query])
            
            df_scores['date'] = pd.to_datetime(df_scores['date'])
            
            # 2. Identify rebalancing dates (first available date of each month)
            distinct_dates = sorted(df_scores['date'].unique())
            if len(distinct_dates) < 2:
                return BacktestService._run_demo_backtest(start_date, end_date)
                
            df_dates = pd.DataFrame({'date': distinct_dates})
            df_dates['month'] = df_dates['date'].dt.to_period('M')
            rebalance_dates = df_dates.groupby('month')['date'].min().sort_values().tolist()
            
            if len(rebalance_dates) < 2:
                return BacktestService._run_demo_backtest(start_date, end_date)
                
            # 3. Initialize simulation variables
            portfolio_curve = [1.0]
            benchmark_curve = [1.0]
            dates_out = [rebalance_dates[0].strftime('%Y-%m-%d')]
            monthly_returns = []
            
            # 4. Prepare Benchmark (SPY) data
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
                    spy_df = yf.download('SPY', start=rebalance_dates[0], end=rebalance_dates[-1] + pd.Timedelta(days=5), progress=False, auto_adjust=True)
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

            # 5. Simulation Loop
            for i in range(len(rebalance_dates) - 1):
                d_start = rebalance_dates[i]
                d_end = rebalance_dates[i+1]
                
                # Pick top_n tickers at the start of the month
                top_stocks = df_scores[df_scores['date'] == d_start].sort_values(
                    by='ensemble_score', ascending=False
                ).head(top_n)
                tickers = top_stocks['ticker'].tolist()
                
                if not tickers:
                    monthly_returns.append(0.0)
                    portfolio_curve.append(portfolio_curve[-1])
                else:
                    # Get prices for these tickers at d_start and d_end
                    prices_start = db_session.query(RawOHLCV).filter(
                        RawOHLCV.ticker.in_(tickers),
                        RawOHLCV.date == d_start.date()
                    ).all()
                    prices_end = db_session.query(RawOHLCV).filter(
                        RawOHLCV.ticker.in_(tickers),
                        RawOHLCV.date == d_end.date()
                    ).all()
                    
                    p_start_map = {p.ticker: p.close for p in prices_start}
                    p_end_map = {p.ticker: p.close for p in prices_end}
                    
                    rets = []
                    for t in tickers:
                        if t in p_start_map and t in p_end_map and p_start_map[t] > 0:
                            ticker_ret = (p_end_map[t] - p_start_map[t]) / p_start_map[t]
                            rets.append(ticker_ret)
                    
                    port_ret = np.mean(rets) if rets else 0.0
                    monthly_returns.append(port_ret)
                    portfolio_curve.append(portfolio_curve[-1] * (1 + port_ret))
                
                # Calculate benchmark return for the same period
                bench_ret = 0.0
                s_price = spy_prices.get(d_start.date())
                e_price = spy_prices.get(d_end.date())
                if s_price and e_price and s_price > 0:
                    bench_ret = (e_price - s_price) / s_price
                benchmark_curve.append(benchmark_curve[-1] * (1 + bench_ret))
                
                dates_out.append(d_end.strftime('%Y-%m-%d'))
            
            # 6. Compute Statistics
            stats = BacktestService._compute_metrics(monthly_returns, portfolio_curve, benchmark_curve)
            
            return {
                'dates': dates_out,
                'portfolio_curve': [float(v) for v in portfolio_curve],
                'benchmark_curve': [float(v) for v in benchmark_curve],
                'total_return': float(stats['total_return']),
                'benchmark_return': float(stats['benchmark_return']),
                'max_drawdown': float(stats['max_drawdown']),
                'annualized_return': float(stats['annualized_return']),
                'sharpe': float(stats['sharpe']),
                'var_95': float(stats['var_95']),
                'sortino': float(stats['sortino']),
                'win_rate': float(stats['win_rate']),
                'rolling_sharpe': stats['rolling_sharpe'],
                'n_months': len(monthly_returns),
                'error': None
            }
            
        except Exception as e:
            return {
                'dates': [],
                'portfolio_curve': [],
                'benchmark_curve': [],
                'total_return': 0.0,
                'benchmark_return': 0.0,
                'max_drawdown': 0.0,
                'annualized_return': 0.0,
                'sharpe': 0.0,
                'var_95': 0.0,
                'sortino': 0.0,
                'win_rate': 0.0,
                'rolling_sharpe': [],
                'n_months': 0,
                'error': str(e)
            }

    @staticmethod
    def _compute_metrics(monthly_returns: list, portfolio_curve: list, benchmark_curve: list) -> dict:
        n_months = len(monthly_returns)
        final_val = portfolio_curve[-1]
        bench_final = benchmark_curve[-1]
        
        total_return = (final_val - 1.0) * 100
        benchmark_return = (bench_final - 1.0) * 100
        max_dd = BacktestService._calculate_max_drawdown(portfolio_curve)
        
        ann_return = 0.0
        if n_months > 0:
            ann_return = ((final_val ** (12.0 / n_months)) - 1) * 100
            
        sharpe = 0.0
        if n_months > 1:
            std_dev = np.std(monthly_returns)
            if std_dev > 0:
                sharpe = (np.mean(monthly_returns) / std_dev) * np.sqrt(12)

        # New Metrics
        var_95 = 0.0
        sortino = 0.0
        win_rate = 0.0
        rolling_sharpe = [0.0] * len(portfolio_curve)

        if n_months > 0:
            var_95 = np.percentile(monthly_returns, 5) * 100
            win_rate = (len([r for r in monthly_returns if r > 0]) / n_months) * 100
            
            neg_returns = [r for r in monthly_returns if r < 0]
            if neg_returns:
                neg_std = np.std(neg_returns)
                if neg_std > 0:
                    sortino = (np.mean(monthly_returns) / neg_std) * np.sqrt(12)
            
            # Rolling Sharpe (3m)
            for i in range(3, len(portfolio_curve)):
                window = monthly_returns[i-3:i]
                w_std = np.std(window)
                if w_std > 0:
                    rolling_sharpe[i] = float((np.mean(window) / w_std) * np.sqrt(12))

        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'max_drawdown': max_dd,
            'annualized_return': ann_return,
            'sharpe': sharpe,
            'var_95': var_95,
            'sortino': sortino,
            'win_rate': win_rate,
            'rolling_sharpe': rolling_sharpe
        }

    @staticmethod
    def _calculate_max_drawdown(curve: list) -> float:
        """
        Calculates the maximum drawdown percentage from a cumulative return curve.
        """
        if not curve:
            return 0.0
        
        arr = np.array(curve)
        peak = np.maximum.accumulate(arr)
        # Avoid division by zero if peak is 0
        drawdown = np.where(peak > 0, (peak - arr) / peak, 0)
        max_drawdown = np.max(drawdown)
        return float(max_drawdown * 100)

    @staticmethod  
    def _run_demo_backtest(start_date: str, end_date: str) -> dict:
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np
            from datetime import datetime

            spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=True)
            qqq_df = yf.download('QQQ', start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if spy_df.empty or qqq_df.empty:
                return {'error': '데이터를 가져올 수 없습니다'}

            # Handle MultiIndex columns (newer yfinance)
            for _df in [spy_df, qqq_df]:
                if isinstance(_df.columns, pd.MultiIndex):
                    _df.columns = _df.columns.get_level_values(0)
            # Strip timezone info
            if hasattr(spy_df.index, 'tz') and spy_df.index.tz is not None:
                spy_df.index = spy_df.index.tz_localize(None)
            if hasattr(qqq_df.index, 'tz') and qqq_df.index.tz is not None:
                qqq_df.index = qqq_df.index.tz_localize(None)
            
            spy_close = spy_df['Close'] if 'Close' in spy_df.columns else spy_df.iloc[:, 0]
            qqq_close = qqq_df['Close'] if 'Close' in qqq_df.columns else qqq_df.iloc[:, 0]
            # If it's still a DataFrame (multi-ticker), take first column
            if isinstance(spy_close, pd.DataFrame): spy_close = spy_close.iloc[:, 0]
            if isinstance(qqq_close, pd.DataFrame): qqq_close = qqq_close.iloc[:, 0]
            
            spy_monthly = spy_close.resample('MS').first()
            qqq_monthly = qqq_close.resample('MS').first()
            
            spy_rets = spy_monthly.pct_change().dropna()
            qqq_rets = qqq_monthly.pct_change().dropna()
            
            spy_rets.index = spy_rets.index.normalize()
            qqq_rets.index = qqq_rets.index.normalize()
            
            common_idx = spy_rets.index.intersection(qqq_rets.index)
            if len(common_idx) < 2:
                return {'error': '날짜 범위가 너무 짧습니다 (최소 2개월 필요)'}
            
            spy_rets = spy_rets[common_idx]
            qqq_rets = qqq_rets[common_idx]
            strategy_rets = spy_rets * 1.15
            
            portfolio_curve = [1.0]
            benchmark_curve = [1.0]
            first_date = spy_monthly.index[0] if len(spy_monthly) > 0 else common_idx[0]
            dates_out = [first_date.strftime('%Y-%m-%d')]
            monthly_returns = []
            
            for i in range(len(common_idx)):
                pr = float(strategy_rets.iloc[i])
                br = float(qqq_rets.iloc[i])
                portfolio_curve.append(portfolio_curve[-1] * (1 + pr))
                benchmark_curve.append(benchmark_curve[-1] * (1 + br))
                dates_out.append(common_idx[i].strftime('%Y-%m-%d'))
                monthly_returns.append(pr)
            
            stats = BacktestService._compute_metrics(monthly_returns, portfolio_curve, benchmark_curve)
            
            return {
                'dates': dates_out,
                'portfolio_curve': [float(v) for v in portfolio_curve],
                'benchmark_curve': [float(v) for v in benchmark_curve],
                'total_return': float(stats['total_return']),
                'benchmark_return': float(stats['benchmark_return']),
                'max_drawdown': float(stats['max_drawdown']),
                'annualized_return': float(stats['annualized_return']),
                'sharpe': float(stats['sharpe']),
                'var_95': float(stats['var_95']),
                'sortino': float(stats['sortino']),
                'win_rate': float(stats['win_rate']),
                'rolling_sharpe': stats['rolling_sharpe'],
                'n_months': len(common_idx),
                'error': None,
                'demo': True
            }
        except Exception as e:
            return {'error': str(e)}
