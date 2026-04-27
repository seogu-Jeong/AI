import json
from sqlalchemy import desc, func
from db.models import AIScore, Feature, Portfolio

class PortfolioService:
    def build_portfolio_custom(self, prefs: dict, amount: float, db_session) -> dict:
        # Get latest data
        latest_date = db_session.query(func.max(AIScore.date)).scalar()
        if not latest_date:
            return self.generate_mock_portfolio('neutral', amount)

        # Get weights from prefs
        w_ret = prefs.get('return_w', 50)
        w_sta = prefs.get('stability_w', 50)
        w_div = prefs.get('dividend_w', 30)
        w_gro = prefs.get('growth_w', 40)
        cash_pct = prefs.get('cash_pct', 0.2)

        # Query AIScore and Feature
        # Use outer join in case Feature data is missing for some tickers
        query = db_session.query(AIScore, Feature).outerjoin(
            Feature, (AIScore.ticker == Feature.ticker) & (AIScore.date == Feature.date)
        ).filter(AIScore.date == latest_date)

        results = query.all()
        if not results:
            return self.generate_mock_portfolio('neutral', amount)

        scored_results = []
        for ai_score, feat in results:
            # Proxies for missing feature data
            ens = ai_score.ensemble_score or 50.0
            div = feat.dividend_yield if (feat and feat.dividend_yield is not None) else (ens / 20.0) # Proxy: 80 score -> 4% yield
            gro = feat.eps_growth if (feat and feat.eps_growth is not None) else (ens / 4.0) # Proxy: 80 score -> 20% growth
            
            # Stability proxy: use inverse of ATR or just a combination of ensemble and dividend
            sta_proxy = (100 - (feat.atr_14 * 10 if feat and feat.atr_14 else 20)) if feat else ens

            # Normalize components to 0-100 scale for composite calculation
            n_ens = ens
            n_div = min(div * 20.0, 100.0) # 5% -> 100
            n_gro = min(max(gro * 2.0 + 50.0, 0.0), 100.0) # 25% -> 100
            n_sta = min(max(sta_proxy, 0.0), 100.0)

            composite_score = (
                w_ret * n_ens + 
                w_sta * n_sta + 
                w_div * n_div + 
                w_gro * n_gro
            ) / max(w_ret + w_sta + w_div + w_gro, 1)

            scored_results.append({
                'ticker': ai_score.ticker,
                'score': ai_score.ensemble_score,
                'composite': composite_score,
                'feat': feat,
                'ai': ai_score
            })

        # Sort by composite score
        scored_results.sort(key=lambda x: x['composite'], reverse=True)
        
        max_tickers = 8
        top_results = scored_results[:max_tickers]
        
        invest_amount = amount * (1 - cash_pct)
        total_comp = sum(r['composite'] for r in top_results)
        
        holdings = []
        for res in top_results:
            comp_weight = res['composite'] / total_comp
            ticker_amount = invest_amount * comp_weight
            
            # Determine reason
            reason = "종합 점수 우수"
            if w_ret > 70 and res['score'] > 80: reason = "AI 고득점 상위 종목"
            elif w_div > 70 and res['feat'] and res['feat'].dividend_yield > 3.0: reason = "고배당 수익 선호"
            elif w_gro > 70 and res['feat'] and res['feat'].eps_growth > 20.0: reason = "성장성·모멘텀 우수"
            elif w_sta > 70: reason = "저변동성/안정성 확보"

            holdings.append({
                "ticker": res['ticker'],
                "weight": round(comp_weight * (1 - cash_pct) * 100, 2),
                "amount": round(ticker_amount, 2),
                "score": round(res['score'], 2),
                "reason": reason
            })

        # Add VOO if stability_w > 50
        if w_sta > 50:
            voo_weight = 10.0
            # Adjust other weights to accommodate VOO
            for h in holdings:
                h['weight'] *= (1.0 - voo_weight/100.0)
                h['amount'] *= (1.0 - voo_weight/100.0)
            
            holdings.append({
                'ticker': 'VOO',
                'weight': voo_weight,
                'amount': round(amount * (voo_weight/100.0), 2),
                'score': 72.0,
                'reason': 'S&P500 지수 추종 (안정성)'
            })

        # Calculate expected return
        avg_score = sum(h['score'] * h['weight'] for h in holdings) / max(sum(h['weight'] for h in holdings), 1)
        expected_return = round(avg_score / 100.0 * 18.0 + 5.0, 1)

        return {
            "risk_level": "custom",
            "total_amount": amount,
            "holdings": holdings,
            "cash_pct": cash_pct * 100,
            "expected_return": expected_return,
            "reference_period": "1년 (사용자 설정 가중치 기반)"
        }

    def build_portfolio(self, risk_level: str, amount: float, db_session) -> dict:
        # Get latest data
        latest_date = db_session.query(func.max(AIScore.date)).scalar()
        if not latest_date:
            return self.generate_mock_portfolio(risk_level, amount)

        # Base query joining AIScore and Feature
        query = db_session.query(AIScore, Feature).join(
            Feature, (AIScore.ticker == Feature.ticker) & (AIScore.date == Feature.date)
        ).filter(AIScore.date == latest_date)

        # Risk-based filters
        if risk_level == 'conservative':
            query = query.filter(AIScore.ensemble_score.between(60, 80))
            query = query.filter(Feature.dividend_yield > 1.5)
            max_tickers = 6
            cash_pct = 0.30
        elif risk_level == 'neutral':
            query = query.filter(AIScore.ensemble_score.between(65, 90))
            max_tickers = 8
            cash_pct = 0.20
        else: # aggressive
            query = query.filter(AIScore.ensemble_score >= 75)
            max_tickers = 10
            cash_pct = 0.10

        results = query.order_by(desc(AIScore.ensemble_score)).limit(max_tickers).all()
        
        if not results:
            return self.generate_mock_portfolio(risk_level, amount)

        invest_amount = amount * (1 - cash_pct)
        total_score = sum(r[0].ensemble_score for r in results)
        
        holdings = []
        reasons = ['강한 모멘텀↑', '안정적 배당', '기술적 강세', '고성장 모멘텀', '방어적 배당주']
        
        for score_obj, feat_obj in results:
            weight = score_obj.ensemble_score / total_score
            ticker_amount = invest_amount * weight
            
            # Determine reason based on data
            if score_obj.ensemble_score > 85: reason = reasons[0]
            elif feat_obj.dividend_yield > 2.0: reason = reasons[4]
            elif score_obj.mlp_signal == 'Buy': reason = reasons[2]
            else: reason = reasons[3]

            holdings.append({
                "ticker": score_obj.ticker,
                "weight": round(weight * (1 - cash_pct) * 100, 2),
                "amount": round(ticker_amount, 2),
                "score": round(score_obj.ensemble_score, 2),
                "reason": reason
            })

        # Add VOO (S&P500 ETF) for stabilization
        if risk_level == 'conservative':
            holdings.append({
                'ticker': 'VOO',
                'weight': 15.0,
                'amount': round(amount * 0.15 * 0.85, 2),
                'score': 72.0,
                'reason': 'S&P500 지수 추종 (안정성)'
            })
        elif risk_level == 'neutral':
            holdings.append({
                'ticker': 'VOO',
                'weight': 10.0,
                'amount': round(amount * 0.10 * 0.80, 2),
                'score': 72.0,
                'reason': 'S&P500 지수 추종 (분산효과)'
            })

        # Calculate expected return based on weighted average score
        avg_score = sum(h['score'] * h['weight'] for h in holdings) / max(sum(h['weight'] for h in holdings), 1)
        expected_return = round(avg_score / 100.0 * 18.0 + 5.0, 1)

        return {
            "risk_level": risk_level,
            "total_amount": amount,
            "holdings": holdings,
            "cash_pct": cash_pct * 100,
            "expected_return": expected_return,
            "reference_period": "1년 (AI 앙상블 기반 추정)"
        }

    def generate_mock_portfolio(self, risk_level: str, amount: float) -> dict:
        data = {
            'conservative': {
                'holdings': [
                    {'ticker': 'MSFT', 'weight': 20.0, 'amount': amount*0.20, 'score': 81.0, 'reason': '안정적 배당'},
                    {'ticker': 'AAPL', 'weight': 15.0, 'amount': amount*0.15, 'score': 76.0, 'reason': '기술적 강세'},
                    {'ticker': 'JNJ', 'weight': 15.0, 'amount': amount*0.15, 'score': 68.0, 'reason': '방어적 배당주'},
                    {'ticker': 'PG', 'weight': 10.0, 'amount': amount*0.10, 'score': 70.0, 'reason': '안정적 배당'},
                    {'ticker': 'VOO', 'weight': 15.0, 'amount': amount*0.15, 'score': 72.0, 'reason': 'S&P500 지수 추종 (안정성)'},
                ],
                'cash_pct': 40.0
            },
            'neutral': {
                'holdings': [
                    {'ticker': 'NVDA', 'weight': 20.0, 'amount': amount*0.20, 'score': 87.0, 'reason': '강한 모멘텀↑'},
                    {'ticker': 'MSFT', 'weight': 20.0, 'amount': amount*0.20, 'score': 81.0, 'reason': '기술적 강세'},
                    {'ticker': 'AAPL', 'weight': 15.0, 'amount': amount*0.15, 'score': 76.0, 'reason': '고성장 모멘텀'},
                    {'ticker': 'JNJ', 'weight': 15.0, 'amount': amount*0.15, 'score': 68.0, 'reason': '방어적 배당주'},
                    {'ticker': 'VOO', 'weight': 10.0, 'amount': amount*0.10, 'score': 72.0, 'reason': 'S&P500 지수 추종 (분산효과)'},
                ],
                'cash_pct': 30.0
            },
            'aggressive': {
                'holdings': [
                    {'ticker': 'NVDA', 'weight': 25.0, 'amount': amount*0.25, 'score': 87.0, 'reason': '강한 모멘텀↑'},
                    {'ticker': 'META', 'weight': 20.0, 'amount': amount*0.20, 'score': 72.0, 'reason': '고성장 모멘텀'},
                    {'ticker': 'AAPL', 'weight': 20.0, 'amount': amount*0.20, 'score': 76.0, 'reason': '기술적 강세'},
                    {'ticker': 'AMD', 'weight': 15.0, 'amount': amount*0.15, 'score': 63.0, 'reason': '강한 모멘텀↑'},
                    {'ticker': 'AMZN', 'weight': 20.0, 'amount': amount*0.20, 'score': 68.0, 'reason': '고성장 모멘텀'},
                ],
                'cash_pct': 0.0
            }
        }
        res = data.get(risk_level, data['neutral'])
        
        # Calculate expected return for mock data as well
        holdings = res['holdings']
        avg_score = sum(h['score'] * h['weight'] for h in holdings) / max(sum(h['weight'] for h in holdings), 1)
        expected_return = round(avg_score / 100.0 * 18.0 + 5.0, 1)

        return {
            "risk_level": risk_level,
            "total_amount": amount,
            "holdings": holdings,
            "cash_pct": res['cash_pct'],
            "expected_return": expected_return,
            "reference_period": "1년 (AI 앙상블 기반 추정)"
        }

    def save_portfolio(self, portfolio: dict, name: str, db_session):
        new_p = Portfolio(
            name=name,
            risk_level=portfolio['risk_level'],
            holdings=json.dumps(portfolio['holdings'])
        )
        db_session.add(new_p)
        db_session.commit()
