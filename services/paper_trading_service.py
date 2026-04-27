import yfinance as yf
from sqlalchemy.orm import Session
from db.models import PaperAccount, PaperPosition, PaperTransaction

class PaperTradingService:
    @staticmethod
    def get_or_create_account(db_session: Session, initial_cash=100000.0) -> PaperAccount:
        """Returns existing account (first row) or creates one."""
        account = db_session.query(PaperAccount).first()
        if not account:
            account = PaperAccount(initial_cash=initial_cash, current_cash=initial_cash)
            db_session.add(account)
            db_session.commit()
            db_session.refresh(account)
        return account

    @staticmethod
    def get_current_price(ticker: str) -> float:
        """
        Use yfinance: yf.Ticker(ticker).fast_info['last_price'] 
        Fallback to yf.download(ticker, period='1d')['Close'].iloc[-1]
        Return 0.0 on any error
        """
        try:
            t = yf.Ticker(ticker)
            # Try fast_info first
            try:
                price = t.fast_info['last_price']
            except (KeyError, AttributeError, Exception):
                price = None
            
            if price is None or price == 0 or (isinstance(price, float) and price != price): # nan check
                # Fallback to download
                df = yf.download(ticker, period='1d', progress=False)
                if not df.empty:
                    price = float(df['Close'].iloc[-1])
                else:
                    price = 0.0
            return float(price)
        except Exception:
            return 0.0

    @staticmethod
    def buy(ticker: str, quantity: float, db_session: Session) -> dict:
        """
        Get account, check if enough cash
        Get current price via get_current_price()
        total_cost = quantity * price
        If not enough cash: return {'success': False, 'error': '잔액 부족'}
        Deduct from account.current_cash
        Upsert PaperPosition: if exists, recalculate avg_cost = (old_qty*old_cost + qty*price)/(old_qty+qty), add quantity
        Insert PaperTransaction(action='BUY', ...)
        Commit and return {'success': True, 'ticker': ticker, 'quantity': quantity, 'price': price, 'total': total_cost}
        """
        account = PaperTradingService.get_or_create_account(db_session)
        price = PaperTradingService.get_current_price(ticker)
        
        if price <= 0:
            return {'success': False, 'error': f'가격 정보를 가져올 수 없습니다: {ticker}'}
        
        total_cost = quantity * price
        if account.current_cash < total_cost:
            return {'success': False, 'error': '잔액 부족'}
        
        # Deduct from account
        account.current_cash -= total_cost
        
        # Upsert PaperPosition
        pos = db_session.query(PaperPosition).filter_by(ticker=ticker).first()
        if pos:
            old_qty = pos.quantity
            old_cost = pos.avg_cost
            new_qty = old_qty + quantity
            new_avg_cost = (old_qty * old_cost + quantity * price) / new_qty
            pos.quantity = new_qty
            pos.avg_cost = new_avg_cost
        else:
            pos = PaperPosition(ticker=ticker, quantity=quantity, avg_cost=price)
            db_session.add(pos)
        
        # Insert PaperTransaction
        tx = PaperTransaction(
            action='BUY',
            ticker=ticker,
            quantity=quantity,
            price=price,
            total=total_cost
        )
        db_session.add(tx)
        
        db_session.commit()
        return {
            'success': True, 
            'ticker': ticker, 
            'quantity': quantity, 
            'price': price, 
            'total': total_cost
        }

    @staticmethod
    def sell(ticker: str, quantity: float, db_session: Session) -> dict:
        """
        Get position, check if enough shares
        If no position or insufficient qty: return {'success': False, 'error': '보유 수량 부족'}
        Get current price, add to cash
        Reduce position quantity (delete if 0)
        Insert PaperTransaction(action='SELL', ...)
        Commit and return {'success': True, ...}
        """
        pos = db_session.query(PaperPosition).filter_by(ticker=ticker).first()
        if not pos or pos.quantity < quantity:
            return {'success': False, 'error': '보유 수량 부족'}
        
        price = PaperTradingService.get_current_price(ticker)
        if price <= 0:
            return {'success': False, 'error': f'가격 정보를 가져올 수 없습니다: {ticker}'}
            
        total_revenue = quantity * price
        account = PaperTradingService.get_or_create_account(db_session)
        account.current_cash += total_revenue
        
        # Reduce position
        pos.quantity -= quantity
        if pos.quantity <= 0:
            db_session.delete(pos)
            
        # Insert PaperTransaction
        tx = PaperTransaction(
            action='SELL',
            ticker=ticker,
            quantity=quantity,
            price=price,
            total=total_revenue
        )
        db_session.add(tx)
        
        db_session.commit()
        return {
            'success': True, 
            'ticker': ticker, 
            'quantity': quantity, 
            'price': price, 
            'total': total_revenue
        }

    @staticmethod
    def get_positions(db_session: Session) -> list[dict]:
        """
        Query all PaperPosition rows
        For each, get current price and compute: 
        current_value = qty * current_price, pnl = current_value - qty * avg_cost, 
        pnl_pct = pnl / (qty * avg_cost) * 100
        Return list of dicts with: ticker, quantity, avg_cost, current_price, current_value, pnl, pnl_pct
        """
        positions = db_session.query(PaperPosition).all()
        result = []
        for p in positions:
            current_price = PaperTradingService.get_current_price(p.ticker)
            current_value = p.quantity * current_price
            cost_basis = p.quantity * p.avg_cost
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis != 0 else 0
            
            result.append({
                'ticker': p.ticker,
                'quantity': p.quantity,
                'avg_cost': p.avg_cost,
                'current_price': current_price,
                'current_value': current_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
        return result

    @staticmethod
    def get_portfolio_summary(db_session: Session) -> dict:
        """
        account = get_or_create_account()
        positions = get_positions()
        total_invested = sum(p['current_value'] for p in positions)
        total_value = account.current_cash + total_invested
        total_pnl = total_value - account.initial_cash
        Return: {cash, total_invested, total_value, total_pnl, pnl_pct, initial_cash}
        """
        account = PaperTradingService.get_or_create_account(db_session)
        positions = PaperTradingService.get_positions(db_session)
        
        total_invested = sum(p['current_value'] for p in positions)
        total_value = account.current_cash + total_invested
        total_pnl = total_value - account.initial_cash
        pnl_pct = (total_pnl / account.initial_cash * 100) if account.initial_cash != 0 else 0
        
        return {
            'cash': account.current_cash,
            'total_invested': total_invested,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'initial_cash': account.initial_cash
        }

    @staticmethod
    def get_transactions(db_session: Session, limit=50) -> list[dict]:
        """
        Query PaperTransaction ordered by timestamp desc, limit
        Return list of dicts
        """
        txs = db_session.query(PaperTransaction).order_by(PaperTransaction.timestamp.desc()).limit(limit).all()
        return [
            {
                'id': t.id,
                'action': t.action,
                'ticker': t.ticker,
                'quantity': t.quantity,
                'price': t.price,
                'total': t.total,
                'timestamp': t.timestamp
            } for t in txs
        ]

    @staticmethod
    def reset_account(db_session: Session, new_initial_cash: float = None):
        account = PaperTradingService.get_or_create_account(db_session)
        db_session.query(PaperPosition).delete()
        db_session.query(PaperTransaction).delete()
        if new_initial_cash is not None and new_initial_cash > 0:
            account.initial_cash = new_initial_cash
        account.current_cash = account.initial_cash
        db_session.commit()
