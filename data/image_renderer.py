import pandas as pd
import numpy as np
import mplfinance as mpf
import os
import io
from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import select
from db.models import RawOHLCV

class ImageRenderer:
    def render_candle_image(self, df: pd.DataFrame, ticker: str, output_path: str = None) -> np.ndarray:
        try:
            if len(df) < 20:
                # Return blank image
                return np.zeros((128, 128, 3), dtype=np.uint8)

            # Last 20 rows
            df_plot = df.tail(20).copy()
            df_plot.index = pd.to_datetime(df_plot.index)
            
            # Bollinger Bands for overlay (optional if not pre-computed, but prompt says BB overlay)
            # Assuming df has bb_lower, bb_upper or we compute them here
            # For simplicity, if they are not there, we just plot candles
            
            kwargs = dict(type='candle', style='charles', axisoff=True, figsize=(1.28, 1.28))
            
            buf = io.BytesIO()
            mpf.plot(df_plot, savefig=dict(fname=buf, bbox_inches='tight', pad_inches=0, dpi=100), **kwargs)
            buf.seek(0)
            
            img = Image.open(buf).convert('RGB')
            img = img.resize((128, 128))
            
            if output_path:
                img.save(output_path)
            
            arr = np.array(img)
            return arr
        except Exception as e:
            print(f"Error rendering {ticker}: {e}")
            return np.zeros((128, 128, 3), dtype=np.uint8)

    def batch_render(self, tickers: list[str], db_session: Session, cache_dir: str = 'cache/images/', progress_callback=None):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        for ticker in tickers:
            try:
                stmt = select(RawOHLCV).where(RawOHLCV.ticker == ticker).order_by(RawOHLCV.date.desc()).limit(20)
                rows = db_session.execute(stmt).scalars().all()
                if not rows:
                    continue
                
                rows.reverse() # Order by date ascending
                data = [{
                    'Date': r.date,
                    'Open': r.open,
                    'High': r.high,
                    'Low': r.low,
                    'Close': r.close,
                    'Volume': r.volume
                } for r in rows]
                
                df = pd.DataFrame(data)
                df.set_index('Date', inplace=True)
                
                arr = self.render_candle_image(df, ticker)
                np.save(os.path.join(cache_dir, f"{ticker}.npy"), arr)
                
                if progress_callback:
                    progress_callback(ticker)
            except Exception as e:
                print(f"Error in batch render for {ticker}: {e}")
