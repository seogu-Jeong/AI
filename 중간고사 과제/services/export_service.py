import os
import pandas as pd
from datetime import datetime
from app_paths import get_user_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import json
from db.models import AIScore, Portfolio
from sqlalchemy import desc, func

class ExportService:
    def _ensure_exports_dir(self):
        exports_path = str(get_user_file('exports'))
        if not os.path.exists(exports_path):
            os.makedirs(exports_path)

    def export_screening_csv(self, data: list[dict], filepath: str = None) -> str:
        self._ensure_exports_dir()
        if not filepath:
            filepath = str(get_user_file(f'exports/screening_{datetime.now():%Y%m%d_%H%M%S}.csv'))
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath

    def export_portfolio_csv(self, portfolio: dict, filepath: str = None) -> str:
        self._ensure_exports_dir()
        if not filepath:
            filepath = str(get_user_file(f'exports/portfolio_{datetime.now():%Y%m%d_%H%M%S}.csv'))
        
        # Meta information as header rows
        meta_data = [
            ['Risk Level', portfolio['risk_level']],
            ['Total Amount', portfolio['total_amount']],
            ['Cash Percentage', portfolio['cash_pct']],
            [], # Empty row separator
            ['Ticker', 'Weight (%)', 'Amount ($)', 'Score', 'Reason']
        ]
        
        # Holdings data
        holdings_data = []
        for h in portfolio['holdings']:
            holdings_data.append([
                h['ticker'],
                h['weight'],
                h['amount'],
                h['score'],
                h['reason']
            ])
        
        full_data = meta_data + holdings_data
        df = pd.DataFrame(full_data)
        df.to_csv(filepath, index=False, header=False, encoding='utf-8-sig')
        return filepath

    def export_watchlist_csv(self, watchlist: list[dict], filepath: str = None) -> str:
        self._ensure_exports_dir()
        if not filepath:
            filepath = str(get_user_file(f'exports/watchlist_{datetime.now():%Y%m%d_%H%M%S}.csv'))
        
        df = pd.DataFrame(watchlist)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath

    def export_pdf_report(self, db_session, output_path: str = None) -> str:
        self._ensure_exports_dir()
        if not output_path:
            output_path = str(get_user_file(f'exports/stocksense_report_{datetime.now():%Y%m%d}.pdf'))

        # Register Korean font (specific to macOS as per project context)
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        if not os.path.exists(font_path):
            font_path = "/System/Library/Fonts/AppleGothic.ttf"
        
        font_name = "AppleGothic"
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
            except:
                font_name = "Helvetica"
        else:
            font_name = "Helvetica"

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        
        ko_title_style = ParagraphStyle(
            'KoTitle',
            parent=styles['Title'],
            fontName=font_name,
            fontSize=18,
            spaceAfter=12
        )
        ko_heading_style = ParagraphStyle(
            'KoHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6
        )
        ko_body_style = ParagraphStyle(
            'KoBody',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
        )

        # Header
        today_str = datetime.now().strftime('%Y-%m-%d')
        elements.append(Paragraph("StockSense AI — 종목 분석 리포트", ko_title_style))
        elements.append(Paragraph(f"날짜: {today_str}", ko_body_style))
        elements.append(Paragraph("Disclaimer: 본 리포트는 인공지능 모델의 분석 결과이며 투자 권유가 아닙니다.", ko_body_style))
        elements.append(Spacer(1, 20))

        # Section 1: AI 스크리닝 상위 30 종목
        elements.append(Paragraph("AI 스크리닝 상위 30 종목", ko_heading_style))
        
        latest_date = db_session.query(func.max(AIScore.date)).scalar()
        if latest_date:
            top_stocks = db_session.query(AIScore).filter(AIScore.date == latest_date).order_by(desc(AIScore.ensemble_score)).limit(30).all()
        else:
            top_stocks = []
            
        table_data = [['Rank', 'Ticker', 'AI Score', 'Signal', 'Direction', 'Pattern']]
        for i, s in enumerate(top_stocks):
            table_data.append([
                i + 1,
                s.ticker,
                f"{s.ensemble_score:.2f}",
                s.mlp_signal,
                s.lstm_dir_5d,
                s.cnn_pattern
            ])
        
        if not top_stocks:
            table_data.append(['-', '-', '-', '-', '-', '-'])

        t = Table(table_data, colWidths=[40, 60, 60, 60, 80, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        # Section 2: 포트폴리오 요약
        portfolio = db_session.query(Portfolio).order_by(desc(Portfolio.created_at)).first()
        if portfolio:
            elements.append(Paragraph("포트폴리오 요약", ko_heading_style))
            elements.append(Paragraph(f"이름: {portfolio.name} (위험도: {portfolio.risk_level})", ko_body_style))
            
            try:
                holdings = json.loads(portfolio.holdings)
                if holdings:
                    p_data = [['Ticker', 'Weight (%)', 'Amount ($)', 'Score']]
                    for h in holdings:
                        p_data.append([
                            h.get('ticker', '-'),
                            f"{h.get('weight', 0):.1f}",
                            f"{h.get('amount', 0):,.0f}",
                            f"{h.get('score', 0):.1f}"
                        ])
                    
                    pt = Table(p_data, colWidths=[80, 80, 100, 80])
                    pt.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('FONTNAME', (0, 0), (-1, -1), font_name),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                    ]))
                    elements.append(pt)
            except:
                pass

        def add_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont(font_name, 9)
            canvas.drawCentredString(A4[0]/2, 30, "본 리포트는 투자 조언이 아닙니다. StockSense AI")
            canvas.restoreState()

        doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
        return output_path
