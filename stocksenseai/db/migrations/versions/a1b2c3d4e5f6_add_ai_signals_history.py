"""add_ai_signals_history

Revision ID: a1b2c3d4e5f6
Revises: c891a3d7f045
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "c891a3d7f045"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ai_signals_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_code", sa.String(10), nullable=False),
        sa.Column("signal", sa.String(10), nullable=False),
        sa.Column("signal_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("tech_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("lstm_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("rsi", sa.Numeric(8, 4), nullable=True),
        sa.Column("macd", sa.Numeric(12, 4), nullable=True),
        sa.Column("predicted_prices", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 2), nullable=True),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ai_signals_history_stock_code", "ai_signals_history", ["stock_code"])
    op.create_index("ix_ai_signals_history_recorded_at", "ai_signals_history", ["recorded_at"])


def downgrade() -> None:
    op.drop_index("ix_ai_signals_history_recorded_at", table_name="ai_signals_history")
    op.drop_index("ix_ai_signals_history_stock_code", table_name="ai_signals_history")
    op.drop_table("ai_signals_history")
