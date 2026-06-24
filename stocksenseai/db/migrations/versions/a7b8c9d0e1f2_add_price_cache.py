"""add_price_cache

Revision ID: a7b8c9d0e1f2
Revises: f6a1b2c3d4e5
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "price_cache",
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("close_price", sa.Numeric(12, 2), nullable=False),
        sa.Column("market", sa.String(10), nullable=False, server_default="KR"),
        sa.PrimaryKeyConstraint("ticker", "trade_date", "market"),
    )
    op.create_index("idx_price_cache_ticker_market", "price_cache", ["ticker", "market"])


def downgrade() -> None:
    op.drop_index("idx_price_cache_ticker_market", table_name="price_cache")
    op.drop_table("price_cache")
