"""add_portfolios_trades

Revision ID: c3d4e5f6a1b2
Revises: b2c3d4e5f6a1
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c3d4e5f6a1b2"
down_revision: Union[str, None] = "b2c3d4e5f6a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "portfolios",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_code", sa.String(10), nullable=False),
        sa.Column("stock_name", sa.String(100), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("avg_price", sa.Numeric(12, 2), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "stock_code", "mode"),
    )
    op.create_index("idx_portfolios_user", "portfolios", ["user_id", "mode"])
    op.create_table(
        "trades",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_code", sa.String(10), nullable=False),
        sa.Column("stock_name", sa.String(100), nullable=True),
        sa.Column("order_type", sa.String(10), nullable=False),
        sa.Column("price_type", sa.String(10), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("order_price", sa.Numeric(12, 2), nullable=True),
        sa.Column("executed_price", sa.Numeric(12, 2), nullable=True),
        sa.Column("commission", sa.Numeric(10, 2), server_default="0", nullable=True),
        sa.Column("status", sa.String(20), server_default="PENDING", nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("kis_order_no", sa.String(50), nullable=True),
        sa.Column("ai_signal_at_order", sa.String(10), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_trades_user_date", "trades", ["user_id", "created_at"])
    op.create_index("idx_trades_status", "trades", ["user_id", "status", "mode"])


def downgrade() -> None:
    op.drop_index("idx_trades_status", table_name="trades")
    op.drop_index("idx_trades_user_date", table_name="trades")
    op.drop_table("trades")
    op.drop_index("idx_portfolios_user", table_name="portfolios")
    op.drop_table("portfolios")
