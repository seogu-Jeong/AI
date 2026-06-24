"""add_auto_trade_tables

Revision ID: g7h8i9j0k1l2
Revises: f6a1b2c3d4e5
Create Date: 2026-06-20 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "g7h8i9j0k1l2"
down_revision: Union[str, None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "auto_trade_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("mode", sa.String(10), server_default="paper", nullable=False),
        sa.Column("total_budget", sa.Integer(), nullable=False, server_default="1000000"),
        sa.Column("budget_per_trade", sa.Integer(), nullable=False, server_default="100000"),
        sa.Column("max_positions", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("signal_threshold", sa.Integer(), nullable=False, server_default="70"),
        sa.Column("stop_loss_pct", sa.Float(), nullable=False, server_default="5.0"),
        sa.Column("take_profit_pct", sa.Float(), nullable=False, server_default="10.0"),
        sa.Column("watch_codes", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index("idx_auto_trade_config_user", "auto_trade_configs", ["user_id"])

    op.create_table(
        "auto_trade_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stock_code", sa.String(10), nullable=False),
        sa.Column("stock_name", sa.String(100), nullable=True),
        sa.Column("action", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("price", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_amount", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reason", sa.String(200), nullable=True),
        sa.Column("signal_score", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("mode", sa.String(10), nullable=False, server_default="paper"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_auto_trade_logs_user", "auto_trade_logs", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_auto_trade_logs_user", table_name="auto_trade_logs")
    op.drop_table("auto_trade_logs")
    op.drop_index("idx_auto_trade_config_user", table_name="auto_trade_configs")
    op.drop_table("auto_trade_configs")
