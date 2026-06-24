"""add_signals_composite_index

Revision ID: e5f6a1b2c3d4
Revises: d4e5f6a1b2c3
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

revision: str = "e5f6a1b2c3d4"
down_revision: Union[str, None] = "d4e5f6a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "idx_signals_code_date",
        "ai_signals_history",
        ["stock_code", "recorded_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_signals_code_date", table_name="ai_signals_history")
