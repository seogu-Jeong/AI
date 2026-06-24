"""add_risk_alert_columns

Revision ID: f6a1b2c3d4e5
Revises: e5f6a1b2c3d4
Create Date: 2026-06-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6a1b2c3d4e5"
down_revision: Union[str, None] = "e5f6a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "risk_settings",
        sa.Column("enforce_hard_stop", sa.Boolean(), server_default="true", nullable=True),
    )
    op.add_column(
        "alert_settings",
        sa.Column("notification_email", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("alert_settings", "notification_email")
    op.drop_column("risk_settings", "enforce_hard_stop")
