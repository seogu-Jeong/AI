"""add_server_defaults_and_selector_notnull

Revision ID: c891a3d7f045
Revises: b465250cc7d4
Create Date: 2026-06-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c891a3d7f045'
down_revision: Union[str, Sequence[str], None] = 'b465250cc7d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add server_default to columns that only had Python-side defaults
    op.alter_column('users', 'is_verified',
                    existing_type=sa.Boolean(),
                    server_default='false',
                    existing_nullable=False)
    op.alter_column('users', 'mode',
                    existing_type=sa.String(length=20),
                    server_default='demo',
                    existing_nullable=False)
    op.alter_column('users', 'dark_mode',
                    existing_type=sa.Boolean(),
                    server_default='true',
                    existing_nullable=False)
    op.alter_column('refresh_tokens', 'revoked',
                    existing_type=sa.Boolean(),
                    server_default='false',
                    existing_nullable=False)

    # Backfill existing rows that have NULL selector (rows inserted before this migration)
    op.execute(
        "UPDATE refresh_tokens SET selector = LEFT(MD5(token_hash), 16) WHERE selector IS NULL"
    )

    # Now make selector NOT NULL
    op.alter_column('refresh_tokens', 'selector',
                    existing_type=sa.String(length=16),
                    nullable=False)


def downgrade() -> None:
    op.alter_column('refresh_tokens', 'selector',
                    existing_type=sa.String(length=16),
                    nullable=True)
    op.alter_column('refresh_tokens', 'revoked',
                    existing_type=sa.Boolean(),
                    server_default=None,
                    existing_nullable=False)
    op.alter_column('users', 'dark_mode',
                    existing_type=sa.Boolean(),
                    server_default=None,
                    existing_nullable=False)
    op.alter_column('users', 'mode',
                    existing_type=sa.String(length=20),
                    server_default=None,
                    existing_nullable=False)
    op.alter_column('users', 'is_verified',
                    existing_type=sa.Boolean(),
                    server_default=None,
                    existing_nullable=False)
