"""create users and prediction_records tables

Revision ID: 0001
Revises:
Create Date: 2026-05-30
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "prediction_records",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("input_data", sa.JSON, nullable=False),
        sa.Column("prediction", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("shap_explanation", sa.JSON, nullable=True),
        sa.Column("client_ip", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_prediction_records_user_id", "prediction_records", ["user_id"])
    op.create_index("ix_prediction_records_created_at", "prediction_records", ["created_at"])


def downgrade() -> None:
    op.drop_table("prediction_records")
    op.drop_table("users")
