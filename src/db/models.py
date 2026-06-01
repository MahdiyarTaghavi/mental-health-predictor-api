import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    predictions: Mapped[list["PredictionRecord"]] = relationship(
        "PredictionRecord", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"


class PredictionRecord(Base):
    """Stores every call to POST /predict — inputs, result, and SHAP explanation."""

    __tablename__ = "prediction_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Raw survey inputs stored as JSON so schema changes don't require a migration
    input_data: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Prediction output
    prediction: Mapped[str] = mapped_column(String(50), nullable=False)   # "likely" | "unlikely"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # SHAP explanation as JSON — list of {feature, contribution} dicts
    shap_explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Optional: store the request source for audit purposes
    client_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    user: Mapped["User | None"] = relationship("User", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<PredictionRecord id={self.id} prediction={self.prediction} confidence={self.confidence:.2f}>"
