from sqlalchemy.orm import Session

from src.db.models import PredictionRecord, User


# Prediction records

def create_prediction_record(
    db: Session,
    *,
    input_data: dict,
    prediction: str,
    confidence: float,
    model_name: str,
    shap_explanation: dict | None = None,
    user_id: str | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> PredictionRecord:
    record = PredictionRecord(
        input_data=input_data,
        prediction=prediction,
        confidence=confidence,
        model_name=model_name,
        shap_explanation=shap_explanation,
        user_id=user_id,
        client_ip=client_ip,
        user_agent=user_agent,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_prediction_history(
    db: Session,
    *,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[PredictionRecord]:
    query = db.query(PredictionRecord)
    if user_id:
        query = query.filter(PredictionRecord.user_id == user_id)
    return (
        query.order_by(PredictionRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_prediction_by_id(db: Session, record_id: str) -> PredictionRecord | None:
    return db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()


# Users

def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, *, email: str, hashed_password: str) -> User:
    user = User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
