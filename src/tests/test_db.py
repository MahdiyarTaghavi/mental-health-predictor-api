"""
Tests for the database layer.

Uses an in-memory SQLite database so no running Postgres is needed in CI.
SQLAlchemy's ORM is DB-agnostic — the same models work on SQLite and PostgreSQL.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import Base
from db.crud import create_prediction_record, get_prediction_history, get_prediction_by_id
from db.models import PredictionRecord


SAMPLE_INPUT = {
    "age": 30,
    "gender": "Male",
    "self_employed": "No",
    "family_history": "Yes",
    "work_interfere": "Sometimes",
    "no_employees": "26-100",
    "remote_work": "No",
    "tech_company": "Yes",
    "benefits": "Yes",
    "care_options": "Not sure",
    "wellness_program": "No",
    "seek_help": "Yes",
    "anonymity": "Yes",
    "leave": "Somewhat easy",
    "mental_health_consequence": "No",
    "phys_health_consequence": "No",
    "coworkers": "Some of them",
    "supervisor": "Yes",
    "mental_health_interview": "No",
    "phys_health_interview": "Maybe",
    "mental_vs_physical": "Yes",
    "obs_consequence": "No",
}


@pytest.fixture()
def db_session():
    """Provide a clean in-memory SQLite session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def test_create_prediction_record(db_session):
    record = create_prediction_record(
        db_session,
        input_data=SAMPLE_INPUT,
        prediction="likely",
        confidence=0.87,
        model_name="XGBClassifier",
        shap_explanation={"age": 0.12, "family_history": 0.35},
    )

    assert record.id is not None
    assert record.prediction == "likely"
    assert record.confidence == pytest.approx(0.87)
    assert record.shap_explanation["family_history"] == 0.35


def test_get_prediction_history(db_session):
    for i in range(3):
        create_prediction_record(
            db_session,
            input_data=SAMPLE_INPUT,
            prediction="likely" if i % 2 == 0 else "unlikely",
            confidence=0.7 + i * 0.05,
            model_name="XGBClassifier",
        )

    records = get_prediction_history(db_session, limit=10)
    assert len(records) == 3
    # Should be newest-first
    assert records[0].confidence >= records[-1].confidence


def test_get_prediction_by_id(db_session):
    record = create_prediction_record(
        db_session,
        input_data=SAMPLE_INPUT,
        prediction="unlikely",
        confidence=0.61,
        model_name="XGBClassifier",
    )

    fetched = get_prediction_by_id(db_session, record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.prediction == "unlikely"


def test_get_prediction_by_id_not_found(db_session):
    result = get_prediction_by_id(db_session, "nonexistent-id")
    assert result is None
