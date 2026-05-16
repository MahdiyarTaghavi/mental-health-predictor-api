import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from main import app

@pytest.fixture(scope="session")
def client():
    """
    Creates a single test client reused across all tests.
    """
    return TestClient(app)


@pytest.fixture(scope="session")
def valid_payload():
    """Valid predict request payload reused across multiple tests."""
    return {
        "Age": 28,
        "Gender": "Male",
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

@pytest.fixture
def blank_image():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

@pytest.fixture
def fake_pdf():
    return io.BytesIO(b"%PDF-1.4 fake content")

@pytest.fixture
def oversized_image():
    return io.BytesIO(b"0" * (5 * 1024 * 1024 + 1))