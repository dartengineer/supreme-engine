"""API tests. They use a temporary in-memory database, so your real data is never touched.
Run with:  pytest -v
"""
import os

os.environ["DATABASE_URL"] = "sqlite://"  # must be set before the app is imported
os.environ["SECRET_KEY"] = "test-secret-key-that-is-at-least-32-bytes-long"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


def signup_and_login(client, email="ada@example.com", password="strongpass123"):
    client.post("/auth/signup", json={"email": email, "password": password})
    token = client.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def user_with_profile(client, email="ada@example.com", name="Ada Obi"):
    headers = signup_and_login(client, email)
    client.put("/profile", json={"full_name": name}, headers=headers)
    return headers


def check_crud(client, path, payload, changes, section):
    """Adds, lists, edits and deletes one item of a section, and checks GET /profile includes it."""
    h = user_with_profile(client)

    r = client.post(path, json=payload, headers=h)
    assert r.status_code == 201, r.text
    item = r.json()
    for field, value in payload.items():
        assert item[field] == value

    assert [i["id"] for i in client.get(path, headers=h).json()] == [item["id"]]
    assert client.get("/profile", headers=h).json()[section][0]["id"] == item["id"]

    r = client.put(f"{path}/{item['id']}", json={**payload, **changes}, headers=h)
    assert r.status_code == 200, r.text
    for field, value in changes.items():
        assert r.json()[field] == value

    assert client.delete(f"{path}/{item['id']}", headers=h).status_code == 204
    assert client.get(path, headers=h).json() == []


def check_blocked_for_other_user(client, path, payload):
    """User B can't see, edit or delete User A's item; User A still can."""
    a = user_with_profile(client, "a@example.com", "User A")
    item_id = client.post(path, json=payload, headers=a).json()["id"]

    b = user_with_profile(client, "b@example.com", "User B")
    assert client.get(path, headers=b).json() == []
    assert client.put(f"{path}/{item_id}", json=payload, headers=b).status_code == 404
    assert client.delete(f"{path}/{item_id}", headers=b).status_code == 404
    assert client.delete(f"{path}/{item_id}", headers=a).status_code == 204


def test_health(client):
    assert client.get("/").json()["status"] == "ok"


def test_signup_and_login(client):
    r = client.post("/auth/signup", json={"email": "Ada@Example.com", "password": "strongpass123"})
    assert r.status_code == 201
    assert r.json()["email"] == "ada@example.com"
    headers = signup_and_login(client)
    assert client.get("/auth/me", headers=headers).json()["email"] == "ada@example.com"


def test_duplicate_signup_rejected(client):
    client.post("/auth/signup", json={"email": "ada@example.com", "password": "strongpass123"})
    r = client.post("/auth/signup", json={"email": "ada@example.com", "password": "strongpass123"})
    assert r.status_code == 400


def test_wrong_password_rejected(client):
    client.post("/auth/signup", json={"email": "ada@example.com", "password": "strongpass123"})
    r = client.post("/auth/login", json={"email": "ada@example.com", "password": "wrongpass123"})
    assert r.status_code == 401


def test_profile_requires_login(client):
    assert client.get("/profile").status_code in (401, 403)


def test_full_profile_flow(client):
    headers = signup_and_login(client)
    r = client.put("/profile", json={"full_name": "Ada Obi", "headline": "Graphic Designer"}, headers=headers)
    assert r.status_code == 200
    exp = {"job_title": "Graphic Designer", "company": "Acme", "start_date": "2022-01-01", "description": "Brand design"}
    r = client.post("/profile/experiences", json=exp, headers=headers)
    assert r.status_code == 201
    full = client.get("/profile", headers=headers).json()
    assert full["full_name"] == "Ada Obi"
    assert full["experiences"][0]["company"] == "Acme"


def test_cannot_touch_another_users_experience(client):
    a = signup_and_login(client, "a@example.com")
    client.put("/profile", json={"full_name": "User A"}, headers=a)
    exp_id = client.post(
        "/profile/experiences",
        json={"job_title": "VA", "company": "X", "start_date": "2023-01-01"},
        headers=a,
    ).json()["id"]

    b = signup_and_login(client, "b@example.com")
    client.put("/profile", json={"full_name": "User B"}, headers=b)
    assert client.delete(f"/profile/experiences/{exp_id}", headers=b).status_code == 404
    assert client.delete(f"/profile/experiences/{exp_id}", headers=a).status_code == 204


# ---------- Education ----------
EDUCATION = {
    "institution": "University of Lagos",
    "qualification": "B.Sc.",
    "field_of_study": "Computer Science",
    "start_date": "2016-09-01",
    "end_date": "2020-07-31",
}


def test_education_crud(client):
    check_crud(client, "/profile/education", EDUCATION, {"grade": "First Class"}, "education")


def test_cannot_touch_another_users_education(client):
    check_blocked_for_other_user(client, "/profile/education", EDUCATION)


# ---------- Skills ----------
SKILL = {"name": "Figma", "category": "Design tools", "level": "Intermediate"}


def test_skill_crud(client):
    check_crud(client, "/profile/skills", SKILL, {"level": "Expert"}, "skills")


def test_cannot_touch_another_users_skill(client):
    check_blocked_for_other_user(client, "/profile/skills", SKILL)


# ---------- Projects ----------
PROJECT = {
    "name": "Brand refresh for a local bakery",
    "role": "Lead designer",
    "url": "https://example.com/bakery",
    "technologies": "Figma, Illustrator",
    "start_date": "2024-03-01",
}


def test_project_crud(client):
    check_crud(client, "/profile/projects", PROJECT, {"end_date": "2024-05-31"}, "projects")


def test_cannot_touch_another_users_project(client):
    check_blocked_for_other_user(client, "/profile/projects", PROJECT)


# ---------- Certifications ----------
CERTIFICATION = {
    "name": "Google UX Design Certificate",
    "issuer": "Google",
    "issue_date": "2023-06-15",
    "credential_url": "https://example.com/cert/123",
}


def test_certification_crud(client):
    check_crud(client, "/profile/certifications", CERTIFICATION, {"expiry_date": "2026-06-15"}, "certifications")


def test_cannot_touch_another_users_certification(client):
    check_blocked_for_other_user(client, "/profile/certifications", CERTIFICATION)


# ---------- Job applications & CV generation ----------
# These tests NEVER call the real AI: generate_cv is replaced with a fake that returns "# Test CV".
from app.services import ai  # noqa: E402


@pytest.fixture(autouse=True)
def block_real_ai(monkeypatch):
    """Safety net for every test: if anything tries to build a real AI client, the test fails."""
    def no_real_calls():
        raise AssertionError("A test tried to call the real AI API. Monkeypatch ai.generate_cv instead.")
    monkeypatch.setattr(ai, "_client", no_real_calls)


@pytest.fixture()
def fake_ai(monkeypatch):
    """Replaces generate_cv with a fake and records what it was called with."""
    calls = []

    def fake_generate_cv(profile_json, job_title, job_description):
        calls.append({"profile_json": profile_json, "job_title": job_title, "job_description": job_description})
        return "# Test CV"

    monkeypatch.setattr(ai, "generate_cv", fake_generate_cv)
    return calls


APPLICATION = {"job_title": "Graphic Designer", "company": "Acme", "job_description": "Design brand assets."}


def test_application_create_list_get(client):
    h = user_with_profile(client)
    r = client.post("/applications", json=APPLICATION, headers=h)
    assert r.status_code == 201, r.text
    app_id = r.json()["id"]
    assert r.json()["status"] == "draft"
    assert [a["id"] for a in client.get("/applications", headers=h).json()] == [app_id]
    assert client.get(f"/applications/{app_id}", headers=h).json()["job_title"] == "Graphic Designer"


def test_generate_cv_twice_creates_versions_1_and_2(client, fake_ai):
    h = user_with_profile(client, name="Ada Obi")
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]

    first = client.post(f"/applications/{app_id}/generate-cv", headers=h)
    second = client.post(f"/applications/{app_id}/generate-cv", headers=h)
    assert first.status_code == 201, first.text
    assert (first.json()["version"], second.json()["version"]) == (1, 2)
    assert first.json()["content"] == "# Test CV"
    assert first.json()["doc_type"] == "cv"
    assert first.json()["model_used"]

    # The AI received the full profile and the job details.
    assert "Ada Obi" in fake_ai[0]["profile_json"]
    assert fake_ai[0]["job_title"] == "Graphic Designer"
    assert fake_ai[0]["job_description"] == "Design brand assets."

    history = client.get(f"/applications/{app_id}/documents", headers=h).json()
    assert [d["version"] for d in history] == [2, 1]


def test_cannot_use_another_users_application(client, fake_ai):
    a = user_with_profile(client, "a@example.com", "User A")
    app_id = client.post("/applications", json=APPLICATION, headers=a).json()["id"]

    b = user_with_profile(client, "b@example.com", "User B")
    assert client.get("/applications", headers=b).json() == []
    assert client.get(f"/applications/{app_id}", headers=b).status_code == 404
    assert client.post(f"/applications/{app_id}/generate-cv", headers=b).status_code == 404
    assert client.get(f"/applications/{app_id}/documents", headers=b).status_code == 404
    assert fake_ai == []  # the AI was never called for user B


def test_generate_cv_needs_a_profile(client, fake_ai):
    h = signup_and_login(client)  # logged in, but no profile yet
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    assert client.post(f"/applications/{app_id}/generate-cv", headers=h).status_code == 404


def test_ai_failure_returns_502_and_saves_nothing(client, monkeypatch):
    def failing_generate_cv(*args):
        raise ai.AIError("The AI service is busy or out of credit. Try again shortly.")

    monkeypatch.setattr(ai, "generate_cv", failing_generate_cv)
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    r = client.post(f"/applications/{app_id}/generate-cv", headers=h)
    assert r.status_code == 502
    assert "busy" in r.json()["detail"]
    assert client.get(f"/applications/{app_id}/documents", headers=h).json() == []
