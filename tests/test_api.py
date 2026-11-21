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
