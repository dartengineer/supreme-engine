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
