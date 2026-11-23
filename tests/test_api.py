"""API tests. They run against the cv_generator_test database in the Docker PostgreSQL (with pgvector),
which is wiped before every test, so your real data is never touched.
Run with:  pytest -v      (set TEST_DATABASE_URL to use a different test database)
"""
import json
import os
import pathlib
import re
import zlib

from dotenv import dotenv_values
from sqlalchemy.engine import make_url


def _test_database_url() -> str:
    """Same server and login as your app's DATABASE_URL, but the cv_generator_test database."""
    if os.environ.get("TEST_DATABASE_URL"):
        url = make_url(os.environ["TEST_DATABASE_URL"])
    else:
        env_file = pathlib.Path(__file__).resolve().parent.parent / ".env"
        url = make_url(dotenv_values(env_file).get("DATABASE_URL") or os.environ["DATABASE_URL"])
        url = url.set(database="cv_generator_test")
    # Every test drops all tables, so refuse anything that isn't clearly a test database.
    if not (url.database or "").endswith("_test"):
        raise RuntimeError(f"Refusing to run tests against database {url.database!r}: its name must end in '_test'.")
    return url.render_as_string(hide_password=False)


os.environ["DATABASE_URL"] = _test_database_url()  # must be set before the app is imported
os.environ["SECRET_KEY"] = "test-secret-key-that-is-at-least-32-bytes-long"
os.environ["WARM_UP_EMBEDDINGS"] = "false"  # tests use fake_embed, never the real model

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services import embeddings  # noqa: E402


def fake_embed(texts):
    """Fast, deterministic stand-in for the real model: a bag of words hashed into 384 slots.

    Texts that share words get similar vectors, so retrieval still behaves sensibly in tests.
    """
    vectors = []
    for text in texts:
        v = [0.0] * embeddings.DIMENSIONS
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(word.encode()) % embeddings.DIMENSIONS] += 1.0
        if not any(v):
            v[0] = 1.0
        norm = sum(x * x for x in v) ** 0.5
        vectors.append([x / norm for x in v])
    return vectors


@pytest.fixture(autouse=True)
def no_real_embedding_model(monkeypatch):
    """Every test uses fake_embed, so the real model is never loaded."""
    monkeypatch.setattr(embeddings, "embed", fake_embed)


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
from app.schemas import JobRequirements  # noqa: E402
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

    def fake_generate_cv(profile_json, job_title, job_description, requirements=None, feedback=None):
        calls.append({
            "feedback": feedback,
            "profile_json": profile_json,
            "job_title": job_title,
            "job_description": job_description,
            "requirements": requirements,
        })
        return "# Test CV"

    monkeypatch.setattr(ai, "generate_cv", fake_generate_cv)
    return calls


FAKE_REQUIREMENTS = JobRequirements(
    required_skills=["Canva", "Brand design"],
    nice_to_have_skills=["Photoshop"],
    keywords=["brand assets", "social media"],
    seniority="entry-level",
    key_responsibilities=["Design social media graphics"],
)


@pytest.fixture()
def fake_extract(monkeypatch):
    """Replaces extract_requirements with a fake that returns FAKE_REQUIREMENTS."""
    calls = []

    def fake_extract_requirements(job_title, job_description):
        calls.append({"job_title": job_title, "job_description": job_description})
        return FAKE_REQUIREMENTS

    monkeypatch.setattr(ai, "extract_requirements", fake_extract_requirements)
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


# ---------- Job description analysis (v2) ----------
def test_analyze_saves_requirements(client, fake_extract):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    assert client.get(f"/applications/{app_id}", headers=h).json()["requirements"] is None

    r = client.post(f"/applications/{app_id}/analyze", headers=h)
    assert r.status_code == 200, r.text
    assert r.json() == FAKE_REQUIREMENTS.model_dump()
    assert fake_extract == [{"job_title": "Graphic Designer", "job_description": "Design brand assets."}]

    # Saved on the application, so it's still there when we read it back.
    assert client.get(f"/applications/{app_id}", headers=h).json()["requirements"] == FAKE_REQUIREMENTS.model_dump()


def test_generate_cv_uses_saved_requirements(client, fake_ai, fake_extract):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    client.post(f"/applications/{app_id}/analyze", headers=h)

    assert client.post(f"/applications/{app_id}/generate-cv", headers=h).status_code == 201
    assert fake_ai[0]["requirements"] == FAKE_REQUIREMENTS


def test_generate_cv_works_without_requirements(client, fake_ai):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]

    r = client.post(f"/applications/{app_id}/generate-cv", headers=h)
    assert r.status_code == 201, r.text
    assert fake_ai[0]["requirements"] is None


def test_analyze_needs_a_job_description(client, fake_extract):
    h = user_with_profile(client)
    app_id = client.post("/applications", json={"job_title": "Graphic Designer"}, headers=h).json()["id"]
    assert client.post(f"/applications/{app_id}/analyze", headers=h).status_code == 400
    assert fake_extract == []  # no AI call wasted


def test_cannot_analyze_another_users_application(client, fake_extract):
    a = user_with_profile(client, "a@example.com", "User A")
    app_id = client.post("/applications", json=APPLICATION, headers=a).json()["id"]

    b = user_with_profile(client, "b@example.com", "User B")
    assert client.post(f"/applications/{app_id}/analyze", headers=b).status_code == 404
    assert fake_extract == []


def test_analyze_failure_returns_502_and_saves_nothing(client, monkeypatch):
    def failing_extract(*args):
        raise ai.AIError("The AI service is busy. Try again shortly.")

    monkeypatch.setattr(ai, "extract_requirements", failing_extract)
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    assert client.post(f"/applications/{app_id}/analyze", headers=h).status_code == 502
    assert client.get(f"/applications/{app_id}", headers=h).json()["requirements"] is None


# ---------- Embeddings & retrieval (v3) ----------
from sqlalchemy import select  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.models import GeneratedDocument, ProfileChunk  # noqa: E402


def chunk_texts(source_type=None):
    with SessionLocal() as db:
        query = select(ProfileChunk.text)
        if source_type:
            query = query.where(ProfileChunk.source_type == source_type)
        return sorted(db.scalars(query).all())


DESIGNER_JOB = {
    "job_title": "Graphic Designer",
    "company": "Acme",
    "start_date": "2021-01-01",
    "description": "- Designed social media graphics in Canva\n- Answered customer phone calls",
}


def test_chunks_follow_profile_changes(client):
    h = user_with_profile(client)
    exp_id = client.post("/profile/experiences", json=DESIGNER_JOB, headers=h).json()["id"]
    # Each bullet becomes its own chunk.
    assert chunk_texts("experience") == ["Answered customer phone calls", "Designed social media graphics in Canva"]

    client.put(f"/profile/experiences/{exp_id}", json={**DESIGNER_JOB, "description": "- Built brand guidelines"}, headers=h)
    assert chunk_texts("experience") == ["Built brand guidelines"]

    client.post("/profile/skills", json={"name": "Canva", "category": "Design tools"}, headers=h)
    assert chunk_texts("skill") == ["Canva (Design tools)"]

    client.delete(f"/profile/experiences/{exp_id}", headers=h)
    assert chunk_texts("experience") == []


def test_matches_pick_relevant_bullets_with_scores(client, fake_extract):
    h = user_with_profile(client)
    client.post("/profile/experiences", json=DESIGNER_JOB, headers=h)
    client.post("/profile/skills", json={"name": "Canva", "category": "Design tools"}, headers=h)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    client.post(f"/applications/{app_id}/analyze", headers=h)

    r = client.get(f"/applications/{app_id}/matches", headers=h)
    assert r.status_code == 200, r.text
    matches = r.json()
    texts = [m["text"] for m in matches]
    assert "Designed social media graphics in Canva" in texts
    assert "Canva (Design tools)" in texts
    assert "Answered customer phone calls" not in texts  # unrelated to every requirement, so filtered out
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores, reverse=True) and all(0 < s <= 1 for s in scores)
    assert all(m["matched_requirement"] for m in matches)


def test_generate_cv_sends_only_selected_bullets(client, fake_ai, fake_extract):
    import json

    h = user_with_profile(client, name="Ada Obi")
    client.put("/profile", json={"full_name": "Ada Obi", "summary": "Expert in Kubernetes"}, headers=h)
    client.post("/profile/experiences", json=DESIGNER_JOB, headers=h)
    client.post("/profile/experiences", json={
        "job_title": "Warehouse Assistant", "company": "BoxCo", "start_date": "2023-05-01",
        "description": "- Stacked pallets in the warehouse",
    }, headers=h)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    client.post(f"/applications/{app_id}/analyze", headers=h)

    assert client.post(f"/applications/{app_id}/generate-cv", headers=h).status_code == 201
    sent = json.loads(fake_ai[0]["profile_json"])
    assert sent["full_name"] == "Ada Obi"

    # The relevant job comes with only its matching bullet.
    assert len(sent["experiences"]) == 1
    job = sent["experiences"][0]
    assert (job["job_title"], job["company"], job["start_date"]) == ("Graphic Designer", "Acme", "2021-01-01")
    assert job["selected_bullets"] == ["Designed social media graphics in Canva"]

    # The unrelated (and more recent) job is still listed, but compactly and with no bullets.
    assert sent["other_experience"] == [{
        "job_title": "Warehouse Assistant", "company": "BoxCo", "location": None,
        "start_date": "2023-05-01", "end_date": None,
    }]

    # Nothing unselected leaks through: no other bullets, and no free-text summary.
    assert "phone calls" not in fake_ai[0]["profile_json"]
    assert "pallets" not in fake_ai[0]["profile_json"]
    assert "summary" not in sent and "Kubernetes" not in fake_ai[0]["profile_json"]


def test_matches_need_analysis_first(client):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    assert client.get(f"/applications/{app_id}/matches", headers=h).status_code == 400


def test_cannot_see_another_users_matches(client, fake_extract):
    a = user_with_profile(client, "a@example.com", "User A")
    app_id = client.post("/applications", json=APPLICATION, headers=a).json()["id"]
    client.post(f"/applications/{app_id}/analyze", headers=a)

    b = user_with_profile(client, "b@example.com", "User B")
    assert client.get(f"/applications/{app_id}/matches", headers=b).status_code == 404


# ---------- Cover letters (v4) ----------
from datetime import date  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from app.services.generation import years_of_experience  # noqa: E402


@pytest.fixture()
def fake_letter(monkeypatch):
    """Replaces generate_cover_letter with a fake that records what it was given."""
    calls = []

    def fake_generate_cover_letter(context, company, motivation_notes, feedback=None):
        calls.append({"context": context, "company": company, "motivation_notes": motivation_notes,
                      "feedback": feedback})
        return "Dear Acme Team,\n\nTest letter.\n\nKind regards,\nAda Obi"

    monkeypatch.setattr(ai, "generate_cover_letter", fake_generate_cover_letter)
    return calls


def versions(client, app_id, headers, doc_type=None):
    url = f"/applications/{app_id}/documents" + (f"?doc_type={doc_type}" if doc_type else "")
    return [(d["doc_type"], d["version"]) for d in client.get(url, headers=headers).json()]


def test_versions_count_separately_per_document_type(client, fake_ai, fake_letter):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    for _ in range(3):
        client.post(f"/applications/{app_id}/generate-cv", headers=h)

    r = client.post(f"/applications/{app_id}/generate-cover-letter", headers=h)
    assert r.status_code == 201, r.text
    assert (r.json()["doc_type"], r.json()["version"]) == ("cover_letter", 1)  # v1, despite 3 CVs
    assert client.post(f"/applications/{app_id}/generate-cover-letter", headers=h).json()["version"] == 2

    assert versions(client, app_id, h, "cover_letter") == [("cover_letter", 2), ("cover_letter", 1)]
    assert versions(client, app_id, h, "cv") == [("cv", 3), ("cv", 2), ("cv", 1)]
    assert len(versions(client, app_id, h)) == 5
    assert client.get(f"/applications/{app_id}/documents?doc_type=essay", headers=h).status_code == 422


def test_cover_letter_with_and_without_motivation_notes(client, fake_letter):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]

    assert client.post(f"/applications/{app_id}/generate-cover-letter", headers=h).status_code == 201
    assert fake_letter[0]["motivation_notes"] is None
    assert fake_letter[0]["company"] == "Acme"

    r = client.patch(f"/applications/{app_id}", json={"motivation_notes": "I use their app every day."}, headers=h)
    assert r.status_code == 200 and r.json()["motivation_notes"] == "I use their app every day."
    assert client.post(f"/applications/{app_id}/generate-cover-letter", headers=h).status_code == 201
    assert fake_letter[1]["motivation_notes"] == "I use their app every day."

    # Blank notes count as no notes.
    client.patch(f"/applications/{app_id}", json={"motivation_notes": "   "}, headers=h)
    client.post(f"/applications/{app_id}/generate-cover-letter", headers=h)
    assert fake_letter[2]["motivation_notes"] is None


def test_cover_letter_uses_the_same_context_as_the_cv(client, fake_ai, fake_letter, fake_extract):
    h = user_with_profile(client)
    client.post("/profile/experiences", json=DESIGNER_JOB, headers=h)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    client.post(f"/applications/{app_id}/analyze", headers=h)

    client.post(f"/applications/{app_id}/generate-cv", headers=h)
    client.post(f"/applications/{app_id}/generate-cover-letter", headers=h)
    context = fake_letter[0]["context"]
    assert context.profile_json == fake_ai[0]["profile_json"]  # identical facts for both documents
    assert context.requirements == FAKE_REQUIREMENTS
    assert "Designed social media graphics in Canva" in [m.text for m in context.matches]
    assert '"total_experience"' in context.profile_json


def test_cannot_generate_cover_letter_for_another_user(client, fake_letter):
    a = user_with_profile(client, "a@example.com", "User A")
    app_id = client.post("/applications", json=APPLICATION, headers=a).json()["id"]

    b = user_with_profile(client, "b@example.com", "User B")
    assert client.post(f"/applications/{app_id}/generate-cover-letter", headers=b).status_code == 404
    assert client.patch(f"/applications/{app_id}", json={"motivation_notes": "x"}, headers=b).status_code == 404
    assert fake_letter == []


def test_cover_letter_failure_returns_502_and_saves_nothing(client, monkeypatch):
    def failing(*args):
        raise ai.AIError("The AI service is busy. Try again shortly.")

    monkeypatch.setattr(ai, "generate_cover_letter", failing)
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    assert client.post(f"/applications/{app_id}/generate-cover-letter", headers=h).status_code == 502
    assert versions(client, app_id, h) == []


def test_changing_the_job_ad_clears_old_requirements(client, fake_extract):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    client.post(f"/applications/{app_id}/analyze", headers=h)

    # Notes don't affect the ad, so the requirements stay.
    assert client.patch(f"/applications/{app_id}", json={"motivation_notes": "x"}, headers=h).json()["requirements"]
    # A new description does, so they're cleared.
    r = client.patch(f"/applications/{app_id}", json={"job_description": "A different ad."}, headers=h)
    assert r.json()["requirements"] is None


def test_years_of_experience_counts_overlaps_once():
    jobs = [
        SimpleNamespace(start_date=date(2020, 1, 1), end_date=date(2020, 12, 1)),  # 12 months
        SimpleNamespace(start_date=date(2020, 6, 1), end_date=date(2021, 5, 1)),  # overlaps 7, adds 5
        SimpleNamespace(start_date=date(2023, 1, 1), end_date=None),  # Jan-Jun 2023 = 6 (to "today")
    ]
    assert years_of_experience(jobs, today=date(2023, 6, 15)) == round(23 / 12, 1)
    assert years_of_experience([]) == 0


def test_unselected_skills_are_still_sent_as_other(client, fake_letter, fake_extract):
    h = user_with_profile(client)
    client.post("/profile/skills", json={"name": "Canva"}, headers=h)  # matches a requirement
    client.post("/profile/skills", json={"name": "Forklift licence"}, headers=h)  # matches nothing
    client.post("/profile/certifications", json={"name": "First Aid"}, headers=h)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    client.post(f"/applications/{app_id}/analyze", headers=h)

    client.post(f"/applications/{app_id}/generate-cover-letter", headers=h)
    sent = json.loads(fake_letter[0]["context"].profile_json)
    assert sent["skills"] == {"most_relevant": ["Canva"], "other": ["Forklift licence"]}
    # Nothing goes missing: a skill that wasn't selected is still visible, so it can't be mistaken for a gap.
    assert sent["certifications"]["other"] == [{"name": "First Aid", "issuer": None}]


# ---------- Cover letter rules enforced in code ----------
from app.schemas import GapAssessment  # noqa: E402


def _context(requirements=FAKE_REQUIREMENTS):
    return SimpleNamespace(profile_json='{"full_name": "Ada Obi"}', full_profile_json='{"full_name": "Ada Obi"}',
                           job_title="Graphic Designer", job_description="Design brand assets.",
                           requirements=requirements)


def _letter(words: int, greeting="Dear Sir,"):
    return f"{greeting}\n\n" + " ".join(["word"] * words) + "\n\nKind regards,\nAda Obi"


def test_letter_word_count_ignores_greeting_and_sign_off():
    assert ai.letter_body_word_count(_letter(260)) == 260


def test_choose_gaps_keeps_at_most_two_required_or_one_nice_to_have():
    assert ai.choose_gaps(GapAssessment(missing_required=["KYC", "AML", "Salesforce"], missing_nice_to_have=["Trello"])) == ["KYC", "AML"]
    assert ai.choose_gaps(GapAssessment(missing_required=[], missing_nice_to_have=["Trello", "Asana"])) == ["Trello"]
    assert ai.choose_gaps(GapAssessment(missing_required=[], missing_nice_to_have=[])) == []


def test_cover_letter_enforces_gaps_length_and_greeting(monkeypatch):
    sent = []
    replies = iter([_letter(200), _letter(300)])  # too short first, fine after one rewrite

    def fake_complete_messages(system, messages, what):
        sent.append([dict(m) for m in messages])
        return next(replies)

    monkeypatch.setattr(ai, "assess_gaps", lambda profile, reqs: GapAssessment(
        missing_required=["KYC", "AML", "Salesforce"], missing_nice_to_have=["Trello"]))
    monkeypatch.setattr(ai, "_complete_messages", fake_complete_messages)

    letter = ai.generate_cover_letter(_context(), "FinEdge", None)

    first_prompt = sent[0][0]["content"]
    assert "<gaps_to_mention>\nKYC\nAML\n</gaps_to_mention>" in first_prompt  # only the top two
    assert "Salesforce" not in first_prompt.split("<gaps_to_mention>")[1].split("</gaps_to_mention>")[0]
    assert "<greeting>Dear FinEdge Team,</greeting>" in first_prompt
    assert "(none given" in first_prompt  # no motivation notes

    assert len(sent) == 2  # exactly one rewrite
    assert "200 words" in sent[1][-1]["content"]
    assert letter.splitlines()[0] == "Dear FinEdge Team,"  # model's "Dear Sir," replaced
    assert ai.letter_body_word_count(letter) == 300


def test_cover_letter_without_requirements_skips_gap_check_and_rewrite(monkeypatch):
    calls = []
    monkeypatch.setattr(ai, "assess_gaps", lambda *a: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(ai, "_complete_messages", lambda s, m, w: calls.append(m) or _letter(300, "Dear Team,"))

    letter = ai.generate_cover_letter(_context(requirements=None), None, "I use their app.")
    assert len(calls) == 1
    assert "<gaps_to_mention>\n(none)\n</gaps_to_mention>" in calls[0][0]["content"]
    assert "I use their app." in calls[0][0]["content"]
    assert letter.splitlines()[0] == "Dear Hiring Manager,"  # no company known


# ---------- History: status, documents, final versions (v5) ----------
def _app_with_documents(client, h, cvs=2, letters=1):
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    docs = [client.post(f"/applications/{app_id}/generate-cv", headers=h).json() for _ in range(cvs)]
    docs += [client.post(f"/applications/{app_id}/generate-cover-letter", headers=h).json() for _ in range(letters)]
    return app_id, docs


def test_application_list_shows_status_counts_and_last_generated(client, fake_ai, fake_letter):
    h = user_with_profile(client)
    app_id, docs = _app_with_documents(client, h, cvs=2, letters=1)
    empty_id = client.post("/applications", json={"job_title": "Other"}, headers=h).json()["id"]

    apps = {a["id"]: a for a in client.get("/applications", headers=h).json()}
    assert apps[app_id]["status"] == "draft"
    assert apps[app_id]["document_counts"] == {"cv": 2, "cover_letter": 1}
    assert apps[app_id]["last_generated_at"] == max(d["created_at"] for d in docs)
    assert apps[empty_id]["document_counts"] == {"cv": 0, "cover_letter": 0}
    assert apps[empty_id]["last_generated_at"] is None


def test_status_moves_through_the_pipeline(client):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    for status in ("applied", "interview", "offer", "rejected", "applied"):  # moving back is allowed
        r = client.patch(f"/applications/{app_id}", json={"status": status}, headers=h)
        assert r.status_code == 200 and r.json()["status"] == status
    assert client.patch(f"/applications/{app_id}", json={"status": "hired"}, headers=h).status_code == 422
    assert client.patch(f"/applications/{app_id}", json={"status": None}, headers=h).status_code == 422
    assert client.get(f"/applications/{app_id}", headers=h).json()["status"] == "applied"


def test_get_and_delete_a_document(client, fake_ai, fake_letter):
    h = user_with_profile(client)
    app_id, docs = _app_with_documents(client, h, cvs=2, letters=0)
    first = docs[0]["id"]

    r = client.get(f"/documents/{first}", headers=h)
    assert r.status_code == 200 and r.json()["content"] == "# Test CV" and r.json()["is_final"] is False

    assert client.delete(f"/documents/{first}", headers=h).status_code == 204
    assert client.get(f"/documents/{first}", headers=h).status_code == 404
    assert versions(client, app_id, h) == [("cv", 2)]  # the other version is untouched


def test_only_one_final_document_per_type(client, fake_ai, fake_letter):
    h = user_with_profile(client)
    app_id, docs = _app_with_documents(client, h, cvs=2, letters=1)
    cv1, cv2, letter = (d["id"] for d in docs)

    assert client.post(f"/documents/{cv1}/mark-final", headers=h).json()["is_final"] is True
    assert client.post(f"/documents/{letter}/mark-final", headers=h).json()["is_final"] is True
    assert client.post(f"/documents/{cv2}/mark-final", headers=h).json()["is_final"] is True

    final = {(d["doc_type"], d["version"]): d["is_final"] for d in client.get(f"/applications/{app_id}/documents", headers=h).json()}
    assert final == {("cv", 2): True, ("cv", 1): False, ("cover_letter", 1): True}  # cv2 replaced cv1; letter unaffected

    # Marking the same document again changes nothing.
    assert client.post(f"/documents/{cv2}/mark-final", headers=h).status_code == 200


def test_database_rejects_two_final_documents_of_the_same_type(client, fake_ai, fake_letter):
    from sqlalchemy.exc import IntegrityError

    h = user_with_profile(client)
    _, docs = _app_with_documents(client, h, cvs=2, letters=0)
    with SessionLocal() as db:
        for d in docs:
            db.get(GeneratedDocument, d["id"]).is_final = True
        with pytest.raises(IntegrityError):
            db.commit()


def test_cannot_touch_another_users_documents(client, fake_ai, fake_letter):
    a = user_with_profile(client, "a@example.com", "User A")
    _, docs = _app_with_documents(client, a, cvs=1, letters=0)
    doc_id = docs[0]["id"]

    b = user_with_profile(client, "b@example.com", "User B")
    assert client.get(f"/documents/{doc_id}", headers=b).status_code == 404
    assert client.post(f"/documents/{doc_id}/mark-final", headers=b).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=b).status_code == 404
    assert client.get(f"/documents/{doc_id}", headers=a).json()["is_final"] is False  # untouched


# ---------- Fact-check (v7) ----------
from app.services import factcheck  # noqa: E402
from app.schemas import ClaimList, ClaimVerdict, ExtractedClaim  # noqa: E402

real_check_document = factcheck.check_document  # kept before the autouse fixture below replaces it


def clean_report(**summary):
    return {"status": "ok", "checked_at": "2026-01-01T00:00:00+00:00", "flags": [], "claims": [], "error": None,
            "summary": {"unsupported": 0, "stretch": 0, "code_flags": 0, "claims_checked": 0} | summary}


@pytest.fixture(autouse=True)
def fake_fact_check(monkeypatch):
    """Every test gets a clean fact-check report unless it sets its own, so no test calls the real judge."""
    monkeypatch.setattr(factcheck, "check_document", lambda *a, **k: clean_report())


FC_PROFILE = {
    "full_name": "Ada Obi", "headline": None, "phone": None, "location": "Lagos", "linkedin_url": None,
    "portfolio_url": "https://ada04.github.io/", "summary": None,
    "experiences": [
        {"id": 1, "job_title": "Graphic Designer", "company": "Acme Ltd.", "location": "Remote",
         "start_date": "2021-01-01", "end_date": "2022-06-01",
         "description": "- Designed 30 social media graphics monthly\n- Answered customer emails"},
        {"id": 2, "job_title": "Support Intern", "company": "BoxCo", "location": None,
         "start_date": "2023-01-01", "end_date": None, "description": "- Resolved 20 tickets weekly in Zendesk"},
    ],
    "education": [], "projects": [], "certifications": [],
    "skills": [{"id": 1, "name": "Canva", "category": None, "level": None}],
}


def code_flags(text, job_title="Support Specialist", years=3.0):
    sources = factcheck.build_sources(FC_PROFILE, job_title, None, None, years)
    return [f["type"] for f in factcheck.code_checks(text, FC_PROFILE, sources, job_title, years)]


@pytest.mark.parametrize("text, expected", [
    ("Designed 45 social media graphics monthly.", ["number"]),
    ("Portfolio: https://ada04.github.io/", []),  # numbers inside URLs are ignored
    ("At Acme Ltd., I designed 30 social media graphics monthly.", []),
    ("Managed customer accounts in Salesforce.", ["tool"]),
    ("I have not used Salesforce yet, but I am keen to learn it.", []),  # gap sentences aren't claims
    ("At Acme Ltd., I designed 30 social media graphics monthly using Canva.", ["tool_attachment"]),
    ("Proficient in Canva.", []),  # a skill, not attached to a piece of work
    ("At BoxCo, I resolved 20 tickets weekly in Zendesk.", []),  # the profile attaches Zendesk to BoxCo
    ("Support Specialist with 3 years of experience.", ["title"]),
    ("I am applying for the Support Specialist role.", []),
    ("Over five years of experience in design.", ["years"]),
    ("Graphic Designer, Acme Ltd., March 2020 – June 2022", ["date"]),
    ("From January 2021 to June 2022 at Acme Ltd., I answered customer emails.", []),
    ("Since January 2023, I have worked at BoxCo.", []),
    ("## Experience\n**Art Director** | Acme Ltd. | Jan 2021 – Jun 2022", ["title"]),
    ("## Experience\n**Designer** | Globex | Remote", ["employer"]),
])
def test_code_checks(text, expected):
    assert code_flags(text) == expected


def test_claim_check_uses_sources_and_skips_easy_cases(monkeypatch):
    judged = []
    monkeypatch.setattr(ai, "extract_claims", lambda doc, src: ClaimList(claims=[
        ExtractedClaim(claim="Designed 30 social media graphics monthly", kind="candidate_fact", source_ids=["E1"]),
        ExtractedClaim(claim="Worked with cross-functional teams", kind="candidate_fact", source_ids=["E2"]),
        ExtractedClaim(claim="Led a team of five designers", kind="candidate_fact", source_ids=[]),
        ExtractedClaim(claim="The role involves brand design", kind="job_fact", source_ids=["A"]),
        ExtractedClaim(claim="Dear Hiring Manager", kind="other", source_ids=[]),
    ]))

    def fake_judge(claim, kind, source_text):
        judged.append((claim, source_text))
        return ClaimVerdict(verdict="stretch", reason="Adds 'cross-functional'.")

    monkeypatch.setattr(ai, "judge_claim", fake_judge)
    report = real_check_document("letter text", FC_PROFILE, job_title="Designer", job_description="Brand design.",
                                 motivation_notes=None, years=3.0)

    verdicts = {c["claim"]: c["verdict"] for c in report["claims"]}
    assert verdicts == {
        "Designed 30 social media graphics monthly": "supported",  # word for word in its source: no judge call
        "Worked with cross-functional teams": "stretch",
        "Led a team of five designers": "unsupported",  # no source: unsupported without a judge call
    }
    assert [c for c, _ in judged] == ["Worked with cross-functional teams"]  # job facts and greetings aren't judged
    assert "Answered customer emails" in judged[0][1]  # the judge sees the cited source text
    assert report["summary"] == {"unsupported": 1, "stretch": 1, "code_flags": 0, "claims_checked": 3}
    assert {f["severity"] for f in report["flags"]} == {"unsupported", "stretch"}


def test_claim_check_failure_keeps_code_results(monkeypatch):
    def broken(*a):
        raise ai.AIError("The AI service is busy. Try again shortly.")

    monkeypatch.setattr(ai, "extract_claims", broken)
    report = real_check_document("Managed accounts in Salesforce.", FC_PROFILE, job_title="Designer",
                                 job_description=None, motivation_notes=None, years=3.0)
    assert report["status"] == "partial" and "busy" in report["error"]
    assert [f["type"] for f in report["flags"]] == ["tool"]


def _unsupported_report(text="Led a team of five designers"):
    r = clean_report(unsupported=1)
    r["flags"] = [{"type": "claim", "severity": "unsupported", "text": text, "source": None, "reason": "No source."}]
    return r


def test_generated_document_comes_with_its_check_report(client, fake_ai):
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]
    doc = client.post(f"/applications/{app_id}/generate-cv", headers=h).json()
    assert doc["check_report"]["summary"]["unsupported"] == 0
    assert doc["check_report"]["regenerated"] is False
    assert len(fake_ai) == 1 and fake_ai[0]["feedback"] is None
    assert client.get(f"/documents/{doc['id']}", headers=h).json()["check_report"] == doc["check_report"]


def test_unsupported_claims_trigger_one_regeneration_with_feedback(client, fake_ai, monkeypatch):
    reports = iter([_unsupported_report(), clean_report()])
    monkeypatch.setattr(factcheck, "check_document", lambda *a, **k: next(reports))
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]

    doc = client.post(f"/applications/{app_id}/generate-cv", headers=h).json()
    assert len(fake_ai) == 2
    assert "Led a team of five designers" in fake_ai[1]["feedback"]  # the flags are fed back
    assert doc["check_report"]["regenerated"] is True
    assert doc["check_report"]["summary"]["unsupported"] == 0
    assert doc["check_report"]["first_attempt"]["summary"]["unsupported"] == 1


def test_remaining_problems_are_returned_as_warnings_not_hidden(client, fake_letter, monkeypatch):
    monkeypatch.setattr(factcheck, "check_document", lambda *a, **k: _unsupported_report())
    h = user_with_profile(client)
    app_id = client.post("/applications", json=APPLICATION, headers=h).json()["id"]

    r = client.post(f"/applications/{app_id}/generate-cover-letter", headers=h)
    assert r.status_code == 201  # still returned...
    assert len(fake_letter) == 2  # ...after exactly one retry
    assert r.json()["check_report"]["summary"]["unsupported"] == 1  # with its warnings visible
    assert r.json()["check_report"]["flags"][0]["text"] == "Led a team of five designers"


# ---------- Fact-check evaluation set: code layer (free, no AI calls) ----------
# The evaluation files contain real profile data, so they're kept out of git (see .gitignore). Where they
# don't exist (e.g. a fresh clone), these tests are skipped.
EVAL_DIR = pathlib.Path(__file__).resolve().parent / "eval_cases"
_EVAL_FILES = [EVAL_DIR / name for name in ("cases.json", "profile.json", "jobs.json")]
EVAL_CASES = (json.loads((EVAL_DIR / "cases.json").read_text(encoding="utf-8"))["cases"]
              if all(f.exists() for f in _EVAL_FILES) else [])


def _eval_code_flags(case):
    from types import SimpleNamespace as NS
    profile = json.loads((EVAL_DIR / "profile.json").read_text(encoding="utf-8"))
    job = json.loads((EVAL_DIR / "jobs.json").read_text(encoding="utf-8"))[case["job"]]
    years = years_of_experience([NS(start_date=date.fromisoformat(e["start_date"]),
                                    end_date=date.fromisoformat(e["end_date"]) if e.get("end_date") else None)
                                 for e in profile["experiences"]])
    sources = factcheck.build_sources(profile, job["job_title"], job["job_description"], case.get("motivation_notes"), years)
    return factcheck.code_checks(case["text"], profile, sources, job["job_title"], years)


@pytest.mark.parametrize("case", [c for c in EVAL_CASES if c.get("code_should_flag")], ids=lambda c: c["id"])
def test_code_checks_catch_known_failures(case):
    assert _eval_code_flags(case), f"code checks missed: {case['problem']}"


@pytest.mark.parametrize("case", [c for c in EVAL_CASES if c["kind"] == "honest"], ids=lambda c: c["id"])
def test_code_checks_leave_honest_lines_alone(case):
    assert _eval_code_flags(case) == []
