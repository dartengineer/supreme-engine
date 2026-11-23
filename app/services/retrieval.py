"""Picks the profile chunks most relevant to a job, using pgvector cosine distance.

For each required skill and key responsibility, we embed the phrase and take its closest chunks.
A chunk that matches several requirements keeps its best score. Weak matches are dropped, so a job
you don't fit gets fewer bullets rather than irrelevant ones.
"""
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Profile, ProfileChunk
from app.schemas import JobRequirements
from app.services import embeddings

PER_REQUIREMENT = 5  # closest chunks looked at for each requirement
# Separate limits, because short skill names score very high against requirements and would otherwise
# crowd the experience bullets (the part that actually needs choosing) out of a shared limit.
MAX_BULLETS = 12  # experience bullets and projects
MAX_ITEMS = 8  # skills and certifications
# Cosine similarity (1 = identical meaning, 0 = unrelated). Tuned on real ads: with all-MiniLM-L6-v2,
# unrelated pairs often score 0.30-0.40, so anything below 0.40 is treated as noise.
MIN_SCORE = 0.40


@dataclass
class Match:
    chunk_id: int
    source_type: str
    source_id: int
    text: str
    score: float  # best cosine similarity across all requirements
    matched_requirement: str  # the requirement that gave the best score


def _requirement_queries(requirements: JobRequirements) -> list[str]:
    seen, queries = set(), []
    for q in requirements.required_skills + requirements.key_responsibilities + requirements.nice_to_have_skills:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            queries.append(q.strip())
    return queries


def find_matches(db: Session, profile: Profile, requirements: JobRequirements) -> list[Match]:
    """Returns the best-matching chunks (up to MAX_BULLETS bullets/projects and MAX_ITEMS skills/certs),
    best match first, each with its score."""
    if db.scalar(select(func.count()).select_from(ProfileChunk).where(ProfileChunk.profile_id == profile.id)) == 0:
        embeddings.reindex_profile(profile, db)  # chunks missing (e.g. an earlier reindex failed): build them now

    queries = _requirement_queries(requirements)
    best: dict[int, Match] = {}
    for query, vector in zip(queries, embeddings.embed(queries)):
        distance = ProfileChunk.embedding.cosine_distance(vector).label("distance")
        rows = db.execute(
            select(ProfileChunk, distance)
            .where(ProfileChunk.profile_id == profile.id)
            .order_by(distance)
            .limit(PER_REQUIREMENT)
        ).all()
        for chunk, dist in rows:
            score = round(1 - float(dist), 3)
            if score < MIN_SCORE:
                continue
            if chunk.id not in best or score > best[chunk.id].score:
                best[chunk.id] = Match(chunk.id, chunk.source_type, chunk.source_id, chunk.text, score, query)

    ranked = sorted(best.values(), key=lambda m: m.score, reverse=True)
    bullets = [m for m in ranked if m.source_type in ("experience", "project")][:MAX_BULLETS]
    items = [m for m in ranked if m.source_type in ("skill", "certification")][:MAX_ITEMS]
    return sorted(bullets + items, key=lambda m: m.score, reverse=True)


def selected_profile(profile: Profile, matches: list[Match]) -> dict:
    """What the CV writer sees: basic details, the full job list, and only the matched bullets/items.

    - Jobs with matched bullets come first, most relevant job first.
    - Jobs with no matched bullets go to "other_experience" (title, company, dates only), so the
      work history stays complete without burying the relevant job under empty entries.
    - The free-text profile summary is left out: it would let skills that weren't selected slip back in.
    """
    bullets_by_exp: dict[int, list[str]] = {}
    best_score_by_exp: dict[int, float] = {}
    picked = {"project": set(), "skill": set(), "certification": set()}
    for m in matches:  # already best-first, so bullets stay in order of relevance
        if m.source_type == "experience":
            bullets_by_exp.setdefault(m.source_id, []).append(m.text)
            best_score_by_exp.setdefault(m.source_id, m.score)
        else:
            picked[m.source_type].add(m.source_id)

    def iso(d):
        return d.isoformat() if d else None

    def job(e, with_bullets: bool) -> dict:
        entry = {
            "job_title": e.job_title,
            "company": e.company,
            "location": e.location,
            "start_date": iso(e.start_date),
            "end_date": iso(e.end_date),
        }
        if with_bullets:
            entry["selected_bullets"] = bullets_by_exp[e.id]
        return entry

    relevant = sorted((e for e in profile.experiences if e.id in bullets_by_exp),
                      key=lambda e: best_score_by_exp[e.id], reverse=True)
    other = sorted((e for e in profile.experiences if e.id not in bullets_by_exp),
                   key=lambda e: e.start_date, reverse=True)

    return {
        "full_name": profile.full_name,
        "headline": profile.headline,
        "phone": profile.phone,
        "location": profile.location,
        "linkedin_url": profile.linkedin_url,
        "portfolio_url": profile.portfolio_url,
        "experiences": [job(e, with_bullets=True) for e in relevant],
        "other_experience": [job(e, with_bullets=False) for e in other],
        "education": [
            {
                "institution": ed.institution,
                "qualification": ed.qualification,
                "field_of_study": ed.field_of_study,
                "start_date": iso(ed.start_date),
                "end_date": iso(ed.end_date),
                "grade": ed.grade,
                "description": ed.description,
            }
            for ed in profile.education
        ],
        "projects": [
            {"name": p.name, "technologies": p.technologies, "url": p.url, "description": p.description}
            for p in profile.projects if p.id in picked["project"]
        ],
        # Every skill and certification is listed (they're short), split by relevance. Sending only the
        # selected ones made the AI think unselected ones were missing (e.g. "keen to learn Canva" when
        # the profile has "Figma and Canva").
        "skills": {
            "most_relevant": [s.name for s in profile.skills if s.id in picked["skill"]],
            "other": [s.name for s in profile.skills if s.id not in picked["skill"]],
        },
        "certifications": {
            "most_relevant": [{"name": c.name, "issuer": c.issuer}
                              for c in profile.certifications if c.id in picked["certification"]],
            "other": [{"name": c.name, "issuer": c.issuer}
                      for c in profile.certifications if c.id not in picked["certification"]],
        },
    }
