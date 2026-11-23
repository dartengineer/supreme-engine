"""Match score: how well the profile fits a job, calculated in code.

Each required skill and nice-to-have is labelled:
- gap: the saved gap check found no evidence for it in the whole profile
- strong: not a gap, and its closest profile chunk is very similar (>= STRONG_SIMILARITY)
- partial: not a gap, but the closest evidence is phrased differently or only related

Why not decide gaps from embedding distance alone: on 38 hand-labelled requirements from real ads, the best
distance threshold got 87% right and called real experience "missing" (e.g. email and calendar management,
outbound calls), while the LLM gap check got 95%. Distance is still used, for strong vs partial and evidence.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Profile, ProfileChunk
from app.schemas import GapAssessment, JobRequirements, MatchReport, RequirementMatch
from app.services import embeddings

# Tuned on the real ads: clearly met requirements with matching wording score 0.57-1.0; met requirements
# described in other words (e.g. "outbound calls" vs "made 50-70 outbound calls") score around 0.47-0.48.
STRONG_SIMILARITY = 0.55
WEIGHTS = {"required": 2, "nice_to_have": 1}
CREDIT = {"strong": 1.0, "partial": 0.5, "gap": 0.0}


def closest_chunks(db: Session, profile: Profile, queries: list[str]) -> list[tuple[ProfileChunk | None, float | None]]:
    """For each query, the profile's closest chunk and its cosine similarity."""
    if not queries:
        return []
    if db.scalar(select(ProfileChunk.id).where(ProfileChunk.profile_id == profile.id).limit(1)) is None:
        embeddings.reindex_profile(profile, db)
    results = []
    for vector in embeddings.embed(queries):
        distance = ProfileChunk.embedding.cosine_distance(vector).label("distance")
        row = db.execute(select(ProfileChunk, distance).where(ProfileChunk.profile_id == profile.id)
                         .order_by(distance).limit(1)).first()
        results.append((row[0], round(1 - float(row[1]), 3)) if row else (None, None))
    return results


def label_for(is_gap: bool, similarity: float | None) -> str:
    if is_gap:
        return "gap"
    return "strong" if similarity is not None and similarity >= STRONG_SIMILARITY else "partial"


def compute_score(items: list[RequirementMatch]) -> int:
    """Matched weight / total weight, as a whole number from 0 to 100."""
    total = sum(WEIGHTS[i.kind] for i in items)
    if total == 0:
        return 0
    matched = sum(WEIGHTS[i.kind] * CREDIT[i.label] for i in items)
    return round(100 * matched / total)


def build_match(db: Session, profile: Profile, requirements: JobRequirements, assessment: GapAssessment) -> MatchReport:
    missing = {
        "required": {m.index for m in assessment.missing_required},
        "nice_to_have": {m.index for m in assessment.missing_nice_to_have},
    }
    rows = [(r, "required", i) for i, r in enumerate(requirements.required_skills, 1)] + \
           [(r, "nice_to_have", i) for i, r in enumerate(requirements.nice_to_have_skills, 1)]
    closest = closest_chunks(db, profile, [r for r, _, _ in rows])

    items = []
    for (requirement, kind, index), (chunk, similarity) in zip(rows, closest):
        is_gap = index in missing[kind]
        items.append(RequirementMatch(
            requirement=requirement, kind=kind, label=label_for(is_gap, similarity), similarity=similarity,
            evidence=None if is_gap or chunk is None else chunk.text,
            evidence_type=None if is_gap or chunk is None else chunk.source_type,
        ))

    return MatchReport(
        score=compute_score(items),
        strong=sum(i.label == "strong" for i in items),
        partial=sum(i.label == "partial" for i in items),
        gaps=sum(i.label == "gap" for i in items),
        matched=[i for i in items if i.label != "gap"],
        missing=[i for i in items if i.label == "gap"],
    )
