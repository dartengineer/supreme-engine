"""Turns profile text into vectors so the app can find the bullets most relevant to a job.

Uses the free local model all-MiniLM-L6-v2 (384 dimensions): no API calls, no cost.
The model (~90 MB) downloads automatically the first time embed() runs, then loads from disk.
"""
from dataclasses import dataclass
from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384


@lru_cache(maxsize=1)
def _model():
    # Imported and loaded on first use only, so app startup and tests that fake embed() stay fast.
    from sentence_transformers import SentenceTransformer

    try:
        # Load the copy already on disk without asking Hugging Face first (those online checks can take minutes).
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError:
        return SentenceTransformer(MODEL_NAME)  # first run: download it (~90 MB)


def warm_up() -> None:
    """Loads the model now (called in the background at server startup). Errors are logged, not raised."""
    import logging
    import time

    log = logging.getLogger("uvicorn.error")
    start = time.time()
    try:
        _model()
        log.info("Embedding model ready (%.1fs)", time.time() - start)
    except Exception:
        log.exception("Couldn't load the embedding model; it will be retried on first use")


def embed(texts: list[str]) -> list[list[float]]:
    """Returns one 384-number vector per text. Vectors are normalised, so cosine similarity = dot product."""
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


@dataclass
class Chunk:
    source_type: str  # "experience", "project", "skill" or "certification"
    source_id: int  # id of the experience/project/skill/certification row it came from
    text: str  # what's shown to the AI and on /matches
    embed_text: str  # what gets embedded: the text plus a little context (e.g. the job title)


def _split_bullets(description: str | None) -> list[str]:
    """Splits a description into bullets. Lines starting with -, * or • are bullets; plain text is one bullet."""
    if not description:
        return []
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    bullets = [line.lstrip("-*• ").strip() for line in lines]
    return [b for b in bullets if b]


def build_chunks(profile) -> list[Chunk]:
    """Breaks a profile into small pieces that can each be matched against a job on their own."""
    chunks: list[Chunk] = []

    for exp in profile.experiences:
        for bullet in _split_bullets(exp.description):
            chunks.append(Chunk("experience", exp.id, bullet, f"{exp.job_title}: {bullet}"))

    for project in profile.projects:
        parts = [project.name]
        if project.technologies:
            parts.append(f"({project.technologies})")
        text = " ".join(parts)
        if project.description:
            text += ". " + " ".join(_split_bullets(project.description))
        chunks.append(Chunk("project", project.id, text, text))

    # Skills and certifications are embedded without labels like "Skill:" or "(Professional skills)":
    # those words made every skill match any requirement containing "skills", and every certificate
    # match any requirement containing "certification".
    for skill in profile.skills:
        text = skill.name if not skill.category else f"{skill.name} ({skill.category})"
        chunks.append(Chunk("skill", skill.id, text, skill.name))

    for cert in profile.certifications:
        text = cert.name if not cert.issuer else f"{cert.name}, {cert.issuer}"
        chunks.append(Chunk("certification", cert.id, text, text))

    return chunks


def reindex_profile(profile, db) -> int:
    """Deletes this profile's chunks and rebuilds them from its current content. Returns the chunk count.

    Embeds first and only then swaps the rows, so if embedding fails the old chunks stay in place.
    """
    from sqlalchemy import delete

    from app.models import ProfileChunk  # imported here because models.py imports DIMENSIONS from this file

    db.refresh(profile)  # make sure we chunk what's actually saved now
    chunks = build_chunks(profile)
    vectors = embed([c.embed_text for c in chunks])

    db.execute(delete(ProfileChunk).where(ProfileChunk.profile_id == profile.id))
    db.add_all(
        ProfileChunk(profile_id=profile.id, source_type=c.source_type, source_id=c.source_id, text=c.text, embedding=v)
        for c, v in zip(chunks, vectors)
    )
    db.commit()
    return len(chunks)
