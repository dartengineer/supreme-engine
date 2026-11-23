"""The shared pipeline behind every generated document (CV, cover letter, ...).

load profile -> requirements -> retrieve chunks -> years of experience, so every document type sees
exactly the same facts about the candidate.
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy.orm import Session

from app.models import JobApplication, Profile
from app.schemas import FullProfileOut, GapAssessment, JobRequirements
from app.services import ai, retrieval


@dataclass
class GenerationContext:
    job_title: str
    company: str | None
    job_description: str | None
    motivation_notes: str | None
    requirements: JobRequirements | None  # None until the application has been analysed
    matches: list[retrieval.Match]  # empty when there are no requirements
    years_of_experience: float
    profile_json: str  # what the AI sees about the candidate when writing
    full_profile_json: str  # the whole profile, for checks that must see everything (e.g. which requirements are missing)
    gaps: list[str] = field(default_factory=list)  # missing requirements a cover letter may mention (see letter_gaps)


def years_of_experience(experiences, today: date | None = None) -> float:
    """Total time covered by the job dates, in years (one decimal). Overlapping months count once.

    End months are inclusive (Jan-Mar = 3 months), and a job with no end date runs to this month.
    """
    today = today or date.today()
    months: set[int] = set()
    for e in experiences:
        end = e.end_date or today
        first = e.start_date.year * 12 + e.start_date.month - 1
        last = end.year * 12 + end.month - 1
        months.update(range(first, last + 1))
    return round(len(months) / 12, 1)


REF_PREFIXES = {"experiences": "job", "projects": "proj", "education": "edu", "certifications": "cert"}


def add_refs(data: dict) -> dict:
    """Gives every job, project, education entry and certification a "ref" (e.g. "job3") the CV can point to."""
    for key, prefix in REF_PREFIXES.items():
        for item in data.get(key) or []:
            item["ref"] = f"{prefix}{item['id']}"
    return data


def build_generation_context(db: Session, profile: Profile, application: JobApplication) -> GenerationContext:
    requirements = JobRequirements.model_validate(application.requirements) if application.requirements else None

    if requirements is not None:
        # Only the basic details, the job list and the bullets most relevant to this job.
        matches = retrieval.find_matches(db, profile, requirements)
        data = retrieval.selected_profile(profile, matches)
    else:
        matches = []
        data = FullProfileOut.model_validate(profile).model_dump(mode="json")
        add_refs(data)

    years = years_of_experience(profile.experiences)
    total_experience = {
        "years": years,
        "how_calculated": "months covered by the job dates in this profile, overlapping jobs counted once",
    }
    data["total_experience"] = total_experience
    full = FullProfileOut.model_validate(profile).model_dump(mode="json")
    full["total_experience"] = total_experience

    return GenerationContext(
        job_title=application.job_title,
        company=application.company,
        job_description=application.job_description,
        motivation_notes=(application.motivation_notes or "").strip() or None,
        requirements=requirements,
        matches=matches,
        years_of_experience=years,
        profile_json=json.dumps(data, ensure_ascii=False),
        full_profile_json=json.dumps(full, ensure_ascii=False),
    )


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def letter_gaps(db: Session, application: JobApplication, context: GenerationContext) -> list[str]:
    """The gaps a cover letter may mention (at most two, in ad order), from the saved gap check."""
    if context.requirements is None:
        return []
    return ai.choose_gaps(gap_assessment(db, application, context), context.requirements)


def gap_assessment(db: Session, application: JobApplication, context: GenerationContext) -> GapAssessment:
    """Which requirements the profile shows no evidence for, from the saved gap check when it's still valid.

    The gap check (one AI call) runs once per application and is saved, so every letter and match score uses
    the same result. It reruns only when the profile or the requirements have changed since it was saved.
    """
    profile_data = json.loads(context.full_profile_json)
    profile_data.pop("total_experience", None)  # changes every month on its own; not a profile edit
    for value in profile_data.values():  # list order from the database isn't fixed; sort so the same profile
        if isinstance(value, list):      # always gives the same fingerprint
            value.sort(key=lambda item: item.get("id", 0) if isinstance(item, dict) else 0)
    key = {"profile": _fingerprint(json.dumps(profile_data, sort_keys=True)),
           "requirements": _fingerprint(context.requirements.model_dump_json())}

    saved = application.gap_check or {}
    if saved.get("key") == key:
        assessment = GapAssessment.model_validate(saved["assessment"])
    else:
        assessment = ai.assess_gaps(context.full_profile_json, context.requirements)
        application.gap_check = {"key": key, "assessment": assessment.model_dump()}
        db.commit()
    return assessment
