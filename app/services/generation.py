"""The shared pipeline behind every generated document (CV, cover letter, ...).

load profile -> requirements -> retrieve chunks -> years of experience, so every document type sees
exactly the same facts about the candidate.
"""
import json
from dataclasses import dataclass
from datetime import date

from sqlalchemy.orm import Session

from app.models import JobApplication, Profile
from app.schemas import FullProfileOut, JobRequirements
from app.services import retrieval


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


def build_generation_context(db: Session, profile: Profile, application: JobApplication) -> GenerationContext:
    requirements = JobRequirements.model_validate(application.requirements) if application.requirements else None

    if requirements is not None:
        # Only the basic details, the job list and the bullets most relevant to this job.
        matches = retrieval.find_matches(db, profile, requirements)
        data = retrieval.selected_profile(profile, matches)
    else:
        matches = []
        data = FullProfileOut.model_validate(profile).model_dump(mode="json")

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
