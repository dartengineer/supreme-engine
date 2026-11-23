"""Job applications and the CVs generated for them. Users only ever see their own."""
import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import GeneratedDocument, JobApplication, User
from app.schemas import (
    FullProfileOut,
    GeneratedDocumentOut,
    JobApplicationCreate,
    JobApplicationOut,
    JobRequirements,
    MatchOut,
)
from app.security import get_current_user
from app.services import ai, retrieval  # called as ai.generate_cv(...) so tests can swap it for a fake

router = APIRouter(prefix="/applications", tags=["applications"])


def _get_own_application(app_id: int, user: User, db: Session) -> JobApplication:
    application = db.get(JobApplication, app_id)
    # Same 404 whether it doesn't exist or belongs to someone else: never reveal other users' data.
    if application is None or application.user_id != user.id:
        raise HTTPException(status_code=404, detail="Application not found")
    return application


@router.post("", response_model=JobApplicationOut, status_code=status.HTTP_201_CREATED)
def create_application(data: JobApplicationCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    application = JobApplication(user_id=user.id, **data.model_dump())
    db.add(application)
    db.commit()
    db.refresh(application)
    return application


@router.get("", response_model=list[JobApplicationOut])
def list_applications(user: User = Depends(get_current_user)):
    return sorted(user.job_applications, key=lambda a: a.created_at, reverse=True)


@router.get("/{app_id}", response_model=JobApplicationOut)
def get_application(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _get_own_application(app_id, user, db)


@router.post("/{app_id}/analyze", response_model=JobRequirements)
def analyze_application(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Reads the job description, extracts its requirements, and saves them on the application."""
    application = _get_own_application(app_id, user, db)
    if not (application.job_description or "").strip():
        raise HTTPException(status_code=400, detail="Add a job description to this application before analysing it")
    try:
        requirements = ai.extract_requirements(application.job_title, application.job_description)
    except ai.AIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

    application.requirements = requirements.model_dump()
    db.commit()
    return requirements


@router.get("/{app_id}/matches", response_model=list[MatchOut])
def list_matches(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """The profile bullets/items picked for this job, best first, with their similarity scores.

    These are exactly what POST /applications/{id}/generate-cv passes to the AI.
    """
    application = _get_own_application(app_id, user, db)
    if user.profile is None:
        raise HTTPException(status_code=404, detail="Create your profile first (PUT /profile)")
    if not application.requirements:
        raise HTTPException(status_code=400, detail="Analyse this application first (POST /applications/{id}/analyze)")
    requirements = JobRequirements.model_validate(application.requirements)
    return retrieval.find_matches(db, user.profile, requirements)


@router.post("/{app_id}/generate-cv", response_model=GeneratedDocumentOut, status_code=status.HTTP_201_CREATED)
def generate_cv(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Writes a CV tailored to this job from your profile, and saves it as the next version.

    With requirements (from POST /applications/{id}/analyze), the AI only sees your basic details, your job list
    and the bullets picked by GET /applications/{id}/matches. Without them, it sees your whole profile.
    """
    application = _get_own_application(app_id, user, db)
    if user.profile is None:
        raise HTTPException(status_code=404, detail="Create your profile first (PUT /profile)")

    requirements = JobRequirements.model_validate(application.requirements) if application.requirements else None
    if requirements is not None:
        matches = retrieval.find_matches(db, user.profile, requirements)
        profile_json = json.dumps(retrieval.selected_profile(user.profile, matches), ensure_ascii=False)
    else:
        profile_json = FullProfileOut.model_validate(user.profile).model_dump_json()
    try:
        content = ai.generate_cv(profile_json, application.job_title, application.job_description, requirements)
    except ai.AIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))

    previous = db.scalar(
        select(func.count()).select_from(GeneratedDocument).where(
            GeneratedDocument.job_application_id == application.id,
            GeneratedDocument.doc_type == "cv",
        )
    )
    document = GeneratedDocument(
        job_application_id=application.id,
        doc_type="cv",
        version=previous + 1,
        content=content,
        model_used=settings.ai_model,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


@router.get("/{app_id}/documents", response_model=list[GeneratedDocumentOut])
def list_documents(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Every CV generated for this application, newest version first."""
    application = _get_own_application(app_id, user, db)
    return sorted(application.documents, key=lambda d: (d.doc_type, -d.version))
