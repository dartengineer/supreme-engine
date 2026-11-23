"""Job applications and the documents (CVs, cover letters) generated for them. Users only ever see their own."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import GeneratedDocument, JobApplication, User
from app.schemas import (
    GeneratedDocumentOut,
    JobApplicationCreate,
    JobApplicationOut,
    JobApplicationUpdate,
    JobRequirements,
    MatchOut,
)
from app.security import get_current_user
from app.services import ai, generation, retrieval  # called as ai.generate_cv(...) so tests can swap in fakes

router = APIRouter(prefix="/applications", tags=["applications"])

DocType = Literal["cv", "cover_letter"]


def _get_own_application(app_id: int, user: User, db: Session) -> JobApplication:
    application = db.get(JobApplication, app_id)
    # Same 404 whether it doesn't exist or belongs to someone else: never reveal other users' data.
    if application is None or application.user_id != user.id:
        raise HTTPException(status_code=404, detail="Application not found")
    return application


def _require_profile(user: User):
    if user.profile is None:
        raise HTTPException(status_code=404, detail="Create your profile first (PUT /profile)")
    return user.profile


def _save_document(db: Session, application: JobApplication, doc_type: DocType, content: str) -> GeneratedDocument:
    """Saves a generated document as the next version *of its type* (CVs and cover letters count separately)."""
    latest = db.scalar(
        select(func.max(GeneratedDocument.version)).where(
            GeneratedDocument.job_application_id == application.id,
            GeneratedDocument.doc_type == doc_type,
        )
    )
    document = GeneratedDocument(
        job_application_id=application.id,
        doc_type=doc_type,
        version=(latest or 0) + 1,
        content=content,
        model_used=settings.ai_model,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


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


@router.patch("/{app_id}", response_model=JobApplicationOut)
def update_application(app_id: int, data: JobApplicationUpdate, user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    """Change some fields of an application, e.g. its status or motivation notes. Fields you leave out stay as they are.

    Status: draft -> applied -> interview -> offer / rejected (moving back is allowed, to correct mistakes).
    Changing the job title or description clears the saved requirements, since they'd describe the old ad.
    """
    application = _get_own_application(app_id, user, db)
    changes = data.model_dump(exclude_unset=True)
    for required in ("job_title", "status"):
        if required in changes and changes[required] is None:
            raise HTTPException(status_code=422, detail=f"{required} can't be empty")
    for field, value in changes.items():
        setattr(application, field, value)
    if any(f in changes for f in ("job_title", "job_description")):
        application.requirements = None
    db.commit()
    db.refresh(application)
    return application


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

    These are exactly what the CV and cover letter generators pass to the AI.
    """
    application = _get_own_application(app_id, user, db)
    profile = _require_profile(user)
    if not application.requirements:
        raise HTTPException(status_code=400, detail="Analyse this application first (POST /applications/{id}/analyze)")
    requirements = JobRequirements.model_validate(application.requirements)
    return retrieval.find_matches(db, profile, requirements)


@router.post("/{app_id}/generate-cv", response_model=GeneratedDocumentOut, status_code=status.HTTP_201_CREATED)
def generate_cv(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Writes a CV tailored to this job from your profile, and saves it as the next CV version.

    With requirements (from POST /applications/{id}/analyze), the AI only sees your basic details, your job list
    and the bullets picked by GET /applications/{id}/matches. Without them, it sees your whole profile.
    """
    application = _get_own_application(app_id, user, db)
    context = generation.build_generation_context(db, _require_profile(user), application)
    try:
        content = ai.generate_cv(context.profile_json, context.job_title, context.job_description, context.requirements)
    except ai.AIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    return _save_document(db, application, "cv", content)


@router.post("/{app_id}/generate-cover-letter", response_model=GeneratedDocumentOut,
             status_code=status.HTTP_201_CREATED)
def generate_cover_letter(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Writes a cover letter for this job from the same facts as the CV, and saves it as the next letter version.

    Add motivation notes with PATCH /applications/{id} to give the letter a personal reason for applying;
    without them, the letter keeps its motivation general.
    """
    application = _get_own_application(app_id, user, db)
    context = generation.build_generation_context(db, _require_profile(user), application)
    try:
        content = ai.generate_cover_letter(context, context.company, context.motivation_notes)
    except ai.AIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    return _save_document(db, application, "cover_letter", content)


@router.get("/{app_id}/documents", response_model=list[GeneratedDocumentOut])
def list_documents(app_id: int, doc_type: DocType | None = None, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Every document generated for this application, newest version first. Filter with ?doc_type=cv or cover_letter."""
    application = _get_own_application(app_id, user, db)
    docs = [d for d in application.documents if doc_type is None or d.doc_type == doc_type]
    return sorted(docs, key=lambda d: (d.doc_type, -d.version))
