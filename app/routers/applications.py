"""Job applications and the CVs generated for them. Users only ever see their own."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import GeneratedDocument, JobApplication, User
from app.schemas import FullProfileOut, GeneratedDocumentOut, JobApplicationCreate, JobApplicationOut
from app.security import get_current_user
from app.services import ai  # called as ai.generate_cv(...) so tests can swap it for a fake

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


@router.post("/{app_id}/generate-cv", response_model=GeneratedDocumentOut, status_code=status.HTTP_201_CREATED)
def generate_cv(app_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Writes a CV tailored to this job from your profile, and saves it as the next version."""
    application = _get_own_application(app_id, user, db)
    if user.profile is None:
        raise HTTPException(status_code=404, detail="Create your profile first (PUT /profile)")

    profile_json = FullProfileOut.model_validate(user.profile).model_dump_json()
    try:
        content = ai.generate_cv(profile_json, application.job_title, application.job_description)
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
