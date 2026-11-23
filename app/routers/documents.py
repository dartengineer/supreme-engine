"""Single generated documents (CVs and cover letters): view, download, delete, and choose the final version."""
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import GeneratedDocument, User
from app.schemas import CVDocument, GeneratedDocumentOut
from app.security import get_current_user
from app.services import cv_document, render

MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

router = APIRouter(prefix="/documents", tags=["documents"])


def _get_own_document(doc_id: int, user: User, db: Session) -> GeneratedDocument:
    document = db.get(GeneratedDocument, doc_id)
    # Same 404 whether it doesn't exist or belongs to someone else: never reveal other users' data.
    if document is None or document.job_application.user_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("/{doc_id}", response_model=GeneratedDocumentOut)
def get_document(doc_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return _get_own_document(doc_id, user, db)


def _filename(document: GeneratedDocument, name: str, ext: str) -> str:
    app = document.job_application
    kind = "CV" if document.doc_type == "cv" else "Cover_Letter"
    parts = [name, kind, app.company or app.job_title, f"v{document.version}"]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(parts)).strip("_") + f".{ext}"


@router.get("/{doc_id}/download", responses={200: {"content": {t: {} for t in MEDIA_TYPES.values()}}})
def download_document(doc_id: int, format: Literal["pdf", "docx"] = "pdf",
                      template: Literal["classic", "modern", "compact"] = "classic",
                      user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Downloads a CV or cover letter as PDF or Word. CVs come in three templates (classic, modern, compact);
    cover letters have one matching template, so `template` is ignored for them."""
    document = _get_own_document(doc_id, user, db)

    if document.doc_type == "cv":
        if not document.data:
            raise HTTPException(status_code=409, detail="This CV was generated before downloads existed. "
                                                        "Generate a new version to download it.")
        cv = CVDocument.model_validate(document.data)
        name = cv.name
        make_pdf = lambda: render.html_to_pdf(render.cv_html(cv, template))  # noqa: E731
        make_docx = lambda: render.cv_docx(cv, template)  # noqa: E731
    else:
        profile = user.profile
        name = profile.full_name if profile else "Cover letter"
        contact = cv_document.contact_parts(profile.email, profile.phone, profile.location, profile.linkedin_url,
                                            profile.portfolio_url) if profile else ""
        make_pdf = lambda: render.html_to_pdf(render.letter_html(document.content, name, contact))  # noqa: E731
        make_docx = lambda: render.letter_docx(document.content, name, contact)  # noqa: E731

    try:
        content = make_pdf() if format == "pdf" else make_docx()
    except render.RendererUnavailable as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    filename = _filename(document, name, format)
    return Response(content=content, media_type=MEDIA_TYPES[format],
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Deletes one document. Other versions keep their numbers. New documents are numbered after the highest
    remaining version, so deleting the latest version frees its number for the next one."""
    document = _get_own_document(doc_id, user, db)
    db.delete(document)
    db.commit()


@router.post("/{doc_id}/mark-final", response_model=GeneratedDocumentOut)
def mark_final(doc_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Marks this document as the one you'll send. Any other final document of the same type for the same
    application stops being final, so there's only ever one final CV and one final cover letter."""
    document = _get_own_document(doc_id, user, db)
    db.execute(
        update(GeneratedDocument)
        .where(
            GeneratedDocument.job_application_id == document.job_application_id,
            GeneratedDocument.doc_type == document.doc_type,
            GeneratedDocument.id != document.id,
            GeneratedDocument.is_final.is_(True),
        )
        .values(is_final=False)
    )
    document.is_final = True
    db.commit()
    db.refresh(document)
    return document
