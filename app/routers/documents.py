"""Single generated documents (CVs and cover letters): view, delete, and choose the final version."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import GeneratedDocument, User
from app.schemas import GeneratedDocumentOut
from app.security import get_current_user

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
