"""Profile endpoints and its sections. Every route works only on the logged-in user's own data."""
import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Certification, Education, Experience, Profile, Project, Skill, User
from app.schemas import (
    CertificationBase,
    CertificationOut,
    EducationBase,
    EducationOut,
    ExperienceBase,
    ExperienceOut,
    FullProfileOut,
    ProfileBase,
    ProfileOut,
    ProjectBase,
    ProjectOut,
    SkillBase,
    SkillOut,
)
from app.security import get_current_user
from app.services import embeddings

router = APIRouter(prefix="/profile", tags=["profile"])
logger = logging.getLogger(__name__)


def get_own_profile(user: User = Depends(get_current_user)) -> Profile:
    if user.profile is None:
        raise HTTPException(status_code=404, detail="Create your profile first (PUT /profile)")
    return user.profile


def _get_owned(model, item_id: int, profile: Profile, db: Session, label: str):
    """Loads one row of any section (Experience, Education, ...) by id, but only if it is this user's.

    Same 404 whether it doesn't exist or belongs to someone else: never reveal other users' data.
    """
    item = db.get(model, item_id)
    if item is None or item.profile_id != profile.id:
        raise HTTPException(status_code=404, detail=f"{label} not found")
    return item


def _reindex(profile: Profile, db: Session) -> None:
    """Rebuilds the profile's search chunks after a section changes. A failure is logged, never shown to the user:
    the change itself is already saved, and chunks are rebuilt again on the next change or when matching."""
    try:
        embeddings.reindex_profile(profile, db)
    except Exception:
        db.rollback()
        logger.exception("Couldn't rebuild chunks for profile %s", profile.id)


def _newest_first(items, date_field: str):
    """Sorts by a date field, newest first. Rows with no date go last."""
    return sorted(items, key=lambda i: getattr(i, date_field) or date.min, reverse=True)


# ---------- Profile ----------
@router.put("", response_model=ProfileOut)
def create_or_update_profile(data: ProfileBase, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    profile = user.profile
    if profile is None:
        profile = Profile(user_id=user.id, **data.model_dump())
        db.add(profile)
    else:
        # Only the fields you send change, so adding e.g. an email doesn't wipe the rest of the profile.
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(profile, field, value)
    db.commit()
    db.refresh(profile)
    return profile


@router.get("", response_model=FullProfileOut)
def get_full_profile(profile: Profile = Depends(get_own_profile)):
    """Returns the profile with every section nested inside it."""
    return profile


# ---------- Experiences ----------
@router.post("/experiences", response_model=ExperienceOut, status_code=status.HTTP_201_CREATED)
def add_experience(data: ExperienceBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    exp = Experience(profile_id=profile.id, **data.model_dump())
    db.add(exp)
    db.commit()
    _reindex(profile, db)
    db.refresh(exp)
    return exp


@router.get("/experiences", response_model=list[ExperienceOut])
def list_experiences(profile: Profile = Depends(get_own_profile)):
    return sorted(profile.experiences, key=lambda e: e.start_date, reverse=True)


@router.put("/experiences/{exp_id}", response_model=ExperienceOut)
def update_experience(exp_id: int, data: ExperienceBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    exp = _get_owned(Experience, exp_id, profile, db, "Experience")
    for field, value in data.model_dump().items():
        setattr(exp, field, value)
    db.commit()
    _reindex(profile, db)
    db.refresh(exp)
    return exp


@router.delete("/experiences/{exp_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experience(exp_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    exp = _get_owned(Experience, exp_id, profile, db, "Experience")
    db.delete(exp)
    db.commit()
    _reindex(profile, db)


# ---------- Education ----------
@router.post("/education", response_model=EducationOut, status_code=status.HTTP_201_CREATED)
def add_education(data: EducationBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    edu = Education(profile_id=profile.id, **data.model_dump())
    db.add(edu)
    db.commit()
    _reindex(profile, db)
    db.refresh(edu)
    return edu


@router.get("/education", response_model=list[EducationOut])
def list_education(profile: Profile = Depends(get_own_profile)):
    return _newest_first(profile.education, "start_date")


@router.put("/education/{edu_id}", response_model=EducationOut)
def update_education(edu_id: int, data: EducationBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    edu = _get_owned(Education, edu_id, profile, db, "Education")
    for field, value in data.model_dump().items():
        setattr(edu, field, value)
    db.commit()
    _reindex(profile, db)
    db.refresh(edu)
    return edu


@router.delete("/education/{edu_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_education(edu_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    edu = _get_owned(Education, edu_id, profile, db, "Education")
    db.delete(edu)
    db.commit()
    _reindex(profile, db)


# ---------- Skills ----------
@router.post("/skills", response_model=SkillOut, status_code=status.HTTP_201_CREATED)
def add_skill(data: SkillBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    skill = Skill(profile_id=profile.id, **data.model_dump())
    db.add(skill)
    db.commit()
    _reindex(profile, db)
    db.refresh(skill)
    return skill


@router.get("/skills", response_model=list[SkillOut])
def list_skills(profile: Profile = Depends(get_own_profile)):
    # Grouped by category, then A-Z. Skills with no category go first.
    return sorted(profile.skills, key=lambda s: ((s.category or "").lower(), s.name.lower()))


@router.put("/skills/{skill_id}", response_model=SkillOut)
def update_skill(skill_id: int, data: SkillBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    skill = _get_owned(Skill, skill_id, profile, db, "Skill")
    for field, value in data.model_dump().items():
        setattr(skill, field, value)
    db.commit()
    _reindex(profile, db)
    db.refresh(skill)
    return skill


@router.delete("/skills/{skill_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_skill(skill_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    skill = _get_owned(Skill, skill_id, profile, db, "Skill")
    db.delete(skill)
    db.commit()
    _reindex(profile, db)


# ---------- Projects ----------
@router.post("/projects", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
def add_project(data: ProjectBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    project = Project(profile_id=profile.id, **data.model_dump())
    db.add(project)
    db.commit()
    _reindex(profile, db)
    db.refresh(project)
    return project


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(profile: Profile = Depends(get_own_profile)):
    return _newest_first(profile.projects, "start_date")


@router.put("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: int, data: ProjectBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    project = _get_owned(Project, project_id, profile, db, "Project")
    for field, value in data.model_dump().items():
        setattr(project, field, value)
    db.commit()
    _reindex(profile, db)
    db.refresh(project)
    return project


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    project = _get_owned(Project, project_id, profile, db, "Project")
    db.delete(project)
    db.commit()
    _reindex(profile, db)


# ---------- Certifications ----------
@router.post("/certifications", response_model=CertificationOut, status_code=status.HTTP_201_CREATED)
def add_certification(data: CertificationBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    cert = Certification(profile_id=profile.id, **data.model_dump())
    db.add(cert)
    db.commit()
    _reindex(profile, db)
    db.refresh(cert)
    return cert


@router.get("/certifications", response_model=list[CertificationOut])
def list_certifications(profile: Profile = Depends(get_own_profile)):
    return _newest_first(profile.certifications, "issue_date")


@router.put("/certifications/{cert_id}", response_model=CertificationOut)
def update_certification(cert_id: int, data: CertificationBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    cert = _get_owned(Certification, cert_id, profile, db, "Certification")
    for field, value in data.model_dump().items():
        setattr(cert, field, value)
    db.commit()
    _reindex(profile, db)
    db.refresh(cert)
    return cert


@router.delete("/certifications/{cert_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_certification(cert_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    cert = _get_owned(Certification, cert_id, profile, db, "Certification")
    db.delete(cert)
    db.commit()
    _reindex(profile, db)
