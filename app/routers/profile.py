"""Profile endpoints and its sections. Every route works only on the logged-in user's own data."""
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Education, Experience, Profile, User
from app.schemas import (
    EducationBase,
    EducationOut,
    ExperienceBase,
    ExperienceOut,
    FullProfileOut,
    ProfileBase,
    ProfileOut,
)
from app.security import get_current_user

router = APIRouter(prefix="/profile", tags=["profile"])


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
        for field, value in data.model_dump().items():
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
    db.refresh(exp)
    return exp


@router.delete("/experiences/{exp_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experience(exp_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    exp = _get_owned(Experience, exp_id, profile, db, "Experience")
    db.delete(exp)
    db.commit()


# ---------- Education ----------
@router.post("/education", response_model=EducationOut, status_code=status.HTTP_201_CREATED)
def add_education(data: EducationBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    edu = Education(profile_id=profile.id, **data.model_dump())
    db.add(edu)
    db.commit()
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
    db.refresh(edu)
    return edu


@router.delete("/education/{edu_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_education(edu_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    edu = _get_owned(Education, edu_id, profile, db, "Education")
    db.delete(edu)
    db.commit()
