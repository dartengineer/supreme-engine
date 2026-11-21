"""Profile and experience endpoints. Every route works only on the logged-in user's own data."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Experience, Profile, User
from app.schemas import ExperienceBase, ExperienceOut, FullProfileOut, ProfileBase, ProfileOut
from app.security import get_current_user

router = APIRouter(prefix="/profile", tags=["profile"])


def get_own_profile(user: User = Depends(get_current_user)) -> Profile:
    if user.profile is None:
        raise HTTPException(status_code=404, detail="Create your profile first (PUT /profile)")
    return user.profile


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


def _get_own_experience(exp_id: int, profile: Profile, db: Session) -> Experience:
    exp = db.get(Experience, exp_id)
    # Same 404 whether it doesn't exist or belongs to someone else: never reveal other users' data.
    if exp is None or exp.profile_id != profile.id:
        raise HTTPException(status_code=404, detail="Experience not found")
    return exp


@router.put("/experiences/{exp_id}", response_model=ExperienceOut)
def update_experience(exp_id: int, data: ExperienceBase, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    exp = _get_own_experience(exp_id, profile, db)
    for field, value in data.model_dump().items():
        setattr(exp, field, value)
    db.commit()
    db.refresh(exp)
    return exp


@router.delete("/experiences/{exp_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experience(exp_id: int, profile: Profile = Depends(get_own_profile), db: Session = Depends(get_db)):
    exp = _get_own_experience(exp_id, profile, db)
    db.delete(exp)
    db.commit()
