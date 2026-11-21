"""Pydantic schemas: the shape of data going in and out of the API."""
from datetime import date

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------- Auth ----------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- Profile ----------
class ProfileBase(BaseModel):
    full_name: str
    headline: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    portfolio_url: str | None = None
    summary: str | None = None


class ProfileOut(ProfileBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Experience ----------
class ExperienceBase(BaseModel):
    job_title: str
    company: str
    location: str | None = None
    start_date: date
    end_date: date | None = None
    description: str | None = None


class ExperienceOut(ExperienceBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Education ----------
class EducationBase(BaseModel):
    institution: str
    qualification: str
    field_of_study: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    grade: str | None = None
    description: str | None = None


class EducationOut(EducationBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Skills ----------
class SkillBase(BaseModel):
    name: str
    category: str | None = None
    level: str | None = None


class SkillOut(SkillBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Other sections (read-only for now; you'll add create/update) ----------


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    role: str | None = None
    url: str | None = None
    technologies: str | None = None
    description: str | None = None
    start_date: date | None = None
    end_date: date | None = None


class CertificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    issuer: str | None = None
    issue_date: date | None = None
    expiry_date: date | None = None
    credential_url: str | None = None


class FullProfileOut(ProfileOut):
    """Everything about the user in one object. This is what the AI will read in Phase 1."""
    experiences: list[ExperienceOut] = []
    education: list[EducationOut] = []
    skills: list[SkillOut] = []
    projects: list[ProjectOut] = []
    certifications: list[CertificationOut] = []
