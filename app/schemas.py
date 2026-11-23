"""Pydantic schemas: the shape of data going in and out of the API."""
from datetime import date, datetime

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


# ---------- Projects ----------
class ProjectBase(BaseModel):
    name: str
    role: str | None = None
    url: str | None = None
    technologies: str | None = None  # comma-separated for now
    description: str | None = None
    start_date: date | None = None
    end_date: date | None = None


class ProjectOut(ProjectBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Certifications ----------
class CertificationBase(BaseModel):
    name: str
    issuer: str | None = None
    issue_date: date | None = None
    expiry_date: date | None = None
    credential_url: str | None = None


class CertificationOut(CertificationBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------- Full profile ----------
class FullProfileOut(ProfileOut):
    """Everything about the user in one object. This is what the AI will read in Phase 1."""
    experiences: list[ExperienceOut] = []
    education: list[EducationOut] = []
    skills: list[SkillOut] = []
    projects: list[ProjectOut] = []
    certifications: list[CertificationOut] = []


# ---------- Job requirements (extracted from a job ad by the AI) ----------
class JobRequirements(BaseModel):
    required_skills: list[str] = Field(description="Skills, tools or qualifications the ad says are required.")
    nice_to_have_skills: list[str] = Field(description="Skills the ad lists as preferred, a plus, or desirable.")
    keywords: list[str] = Field(description="Important terms and phrases from the ad, in the ad's own wording.")
    seniority: str = Field(description="Seniority level, e.g. 'entry-level', 'mid-level', 'senior', or 'not stated'.")
    key_responsibilities: list[str] = Field(description="The main duties of the role, each as a short phrase.")


class MatchOut(BaseModel):
    """A profile chunk picked for a job, and why (GET /applications/{id}/matches)."""
    chunk_id: int
    source_type: str  # experience, project, skill or certification
    source_id: int
    text: str
    score: float  # cosine similarity: 1 = same meaning, 0 = unrelated
    matched_requirement: str


# ---------- Job applications ----------
class JobApplicationCreate(BaseModel):
    job_title: str = Field(min_length=1, max_length=200)
    company: str | None = Field(default=None, max_length=200)
    job_description: str | None = None  # paste the job ad here; the AI tailors the CV to it


class JobApplicationOut(JobApplicationCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str
    requirements: JobRequirements | None = None  # empty until POST /applications/{id}/analyze
    created_at: datetime


# ---------- Generated documents ----------
class GeneratedDocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    job_application_id: int
    doc_type: str
    version: int
    content: str
    model_used: str | None = None
    created_at: datetime
