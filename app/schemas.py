"""Pydantic schemas: the shape of data going in and out of the API."""
from datetime import date, datetime
from typing import Literal

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
    email: EmailStr | None = None  # shown on CVs and cover letters (can differ from your login email)
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


class ExtractedClaim(BaseModel):
    claim: str = Field(description="One short, self-contained, checkable statement. Keep any numbers exactly.")
    kind: Literal["candidate_fact", "job_fact", "gap", "motivation", "other"] = Field(
        description="candidate_fact: about the candidate's work, results, skills, qualities, education or contact "
                    "details. job_fact: only about the job or company. gap: something the candidate lacks or wants "
                    "to learn. motivation: why the candidate wants this job. other: greetings, sign-offs, offers "
                    "to talk.")
    source_ids: list[str] = Field(description="IDs of the sources this claim is based on, e.g. ['E3', 'J2']. "
                                              "Empty if no source fits.")


class ClaimList(BaseModel):
    claims: list[ExtractedClaim]


class ClaimVerdict(BaseModel):
    verdict: Literal["supported", "stretch", "unsupported"]
    reason: str = Field(description="One sentence: what in the claim is or isn't backed by the source.")


class MissingRequirement(BaseModel):
    index: int = Field(description="The requirement's number in the list it came from (starting at 1).")
    label: str = Field(description="A short noun phrase that reads naturally after 'I have not yet worked with', "
                                   "e.g. 'Windows Server', 'KYC and AML procedures', 'banking customers'.")


class GapAssessment(BaseModel):
    """Requirements the candidate's profile shows no evidence for (used to keep cover letters honest)."""
    missing_required: list[MissingRequirement] = Field(
        description="Required skills/qualifications the profile shows no evidence for.")
    missing_nice_to_have: list[MissingRequirement] = Field(
        description="Nice-to-have skills the profile shows no evidence for.")


# ---------- Structured CV (v8) ----------
class CVJobDraft(BaseModel):
    ref: str = Field(description="The job's ref from the profile, e.g. 'job3'.")
    bullets: list[str] = Field(description="Bullets for this job, rewritten for the target role.")


class CVProjectDraft(BaseModel):
    ref: str = Field(description="The project's ref from the profile, e.g. 'proj2'.")
    bullets: list[str]


class CVDraft(BaseModel):
    """What the model writes. Names, employers, titles and dates are filled in from the profile by code."""
    summary: str = Field(description="2-4 sentence professional summary aimed at this job.")
    experience: list[CVJobDraft] = Field(description="Jobs shown with bullets, most relevant first.")
    other_experience_refs: list[str] = Field(description="Refs of jobs listed in one line without bullets.")
    projects: list[CVProjectDraft]
    skills: list[str] = Field(description="Skill names copied exactly from the profile, most relevant first.")
    education_refs: list[str]
    certification_refs: list[str]


class CVContact(BaseModel):
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    portfolio_url: str | None = None


class CVJob(BaseModel):
    job_title: str
    company: str
    location: str | None = None
    start: str  # "Mar 2021"
    end: str  # "Oct 2022" or "Present"
    bullets: list[str] = []


class CVProject(BaseModel):
    name: str
    technologies: str | None = None
    url: str | None = None
    bullets: list[str] = []


class CVEducation(BaseModel):
    qualification: str
    field_of_study: str | None = None
    institution: str
    dates: str | None = None
    details: str | None = None


class CVCertification(BaseModel):
    name: str
    issuer: str | None = None


class CVDocument(BaseModel):
    """A complete CV as data: rendered to Markdown, HTML templates, PDF and DOCX."""
    name: str
    contact: CVContact
    summary: str
    experience: list[CVJob]
    other_experience: list[CVJob] = []
    projects: list[CVProject] = []
    education: list[CVEducation] = []
    skills: list[str] = []
    certifications: list[CVCertification] = []


class RequirementMatch(BaseModel):
    requirement: str
    kind: Literal["required", "nice_to_have"]
    label: Literal["strong", "partial", "gap"]
    similarity: float | None = None  # cosine similarity of the closest profile chunk (1 = same meaning)
    evidence: str | None = None  # the closest profile bullet/skill/project/certification
    evidence_type: str | None = None


class MatchReport(BaseModel):
    """How well the profile fits the job (GET /applications/{id}/match). The score is calculated in code."""
    score: int  # 0-100: required items weigh 2, nice-to-haves 1; strong = full credit, partial = half, gap = none
    strong: int
    partial: int
    gaps: int
    matched: list[RequirementMatch]  # strong and partial, each with the bullet that matched it
    missing: list[RequirementMatch]  # the gaps


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
    # Why you want this job, in your own words (e.g. "I use their app daily"). Used only by cover letters.
    motivation_notes: str | None = Field(default=None, max_length=1000)


ApplicationStatus = Literal["draft", "applied", "interview", "offer", "rejected"]


class JobApplicationUpdate(BaseModel):
    """Change any of these fields; leave out the ones you don't want to change."""
    job_title: str | None = Field(default=None, min_length=1, max_length=200)
    company: str | None = Field(default=None, max_length=200)
    job_description: str | None = None
    motivation_notes: str | None = Field(default=None, max_length=1000)
    status: ApplicationStatus | None = None  # draft -> applied -> interview -> offer / rejected


class JobApplicationOut(JobApplicationCreate):
    model_config = ConfigDict(from_attributes=True)
    id: int
    status: str
    requirements: JobRequirements | None = None  # empty until POST /applications/{id}/analyze
    match_score: int | None = None  # 0-100, from GET /applications/{id}/match
    document_counts: dict[str, int] = {}  # e.g. {"cv": 3, "cover_letter": 1}
    last_generated_at: datetime | None = None  # when the latest document was generated
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
    is_final: bool = False  # the version you've chosen to send (one per type per application)
    check_report: dict | None = None  # fact-check: summary, flags (type, text, source, reason) and judged claims
    data: dict | None = None  # the CV as structured data (CVs from v8 on); used by the download templates
    created_at: datetime
