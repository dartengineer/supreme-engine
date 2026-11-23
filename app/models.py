"""Database tables (SQLAlchemy 2.0 style).

One user has one profile. A profile has many experiences, education entries,
skills, projects and certifications. A user also has many job applications,
and each application has many generated documents (CVs and cover letters).
"""
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, Date, DateTime, ForeignKey, Index, Integer, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.services.embeddings import DIMENSIONS


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))

    profile: Mapped["Profile"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    job_applications: Mapped[list["JobApplication"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Profile(TimestampMixin, Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    full_name: Mapped[str] = mapped_column(String(150))
    headline: Mapped[str | None] = mapped_column(String(200))  # e.g. "Graphic Designer & Virtual Assistant"
    phone: Mapped[str | None] = mapped_column(String(50))
    location: Mapped[str | None] = mapped_column(String(150))
    linkedin_url: Mapped[str | None] = mapped_column(String(255))
    portfolio_url: Mapped[str | None] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="profile")
    experiences: Mapped[list["Experience"]] = relationship(back_populates="profile", cascade="all, delete-orphan")
    education: Mapped[list["Education"]] = relationship(back_populates="profile", cascade="all, delete-orphan")
    skills: Mapped[list["Skill"]] = relationship(back_populates="profile", cascade="all, delete-orphan")
    projects: Mapped[list["Project"]] = relationship(back_populates="profile", cascade="all, delete-orphan")
    certifications: Mapped[list["Certification"]] = relationship(back_populates="profile", cascade="all, delete-orphan")
    chunks: Mapped[list["ProfileChunk"]] = relationship(back_populates="profile", cascade="all, delete-orphan")


class ProfileChunk(Base):
    """One small, searchable piece of a profile (an experience bullet, a project, a skill or a certification).

    Rebuilt from scratch by reindex_profile() whenever the profile changes, so never edit these by hand.
    """
    __tablename__ = "profile_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(30))  # experience, project, skill or certification
    source_id: Mapped[int] = mapped_column(Integer)  # id of the row it came from
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(DIMENSIONS))

    profile: Mapped[Profile] = relationship(back_populates="chunks")


class Experience(TimestampMixin, Base):
    __tablename__ = "experiences"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    job_title: Mapped[str] = mapped_column(String(150))
    company: Mapped[str] = mapped_column(String(150))
    location: Mapped[str | None] = mapped_column(String(150))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)  # empty = current job
    description: Mapped[str | None] = mapped_column(Text)  # responsibilities and achievements

    profile: Mapped[Profile] = relationship(back_populates="experiences")


class Education(TimestampMixin, Base):
    __tablename__ = "education"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    institution: Mapped[str] = mapped_column(String(200))
    qualification: Mapped[str] = mapped_column(String(150))  # e.g. "B.Sc."
    field_of_study: Mapped[str | None] = mapped_column(String(150))
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    grade: Mapped[str | None] = mapped_column(String(50))
    description: Mapped[str | None] = mapped_column(Text)

    profile: Mapped[Profile] = relationship(back_populates="education")


class Skill(TimestampMixin, Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    category: Mapped[str | None] = mapped_column(String(100))  # e.g. "Design tools", "Soft skills"
    level: Mapped[str | None] = mapped_column(String(50))  # e.g. "Beginner", "Expert"

    profile: Mapped[Profile] = relationship(back_populates="skills")


class Project(TimestampMixin, Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(150))
    role: Mapped[str | None] = mapped_column(String(150))
    url: Mapped[str | None] = mapped_column(String(255))
    technologies: Mapped[str | None] = mapped_column(String(255))  # comma-separated for now
    description: Mapped[str | None] = mapped_column(Text)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)

    profile: Mapped[Profile] = relationship(back_populates="projects")


class Certification(TimestampMixin, Base):
    __tablename__ = "certifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    issuer: Mapped[str | None] = mapped_column(String(150))
    issue_date: Mapped[date | None] = mapped_column(Date)
    expiry_date: Mapped[date | None] = mapped_column(Date)
    credential_url: Mapped[str | None] = mapped_column(String(255))

    profile: Mapped[Profile] = relationship(back_populates="certifications")


class JobApplication(TimestampMixin, Base):
    """A job the user wants to apply for. Used in Phase 1 to generate documents."""
    __tablename__ = "job_applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_title: Mapped[str] = mapped_column(String(200))
    company: Mapped[str | None] = mapped_column(String(200))
    job_description: Mapped[str | None] = mapped_column(Text)
    requirements: Mapped[dict | None] = mapped_column(JSON)  # filled by POST /applications/{id}/analyze
    # Optional, written by the user: why they want this job ("I use their app"). The only allowed source of
    # personal motivation in a cover letter, because the profile can't supply it.
    motivation_notes: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(50), default="draft")  # draft, applied, interview, offer, rejected

    user: Mapped[User] = relationship(back_populates="job_applications")
    documents: Mapped[list["GeneratedDocument"]] = relationship(back_populates="job_application", cascade="all, delete-orphan")

    @property
    def document_counts(self) -> dict[str, int]:
        """How many documents of each type exist, e.g. {"cv": 3, "cover_letter": 1}."""
        counts = {"cv": 0, "cover_letter": 0}
        for d in self.documents:
            counts[d.doc_type] = counts.get(d.doc_type, 0) + 1
        return counts

    @property
    def last_generated_at(self) -> datetime | None:
        return max((d.created_at for d in self.documents), default=None)


class GeneratedDocument(TimestampMixin, Base):
    """Every CV or cover letter the AI generates, kept as history."""
    __tablename__ = "generated_documents"
    __table_args__ = (
        # At most one final document per type per application, enforced by the database itself.
        Index("uq_one_final_per_doc_type", "job_application_id", "doc_type", unique=True,
              postgresql_where=text("is_final"), sqlite_where=text("is_final")),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_application_id: Mapped[int] = mapped_column(ForeignKey("job_applications.id", ondelete="CASCADE"), index=True)
    doc_type: Mapped[str] = mapped_column(String(30))  # "cv" or "cover_letter"
    version: Mapped[int] = mapped_column(Integer, default=1)
    content: Mapped[str] = mapped_column(Text)  # the generated text/JSON
    model_used: Mapped[str | None] = mapped_column(String(100))
    is_final: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))  # the one you'll send

    job_application: Mapped[JobApplication] = relationship(back_populates="documents")
