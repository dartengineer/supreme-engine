"""Turns the model's CV draft into a complete CV document, using the profile for every hard fact.

The model only writes the summary, chooses jobs/projects/education/certifications by ref, writes bullets and
picks skills. Names, employers, job titles, dates, institutions and certificate names come from the profile,
so they can't be invented or misspelt. Refs that don't exist and skills that aren't in the profile are dropped
(and reported), and any job the model left out is still listed under Other Experience.
"""
from datetime import date

from app.schemas import (
    CVCertification,
    CVContact,
    CVDocument,
    CVDraft,
    CVEducation,
    CVJob,
    CVProject,
)


def _month_year(iso: str | None) -> str:
    return date.fromisoformat(iso).strftime("%b %Y") if iso else "Present"


def _clean(bullets: list[str]) -> list[str]:
    out = []
    for b in bullets:
        b = b.strip().lstrip("-*• ").strip()
        if b:
            out.append(b)
    return out


def _job(e: dict, bullets: list[str] | None = None) -> CVJob:
    return CVJob(job_title=e["job_title"], company=e["company"], location=e.get("location"),
                 start=_month_year(e["start_date"]), end=_month_year(e.get("end_date")), bullets=bullets or [])


def build_cv(draft: CVDraft, profile: dict) -> tuple[CVDocument, list[str]]:
    """Returns the CV and a list of what was dropped (unknown refs or skills), for the check report."""
    jobs = {f"job{e['id']}": e for e in profile.get("experiences", [])}
    projects = {f"proj{p['id']}": p for p in profile.get("projects", [])}
    education = {f"edu{ed['id']}": ed for ed in profile.get("education", [])}
    certs = {f"cert{c['id']}": c for c in profile.get("certifications", [])}
    skills = {s["name"].strip().lower(): s["name"] for s in profile.get("skills", [])}
    dropped: list[str] = []

    def known(ref: str, table: dict, what: str) -> bool:
        if ref in table:
            return True
        dropped.append(f"Unknown {what} ref '{ref}' was removed.")
        return False

    used_jobs: set[str] = set()
    chosen: list[tuple[dict, list[str]]] = []
    for j in draft.experience:
        if known(j.ref, jobs, "job") and j.ref not in used_jobs:
            used_jobs.add(j.ref)
            chosen.append((jobs[j.ref], _clean(j.bullets)))
    # The model picks which jobs and bullets to show; code shows them newest first, as recruiters expect
    # (a current job first, then by start date).
    chosen.sort(key=lambda pair: (pair[0].get("end_date") is None, pair[0]["start_date"]), reverse=True)
    experience = [_job(e, bullets) for e, bullets in chosen]

    # Every job appears somewhere: the ones not shown with bullets are listed compactly, newest first.
    other_refs = [r for r in draft.other_experience_refs if known(r, jobs, "job") and r not in used_jobs]
    other_refs += [r for r in jobs if r not in used_jobs and r not in other_refs]
    other = [_job(jobs[r]) for r in sorted(set(other_refs), key=lambda r: jobs[r]["start_date"], reverse=True)]

    chosen_projects = []
    for p in draft.projects:
        if known(p.ref, projects, "project"):
            src = projects[p.ref]
            chosen_projects.append(CVProject(name=src["name"], technologies=src.get("technologies"),
                                             url=src.get("url"), bullets=_clean(p.bullets)))

    chosen_skills = []
    for name in draft.skills:
        real = skills.get(name.strip().lower())
        if real is None:
            dropped.append(f"Skill '{name}' isn't in the profile and was removed.")
        elif real not in chosen_skills:
            chosen_skills.append(real)

    def edu(ed: dict) -> CVEducation:
        years = [d[:4] for d in (ed.get("start_date"), ed.get("end_date")) if d]
        return CVEducation(qualification=ed["qualification"], field_of_study=ed.get("field_of_study"),
                           institution=ed["institution"], dates=" – ".join(years) or None,
                           details=ed.get("description") or (f"Grade: {ed['grade']}" if ed.get("grade") else None))

    return CVDocument(
        name=profile["full_name"],
        contact=CVContact(email=profile.get("email"), phone=profile.get("phone"), location=profile.get("location"),
                          linkedin_url=profile.get("linkedin_url"), portfolio_url=profile.get("portfolio_url")),
        summary=draft.summary.strip(),
        experience=experience,
        other_experience=other,
        projects=chosen_projects,
        education=[edu(education[r]) for r in dict.fromkeys(draft.education_refs) if known(r, education, "education")],
        skills=chosen_skills,
        certifications=[CVCertification(name=certs[r]["name"], issuer=certs[r].get("issuer"))
                        for r in dict.fromkeys(draft.certification_refs) if known(r, certs, "certification")],
    ), dropped


def _short_url(url: str | None) -> str | None:
    """'https://name.github.io/' -> 'name.github.io': shorter, and still a working address."""
    if not url:
        return url
    return url.strip().removeprefix("https://").removeprefix("http://").removeprefix("www.").rstrip("/")


def contact_parts(email, phone, location, linkedin_url, portfolio_url) -> str:
    """One contact line for CVs and cover letters: email first, then phone, location and short links."""
    return " | ".join(x for x in (email, phone, location, _short_url(linkedin_url), _short_url(portfolio_url)) if x)


def contact_line(cv: CVDocument) -> str:
    c = cv.contact
    return contact_parts(c.email, c.phone, c.location, c.linkedin_url, c.portfolio_url)


def render_markdown(cv: CVDocument) -> str:
    """The CV as Markdown: shown in the app and read by the fact-checker."""
    out = [f"# {cv.name}"]
    if contact_line(cv):
        out.append(contact_line(cv))
    out += ["", "## Summary", cv.summary]
    if cv.experience:
        out += ["", "## Experience"]
        for j in cv.experience:
            out.append(" | ".join(x for x in (f"**{j.job_title}**", j.company, j.location, f"{j.start} – {j.end}") if x))
            out += [f"- {b}" for b in j.bullets]
            out.append("")
    if cv.other_experience:
        out += ["", "## Other Experience"] + [f"- {j.job_title}, {j.company}, {j.start} – {j.end}" for j in cv.other_experience]
    if cv.projects:
        out += ["", "## Projects"]
        for p in cv.projects:
            out.append(" | ".join(x for x in (f"**{p.name}**", p.technologies) if x))
            out += [f"- {b}" for b in p.bullets]
    if cv.education:
        out += ["", "## Education"]
        for ed in cv.education:
            title = ed.qualification + (f", {ed.field_of_study}" if ed.field_of_study else "")
            out.append(" | ".join(x for x in (f"**{title}**", ed.institution, ed.dates) if x))
            if ed.details:
                out.append(ed.details)
    if cv.skills:
        out += ["", "## Skills"] + [f"- {s}" for s in cv.skills]
    if cv.certifications:
        out += ["", "## Certifications"] + [f"- {c.name}" + (f", {c.issuer}" if c.issuer else "") for c in cv.certifications]
    return "\n".join(out).replace("\n\n\n", "\n\n").strip() + "\n"
