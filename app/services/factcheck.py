"""Fact-checks a generated CV or cover letter against the candidate's profile.

Two layers:
1. Code checks (cheap and certain): numbers, tools, employers/titles/dates, years of experience.
2. Claim check (LLM judge): the document is split into claims, each tied to the sources it came from,
   and every claim is judged supported / stretch / unsupported against those sources.

Everything works on the profile as plain data (FullProfileOut.model_dump(mode="json")), so the same
checker runs in the app and in the offline evaluation (scripts/eval_factcheck.py).
"""
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone

from app.services import ai

# ---------------------------------------------------------------- vocabulary

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
MONTH_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
MONTH_YEAR = rf"({MONTH_RE})\s+(\d{{4}})"
RANGE_RE = re.compile(rf"{MONTH_YEAR}\s*(?:–|—|-|to|until)\s*(?:{MONTH_YEAR}|(present|current|now|date))", re.I)
SINCE_RE = re.compile(rf"\bsince\s+{MONTH_YEAR}", re.I)

# Named tools, platforms, systems, regulations and certifications worth checking. Each is matched as a whole
# phrase, case-insensitively. A tool in the output must exist in the profile (outside gap sentences), and when
# it's attached to a piece of work ("using Canva") the profile must attach it to that work too.
TOOLS = [
    "Canva", "Figma", "Adobe", "Photoshop", "Illustrator", "After Effects", "InDesign", "Premiere Pro", "Lightroom",
    "CorelDRAW", "Microsoft Office", "Microsoft 365", "Office 365", "Excel", "PowerPoint", "Outlook", "Google Workspace",
    "Google Sheets", "Google Docs", "Google Drive", "Google Calendar", "Gmail", "Slack", "Zoom", "Notion", "Jira",
    "Trello", "Asana", "ClickUp", "Monday.com", "HubSpot", "Salesforce", "Zendesk", "Freshdesk", "Intercom",
    "ServiceNow", "Active Directory", "Windows Server", "Azure", "AWS", "Google Cloud", "Firebase", "Flutter", "Dart",
    "Python", "JavaScript", "TypeScript", "HTML", "CSS", "SQL", "React", "Node.js", "Git", "GitHub", "REST API",
    "WordPress", "Shopify", "QuickBooks", "Mailchimp", "Hootsuite", "Buffer", "KYC", "AML", "ITIL", "CompTIA",
    "CRM software", "CRM platform", "CRM platforms", "CRM tool", "CRM tools", "CRM system", "CRM systems",
]
_TOOL_RES = {t: re.compile(rf"(?<![\w.]){re.escape(t)}(?![\w])", re.I) for t in TOOLS}

# Sentences with these words talk about what the candidate lacks or wants to learn; tools there aren't claims.
GAP_MARKERS = re.compile(r"\b(not|never|no experience|haven't|have not|hasn't|yet to|keen to learn|eager to learn|"
                         r"want to learn|willing to learn|happy to learn|new to)\b", re.I)
# Phrases that attach a tool to a piece of work, e.g. "using Figma and Canva", "designed in Canva".
LINK_WORDS = r"(?:using|in|with|via|through|on)"
SKILL_LEAD = re.compile(r"\b(proficien\w*|skill\w*|knowledge|familiar\w*|experienced|competen\w*|fluent|comfortable|"
                        r"work(?:s|ing)?\s+with|use[sd]?|using\s+tools)\b", re.I)
# Words showing a sentence is about the job being applied for, not a title the candidate holds.
ROLE_WORDS = re.compile(r"\b(role|position|job|vacancy|opening|opportunity|apply|applying|application|this|your)\b", re.I)
URL_OR_EMAIL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+|\b[\w-]+(?:\.[\w-]+)+/\S*|\b[\w-]+\.(?:io|com|org|net|ng|dev)\b\S*)", re.I)
NUMBER_WORDS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"])}
YEARS_RE = re.compile(r"\b(over|more than|nearly|almost|about|around|approximately)?\s*(\d+|" + "|".join(NUMBER_WORDS)
                      + r")\+?\s+years?\b", re.I)


# ---------------------------------------------------------------- sources

@dataclass
class Source:
    id: str
    text: str


def _month_year(d: str | None) -> str:
    if not d:
        return "Present"
    dt = date.fromisoformat(d)
    return dt.strftime("%B %Y")


def _bullets(description: str | None) -> list[str]:
    return [b.lstrip("-*• ").strip() for b in (description or "").splitlines() if b.strip().lstrip("-*• ").strip()]


def build_sources(profile: dict, job_title: str, job_description: str | None, motivation_notes: str | None,
                  years: float) -> list[Source]:
    """Numbered pieces of everything a document may legitimately draw on."""
    sources = [Source("B", "Candidate: " + ", ".join(
        f"{k.replace('_', ' ')}: {profile[k]}" for k in ("full_name", "headline", "phone", "location",
                                                         "linkedin_url", "portfolio_url") if profile.get(k)))]
    if profile.get("summary"):
        sources.append(Source("SUM", f"Profile summary written by the candidate: {profile['summary']}"))
    for i, e in enumerate(profile.get("experiences", []), 1):
        sources.append(Source(f"J{i}", f"Job: {e['job_title']} at {e['company']}"
                                       f"{', ' + e['location'] if e.get('location') else ''}, "
                                       f"{_month_year(e['start_date'])} – {_month_year(e.get('end_date'))}"))
    n = 0
    for e in profile.get("experiences", []):
        for b in _bullets(e.get("description")):
            n += 1
            sources.append(Source(f"E{n}", f"{b} (as {e['job_title']} at {e['company']})"))
    for i, p in enumerate(profile.get("projects", []), 1):
        sources.append(Source(f"P{i}", f"Project: {p['name']}"
                                       f"{' (' + p['technologies'] + ')' if p.get('technologies') else ''}. "
                                       + " ".join(_bullets(p.get("description")))))
    for i, s in enumerate(profile.get("skills", []), 1):
        sources.append(Source(f"S{i}", f"Skill: {s['name']}" + (f" ({s['category']})" if s.get("category") else "")))
    for i, c in enumerate(profile.get("certifications", []), 1):
        sources.append(Source(f"C{i}", f"Certification: {c['name']}" + (f", {c['issuer']}" if c.get("issuer") else "")))
    for i, ed in enumerate(profile.get("education", []), 1):
        sources.append(Source(f"ED{i}", "Education: " + ", ".join(
            str(ed[k]) for k in ("qualification", "field_of_study", "institution", "grade", "description")
            if ed.get(k))))
    sources.append(Source("T", f"Total experience: {years} years (computed from the job dates)"))
    sources.append(Source("N", f"Candidate's motivation notes: {motivation_notes}" if motivation_notes
                          else "Candidate's motivation notes: (none given)"))
    if job_description:
        sources.append(Source("A", f"Job ad for {job_title}: {job_description}"))
    return sources


def sources_text(sources: list[Source]) -> str:
    return "\n".join(f"[{s.id}] {s.text}" for s in sources)


# ---------------------------------------------------------------- code checks

def _flag(type_: str, text: str, reason: str, source: str | None = None, severity: str = "unsupported") -> dict:
    return {"type": type_, "severity": severity, "text": text.strip(), "source": source, "reason": reason}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("&", " and ")).strip()


def _tokens(s: str) -> set[str]:
    return set(_norm(s).split()) - {"in", "of", "the", "and", "a", "an", "for"}


def _company_core(company: str) -> str:
    """First distinctive word of a company name, used to spot it in text ("TechNova Solutions Ltd." -> "technova")."""
    words = [w for w in _norm(company).split() if w not in {"the"}]
    return words[0] if words else ""


def _numbers(text: str) -> set[str]:
    return {str(int(float(n.replace(",", "")))) if "." not in n else n
            for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def _profile_text(profile: dict) -> str:
    """All the profile's values as text (no ids), for presence checks."""
    parts: list[str] = []

    def walk(v, key=None):
        if key == "id":
            return
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif v is not None:
            parts.append(str(v))
    walk(profile)
    return "\n".join(parts)


def _month(name: str) -> int:
    return MONTHS[name.lower().rstrip(".")[:3]]


def _same_month(year: str, month: str, iso: str | None) -> bool:
    if not iso:
        return False
    d = date.fromisoformat(iso)
    return (int(year), _month(month)) == (d.year, d.month)


def _segments(text: str):
    """Yields (section, paragraph_index, line, is_first_sentence_of_line, sentence), keeping CV structure."""
    section, para = "", 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            para += 1
            continue
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip().lower()
            para += 1
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", stripped) if s.strip()]
        for i, sentence in enumerate(sentences):
            yield section, para, stripped, i == 0, sentence


def code_checks(text: str, profile: dict, sources: list[Source], job_title: str, years: float) -> list[dict]:
    flags: list[dict] = []
    experiences = profile.get("experiences", [])
    profile_txt = _profile_text(profile)
    work_txt = "\n".join([e.get("description") or "" for e in experiences]
                         + [f"{p.get('name')} {p.get('technologies') or ''} {p.get('description') or ''}"
                            for p in profile.get("projects", [])])
    cores = {_company_core(e["company"]): e for e in experiences if _company_core(e["company"])}
    held_titles = [e["job_title"] for e in experiences]

    # Numbers may come from the profile, the job ad, the notes, or the computed years.
    allowed = _numbers(sources_text(sources) + " " + profile_txt)
    allowed |= {str(math.floor(years)), str(round(years))}

    context_exp, context_para = None, -1
    for section, para, line, first_in_line, sentence in _segments(text):
        low = sentence.lower()
        if para != context_para and section not in ("experience", "other experience"):
            # New paragraph outside a CV job list: forget which employer we were talking about.
            context_exp = None
        context_para = para
        mentioned = [e for core, e in cores.items() if re.search(rf"\b{re.escape(core)}\b", _norm(sentence))]
        if len(mentioned) == 1:
            context_exp = mentioned[0]
        exp = mentioned[0] if len(mentioned) == 1 else (context_exp if not mentioned else None)

        # 1. Numbers
        # (URLs and emails are ignored, and "March 2021"-style dates are left to the date check below.)
        number_text = re.sub(MONTH_YEAR, " ", URL_OR_EMAIL.sub(" ", sentence), flags=re.I)
        for n in sorted(_numbers(number_text) - allowed, key=float):
            flags.append(_flag("number", sentence, f"The number {n} doesn't appear anywhere in the profile."))

        # 2. Years of experience
        for qualifier, amount in YEARS_RE.findall(sentence):
            if "experience" not in low:
                continue
            n = int(amount) if amount.isdigit() else NUMBER_WORDS[amount.lower()]
            too_many = n >= years if qualifier.lower() in ("over", "more than") else n > math.floor(years)
            if too_many:
                flags.append(_flag("years", sentence, f"Claims {qualifier + ' ' if qualifier else ''}{n} years of "
                                   f"experience; the job dates add up to {years} years.", f"Total experience: {years}"))

        # 3. Tools
        is_gap_sentence = bool(GAP_MARKERS.search(sentence))
        for tool, rx in _TOOL_RES.items():
            if not rx.search(sentence):
                continue
            if not rx.search(profile_txt):
                if not is_gap_sentence and section not in ("skills",):
                    flags.append(_flag("tool", sentence, f'"{tool}" isn\'t mentioned anywhere in the profile.'))
                elif section == "skills":
                    flags.append(_flag("tool", sentence, f'"{tool}" is listed as a skill but isn\'t in the profile.'))
                continue
            linked = re.search(rf"\b{LINK_WORDS}\s+(?:the\s+)?(?:[\w.&+-]+\s*(?:,|and|or)\s*){{0,3}}{re.escape(tool)}\b",
                               sentence, re.I)
            # "proficient in Canva", "I work with Excel", "knowledge of ..." describe a skill, not a piece of work.
            if linked and SKILL_LEAD.search(sentence[max(0, linked.start() - 30):linked.start() + 6]):
                linked = None
            if linked and not is_gap_sentence:
                where = exp.get("description") or "" if exp else work_txt
                if not rx.search(where):
                    whose = f"the {exp['job_title']} job at {exp['company']}" if exp else "any job or project"
                    flags.append(_flag("tool_attachment", sentence,
                                       f'"{tool}" is attached to work here, but the profile never connects it to {whose}.',
                                       exp.get("description") if exp else None))

        # 4. Titles: the target job title used as the candidate's own title
        jt = job_title.strip()
        if jt and re.search(rf"\b{re.escape(jt)}\b", sentence, re.I) \
                and not any(_tokens(jt) <= _tokens(t) for t in held_titles) and not ROLE_WORDS.search(sentence):
            flags.append(_flag("title", sentence, f'Calls the candidate a "{jt}", a title they haven\'t held.',
                               "Titles held: " + "; ".join(held_titles)))

        # "as a <title> at <company>" must match the title held there
        m = re.search(r"\bas an?\s+([A-Z][\w&–\- ]+?)\s+(?:at|with|for)\s+", sentence)
        if m and exp and not _tokens(m.group(1)) <= _tokens(exp["job_title"]):
            flags.append(_flag("title", sentence, f'Says "{m.group(1)}" at {exp["company"]}; the profile says '
                                                  f'"{exp["job_title"]}".', exp["job_title"]))

        # 5. Dates
        if exp:
            for m1, y1, m2, y2, present in RANGE_RE.findall(sentence):
                start_ok = _same_month(y1, m1, exp["start_date"])
                end_ok = (not exp.get("end_date")) if present else (bool(m2) and _same_month(y2, m2, exp.get("end_date")))
                if not (start_ok and end_ok):
                    flags.append(_flag("date", sentence, f"Dates don't match the profile for {exp['company']}.",
                                       f"{_month_year(exp['start_date'])} – {_month_year(exp.get('end_date'))}"))
            for m1, y1 in SINCE_RE.findall(sentence):
                if not _same_month(y1, m1, exp["start_date"]):
                    flags.append(_flag("date", sentence, f"Start date doesn't match the profile for {exp['company']}.",
                                       _month_year(exp["start_date"])))

        # 6. Employers and titles in CV job headings ("**Title** | Company | ... ")
        if line.startswith("**") and first_in_line and section in ("experience", "other experience"):
            heading_title = re.match(r"\*\*(.+?)\*\*", line)
            if not mentioned:
                flags.append(_flag("employer", line, "This job heading names no employer from the profile."))
            elif heading_title and len(mentioned) == 1 and not _tokens(heading_title.group(1)) <= _tokens(mentioned[0]["job_title"]):
                flags.append(_flag("title", line, f'Heading says "{heading_title.group(1)}"; the profile title at '
                                                  f'{mentioned[0]["company"]} is "{mentioned[0]["job_title"]}".',
                                   mentioned[0]["job_title"]))
    return _dedupe(flags)


def _dedupe(flags: list[dict]) -> list[dict]:
    seen, out = set(), []
    for f in flags:
        key = (f["type"], f["text"], f["reason"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


# ---------------------------------------------------------------- claim check (LLM)

JUDGED_KINDS = {"candidate_fact", "gap", "motivation"}


def claim_check(text: str, sources: list[Source]) -> list[dict]:
    """Splits the document into claims and judges each one against its cited sources."""
    by_id = {s.id: s for s in sources}
    claims = [c for c in ai.extract_claims(text, sources_text(sources)).claims if c.kind in JUDGED_KINDS]
    profile_overview = "\n".join(s.text for s in sources if s.id[0] in "JESCP" or s.id.startswith("ED"))

    def judge(c) -> dict:
        cited = [by_id[i] for i in c.source_ids if i in by_id]
        base = {"claim": c.claim, "kind": c.kind, "source_ids": [s.id for s in cited]}
        notes = by_id["N"]
        has_notes = not notes.text.endswith("(none given)")
        if c.kind == "gap":
            # "Keen to learn" can come from the notes as well as from what the profile lacks.
            source_text = profile_overview + "\n" + notes.text
        else:
            # The notes travel with every claim when they exist: a motivation sentence is sometimes
            # labelled as a candidate fact, and it must still be judged against the notes.
            extra = [notes] if (c.kind == "motivation" or has_notes) and notes not in cited else []
            source_text = "\n".join(s.text for s in cited + extra)
        if not cited and c.kind == "candidate_fact" and not has_notes:
            return base | {"verdict": "unsupported", "reason": "No source in the profile backs this claim."}
        # A claim copied word for word from its source needs no judge.
        if c.kind == "candidate_fact" and any(_norm(c.claim) in _norm(s.text) for s in cited):
            return base | {"verdict": "supported", "reason": "Appears word for word in the source."}
        v = ai.judge_claim(c.claim, c.kind, source_text)
        return base | {"verdict": v.verdict, "reason": v.reason}

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(judge, claims))


# ---------------------------------------------------------------- report

def check_document(text: str, profile: dict, *, job_title: str, job_description: str | None,
                   motivation_notes: str | None, years: float, use_llm: bool = True) -> dict:
    """Runs every check and returns a report: {status, checked_at, summary, flags, claims, error}."""
    sources = build_sources(profile, job_title, job_description, motivation_notes, years)
    flags = code_checks(text, profile, sources, job_title, years)
    claims: list[dict] = []
    status, error = "ok", None
    if use_llm:
        try:
            claims = claim_check(text, sources)
        except ai.AIError as e:
            status, error = "partial", f"The claim check couldn't run: {e}"
    by_id = {s.id: s.text for s in sources}
    for c in claims:
        if c["verdict"] != "supported":
            flags.append({"type": "claim", "severity": c["verdict"], "text": c["claim"],
                          "source": "\n".join(by_id[i] for i in c["source_ids"]) or None, "reason": c["reason"]})
    return {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "unsupported": sum(f["severity"] == "unsupported" for f in flags),
            "stretch": sum(f["severity"] == "stretch" for f in flags),
            "code_flags": sum(f["type"] != "claim" for f in flags),
            "claims_checked": len(claims),
        },
        "flags": flags,
        "claims": claims,
        "error": error,
    }


def has_unsupported(report: dict) -> bool:
    return report.get("summary", {}).get("unsupported", 0) > 0


def feedback_text(report: dict) -> str:
    """The flags, written as instructions for the one regeneration."""
    lines = []
    for f in report.get("flags", []):
        label = "UNSUPPORTED" if f["severity"] == "unsupported" else "STRETCH"
        lines.append(f'- {label}: "{f["text"]}" ({f["reason"]})')
    return "\n".join(lines)
