"""Talks to Claude. Everything AI-related lives here so the rest of the app stays simple to test."""
from contextlib import contextmanager

import anthropic
from pydantic import ValidationError

import re

from app.config import settings
from app.schemas import GapAssessment, JobRequirements

CV_SYSTEM_PROMPT = """You write CVs tailored to one specific job.

You receive the candidate's profile as JSON, the job they are applying for, and sometimes a list of \
requirements extracted from the job ad. Follow these rules:

Truthfulness (most important)
1. The profile is the single source of truth. Every fact in the CV (employers, job titles, dates, \
numbers, tools, skills, grades, certifications, achievements) must come from the profile.
2. Never invent or guess. If a detail is missing, leave it out. Do not add numbers, tools or results \
that the profile doesn't state, and don't link a tool to a task unless the profile does. If you mention \
total years of experience, use total_experience.years (rounded down to a whole number) and ignore any \
other figure in the profile.
3. Use only the bullets provided. When jobs come with "selected_bullets", those are the only bullets \
you may use for them; do not add bullets from elsewhere. Keep the jobs in "experiences" in the order given \
(most relevant first). List every job in "other_experience" under an "Other Experience" heading, one line \
each (job title, company, dates), with no bullets.
4. A general skill is not experience with a specific tool or platform. For example, "Customer Relationship \
Management" as a skill does not mean the person has used CRM software such as Salesforce or Zendesk. \
Only mention a tool, platform or system if the profile names it.
5. Never describe the person with a job title they haven't held. If they were a "Technical Support Intern", \
don't call them a "Technical Support Specialist", even if that's the job they're applying for. \
Describe what they did instead.

Tailoring
6. Choose the experience, projects, skills and certifications most relevant to this job and put them first. \
Leave out items that add nothing for this role.
7. Rewrite bullet points so each one leads with what this job cares about, using the job's own wording where \
it truthfully describes the work (e.g. "email correspondence and scheduling" -> "managed email and calendars" \
for a role that asks for calendar management). Keep every number, tool, employer and fact exactly as given, \
and never add a duty or result the original bullet doesn't state. You may merge, shorten and reorder bullets. \
Only copy a bullet word for word if it already fits the job well.
8. Use the job's own terminology and keywords where they truthfully describe what the person did. \
Never use a keyword to claim experience the profile doesn't show.
9. In the Skills section, keep only skills that are relevant to this job and that appear in the profile. \
When skills come as "most_relevant" and "other", start from most_relevant and add an "other" skill only if \
this job asks for it. The same goes for certifications.
10. Write a 2-4 sentence professional summary aimed at this role, built only from facts in the profile, \
in the third person without pronouns (e.g. "Resolved 25-35 requests weekly", not "I resolve...").

Format
11. Output a clean CV in Markdown: the candidate's name as a level-1 heading, contact details on one line, \
then sections with level-2 headings (Summary, Experience, Other Experience, Projects, Education, Skills, \
Certifications). Skip any section with nothing to put in it.
12. Output only the CV itself: no preface, no notes to the user, no code fences."""

COVER_LETTER_SYSTEM_PROMPT = """You write cover letters for one specific job.

You receive the candidate's profile as JSON (sometimes only the parts most relevant to this job), the job, \
sometimes a list of requirements extracted from the job ad, and sometimes the candidate's own motivation notes.

Truthfulness (most important)
1. Every fact in the letter (employers, job titles, dates, numbers, tools, skills, certifications, achievements) \
must come from the profile. Never invent or guess. Don't add numbers, tools or results the profile doesn't state.
2. A general skill is not experience with a specific tool or platform. Only mention a tool, platform or system \
if the profile names it, and don't link a tool to a task unless the profile does (a listed "Canva" skill doesn't \
mean a particular piece of work was made in Canva).
3. Never describe the candidate with a job title they haven't held.
3a. No invented detail or colour: don't describe what customers or situations were like, how often something \
happened ("daily", "every"), what a certificate covers, or what a role "taught" or "showed" the candidate. \
Restate the profile's facts and connect them to the job's needs; that is enough.
4. Never claim a requirement the profile doesn't show. Gaps: mention ONLY the items in <gaps_to_mention>, \
in a single short sentence after the strengths, saying the candidate is keen to learn them, without \
apologising. Don't mention any other missing requirement, industry or tool. If <gaps_to_mention> is empty, \
don't mention gaps at all.
5. Before calling something a gap, check the whole profile, including skills and certifications listed as \
"other". The profile may show only the parts most relevant to this job, so something you can't see is not \
proof the candidate lacks it. If unsure, say nothing about it.
6. No unsupported personal qualities. Only describe the candidate as independent, detail-oriented, a fast \
learner, a team player and so on if the profile shows it (a listed skill such as "Time Management" counts).
7. Motivation can come ONLY from the motivation notes. If notes are given, use them faithfully: don't exaggerate \
them or add details. If there are no notes, keep the motivation general and tied to the role itself (for example, \
how the role's work matches what the candidate has done). Never invent personal reasons, admiration for the \
company, knowledge of its products, values or reputation, or career plans.
8. If you mention total years of experience, use total_experience.years (rounded down to a whole number) and \
ignore any other figure in the profile.

Content
9. Pick the 2 or 3 strongest achievements in the profile that best match the job's requirements, and connect \
each one to what the job needs. Prefer achievements with concrete numbers.
10. Address the company by name when it's known. If it isn't, refer to "your team".

Voice: sound like a real person, not a template
11. Open with a concrete fact about the candidate's most relevant work (e.g. "For a year and a half I handled \
40-50 customer enquiries a day at ConnectPlus Services..."), not with a statement that they're applying.
12. Close with one short, plain sentence (e.g. offering to talk or to share examples of their work), not a \
stock closing line.
13. Never use these phrases or close variants: "I am writing to", "I am applying for", "express my interest", \
"keen interest", "excited", "thrilled", "passionate", "perfect fit", "great fit", "aligns with", \
"I believe I would", "I am confident that", "track record", "proven ability", "valuable asset", "leverage", \
"dynamic", "fast-paced", "hit the ground running", "this experience taught me", "I should be open", \
"I would welcome the chance", "I would welcome the opportunity", "I look forward to hearing from you", \
"thank you for considering", "Furthermore", "Moreover", "In today's", "taught me", "showed me".
14. Vary sentence length and structure. Don't start every paragraph the same way.

Format
15. Length: the letter body (everything between the greeting and "Kind regards,") must be 260-350 words: \
3 or 4 full paragraphs, the last of which ends with the closing sentence. Don't put the closing sentence in a \
paragraph of its own. If the strongest achievements run short, explain more fully how each one fits the job's \
requirements rather than adding new claims. Plain text, no Markdown.
16. Start with exactly the greeting line given in <greeting>, and end with "Kind regards," followed by the \
candidate's full name. No address block, no date, no subject line, no placeholders such as \
[Your Address].
17. Output only the letter: no preface, no notes to the user."""

GAP_SYSTEM_PROMPT = """You compare a candidate's profile with a job's requirements and list what's missing.

Rules:
- A requirement is met if the profile shows the skill, tool, certification, qualification or clearly \
equivalent experience anywhere (experience bullets, skills, projects, certifications, education).
- List a requirement as missing only if the specific thing asked for (a named tool, platform, system, \
certification, regulation, degree subject or industry) is absent from the profile. Related general experience \
doesn't make a named tool or industry present, but don't list soft skills the profile reasonably supports.
- Don't list practical conditions the profile can't show (e.g. "reliable internet connection").
- Use short wording taken from the requirement. Order each list from most to least important for this job."""

EXTRACT_SYSTEM_PROMPT = """You read job ads and extract what the employer is asking for.

Rules:
- Include only what the ad states or clearly implies. Don't add skills that are typical for this kind \
of role but not mentioned in the ad.
- Keep each item short: a few words, not full sentences.
- For keywords, use the ad's own wording.
- If the ad doesn't state a seniority level, use "not stated".
- Use an empty list when nothing in the ad fits a field."""


class AIError(Exception):
    """Raised when the AI step failed. The message is safe to show to the user."""


def _client() -> anthropic.Anthropic:
    # Created per call, so tests and app startup never need an API key.
    # With no key in .env, the SDK falls back to the ANTHROPIC_API_KEY environment variable.
    return anthropic.Anthropic(api_key=settings.anthropic_api_key or None)


@contextmanager
def _friendly_errors():
    """Turns SDK errors into AIError with a message the user can act on."""
    try:
        yield
    except anthropic.AuthenticationError:
        raise AIError("The AI API key is invalid. Check ANTHROPIC_API_KEY in .env.")
    except TypeError as e:
        # The SDK raises a plain TypeError when no key is configured at all.
        if "authentication" not in str(e).lower():
            raise
        raise AIError("No AI API key is set. Add ANTHROPIC_API_KEY to .env and restart the server.")
    except anthropic.RateLimitError:
        raise AIError("The AI service is busy. Try again shortly.")
    except anthropic.APIStatusError as e:
        # Pass on Anthropic's own explanation (e.g. "Your credit balance is too low...").
        reason = e.body.get("error", {}).get("message") if isinstance(e.body, dict) else None
        raise AIError(f"The AI service returned an error ({e.status_code}): {reason or 'no details given'}")
    except anthropic.APIConnectionError:
        raise AIError("Couldn't reach the AI service. Check your internet connection.")


def _check_finished(response) -> None:
    if response.stop_reason == "refusal":
        raise AIError("The AI declined this request.")
    if response.stop_reason == "max_tokens":
        raise AIError("The AI's answer was cut off before it finished. Try again.")


def extract_requirements(job_title: str, job_description: str) -> JobRequirements:
    """Reads a job ad and returns its requirements as structured data."""
    user_message = f"<job_title>{job_title}</job_title>\n\n<job_ad>\n{job_description}\n</job_ad>"

    with _friendly_errors():
        try:
            # messages.parse sends JobRequirements' JSON schema as a structured-output format, so the
            # model must answer with JSON matching it, and the SDK validates it into a JobRequirements.
            response = _client().messages.parse(
                model=settings.ai_model,
                max_tokens=16000,
                system=EXTRACT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                output_format=JobRequirements,
            )
        except ValidationError:
            raise AIError("The AI returned requirements in an unexpected shape. Try again.")

    _check_finished(response)
    if response.parsed_output is None:
        raise AIError("The AI didn't return any requirements. Try again.")
    return response.parsed_output


def generate_cv(
    profile_json: str,
    job_title: str,
    job_description: str | None,
    requirements: JobRequirements | None = None,
) -> str:
    """Returns a Markdown CV tailored to the job, written only from the profile."""
    parts = [
        f"<profile>\n{profile_json}\n</profile>",
        f"<job>\nJob title: {job_title}\n\nJob description:\n"
        f"{job_description or '(none provided; tailor to the job title)'}\n</job>",
    ]
    if requirements is not None:
        parts.append(f"<requirements>\n{requirements.model_dump_json(indent=2)}\n</requirements>")
    parts.append("Write the tailored CV now.")
    return _complete(CV_SYSTEM_PROMPT, "\n\n".join(parts), "CV")


LETTER_MIN_WORDS, LETTER_MAX_WORDS = 250, 350
MAX_GAPS_TO_MENTION = 2


def assess_gaps(full_profile_json: str, requirements: JobRequirements) -> GapAssessment:
    """Asks the model (with structured output) which requirements the whole profile shows no evidence for."""
    user_message = (
        f"<profile>\n{full_profile_json}\n</profile>\n\n"
        f"<requirements>\n{requirements.model_dump_json(indent=2)}\n</requirements>"
    )
    with _friendly_errors():
        try:
            response = _client().messages.parse(
                model=settings.ai_model,
                max_tokens=16000,
                system=GAP_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                output_format=GapAssessment,
            )
        except ValidationError:
            raise AIError("The AI returned the gap check in an unexpected shape. Try again.")
    _check_finished(response)
    if response.parsed_output is None:
        raise AIError("The AI didn't return a gap check. Try again.")
    return response.parsed_output


def choose_gaps(assessment: GapAssessment) -> list[str]:
    """At most two required gaps; if nothing required is missing, at most one nice-to-have."""
    if assessment.missing_required:
        return assessment.missing_required[:MAX_GAPS_TO_MENTION]
    return assessment.missing_nice_to_have[:1]


def letter_greeting(company: str | None) -> str:
    return f"Dear {company.strip()} Team," if company and company.strip() else "Dear Hiring Manager,"


def set_greeting(letter: str, company: str | None) -> str:
    """Makes sure the letter opens with the right greeting, replacing whatever the model wrote."""
    lines = letter.strip().splitlines()
    if lines and lines[0].strip().lower().startswith("dear"):
        lines[0] = letter_greeting(company)
    else:
        lines = [letter_greeting(company), ""] + lines
    return "\n".join(lines)


def letter_body_word_count(letter: str) -> int:
    """Words between the greeting line and the sign-off."""
    lines = letter.strip().splitlines()
    if lines and lines[0].strip().lower().startswith("dear"):
        lines = lines[1:]
    body = re.split(r"^\s*(kind regards|best regards|regards|sincerely|yours sincerely)\b",
                    "\n".join(lines), maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]
    return len(body.split())


def generate_cover_letter(context, company: str | None, motivation_notes: str | None) -> str:
    """Returns a plain-text cover letter for the job, built from the same context as the CV.

    `context` is a GenerationContext. Motivation comes only from `motivation_notes`; without notes the letter
    keeps its motivation general. Three things are enforced in code because the model doesn't follow them
    reliably: which gaps may be mentioned (at most two, chosen by assess_gaps), the length (one rewrite if
    it's outside 250-350 words), and the greeting.
    """
    gaps = choose_gaps(assess_gaps(context.full_profile_json, context.requirements)) if context.requirements else []

    parts = [
        f"<profile>\n{context.profile_json}\n</profile>",
        f"<job>\nJob title: {context.job_title}\nCompany: {company or '(not given)'}\n\nJob description:\n"
        f"{context.job_description or '(none provided)'}\n</job>",
    ]
    if context.requirements is not None:
        parts.append(f"<requirements>\n{context.requirements.model_dump_json(indent=2)}\n</requirements>")
    parts.append(f"<gaps_to_mention>\n{chr(10).join(gaps) if gaps else '(none)'}\n</gaps_to_mention>")
    parts.append(
        f"<motivation_notes>\n{motivation_notes}\n</motivation_notes>" if motivation_notes
        else "<motivation_notes>(none given: keep the motivation general, as the rules say)</motivation_notes>"
    )
    parts.append(f"<greeting>{letter_greeting(company)}</greeting>")
    parts.append("Write the cover letter now.")

    messages = [{"role": "user", "content": "\n\n".join(parts)}]
    letter = _complete_messages(COVER_LETTER_SYSTEM_PROMPT, messages, "cover letter")

    words = letter_body_word_count(letter)
    if not LETTER_MIN_WORDS <= words <= LETTER_MAX_WORDS:
        messages += [
            {"role": "assistant", "content": letter},
            {"role": "user", "content": (
                f"The letter body is {words} words; it must be between {LETTER_MIN_WORDS} and {LETTER_MAX_WORDS}. "
                "Rewrite the whole letter so the body is 280-320 words. Follow every rule as before: use only "
                "facts from the profile, add no new claims, and expand by explaining more fully how the "
                "achievements fit the job's requirements."
            )},
        ]
        letter = _complete_messages(COVER_LETTER_SYSTEM_PROMPT, messages, "cover letter")

    return set_greeting(letter, company)


def _complete(system: str, user_message: str, what: str) -> str:
    """Sends one request and returns the text, with the same error handling for every document type."""
    return _complete_messages(system, [{"role": "user", "content": user_message}], what)


def _complete_messages(system: str, messages: list[dict], what: str) -> str:
    with _friendly_errors():
        response = _client().messages.create(
            model=settings.ai_model,
            max_tokens=16000,
            system=system,
            messages=messages,
        )

    _check_finished(response)
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise AIError(f"The AI returned an empty {what}. Try again.")
    return text
