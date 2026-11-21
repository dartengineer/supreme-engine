"""Talks to Claude. Everything AI-related lives here so the rest of the app stays simple to test."""
from contextlib import contextmanager

import anthropic
from pydantic import ValidationError

from app.config import settings
from app.schemas import JobRequirements

CV_SYSTEM_PROMPT = """You write CVs tailored to one specific job.

You receive the candidate's profile as JSON, the job they are applying for, and sometimes a list of \
requirements extracted from the job ad. Follow these rules:

Truthfulness (most important)
1. The profile is the single source of truth. Every fact in the CV (employers, job titles, dates, \
numbers, tools, skills, grades, certifications, achievements) must come from the profile.
2. Never invent or guess. If a detail is missing, leave it out. Do not add numbers, tools or results \
that the profile doesn't state, and don't link a tool to a task unless the profile does.

Tailoring
3. Choose the experience, projects, skills and certifications most relevant to this job and put them first. \
Leave out items that add nothing for this role.
4. Rephrase bullet points to highlight experience that matches the required skills and key responsibilities. \
You may reword, merge, shorten and reorder bullets, as long as the meaning stays true.
5. Use the job's own terminology and keywords where they truthfully describe what the person did. \
Never use a keyword to claim experience the profile doesn't show.
6. In the Skills section, keep only skills that are relevant to this job and that appear in the profile.
7. Write a 2-4 sentence professional summary aimed at this role, built only from facts in the profile.

Format
8. Output a clean CV in Markdown: the candidate's name as a level-1 heading, contact details on one line, \
then sections with level-2 headings (Summary, Experience, Projects, Education, Skills, Certifications). \
Skip any section with nothing to put in it.
9. Output only the CV itself: no preface, no notes to the user, no code fences."""

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

    with _friendly_errors():
        response = _client().messages.create(
            model=settings.ai_model,
            max_tokens=16000,
            system=CV_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n\n".join(parts)}],
        )

    _check_finished(response)
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise AIError("The AI returned an empty CV. Try again.")
    return text
