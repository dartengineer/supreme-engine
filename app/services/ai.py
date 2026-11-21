"""Talks to Claude. Everything AI-related lives here so the rest of the app stays simple to test."""
import anthropic

from app.config import settings

SYSTEM_PROMPT = """You write CVs tailored to one specific job.

You receive the candidate's profile as JSON and the job they are applying for. Follow these rules:

1. Use ONLY information that appears in the profile. The profile is the single source of truth.
2. Never invent jobs, employers, job titles, skills, tools, dates, grades, numbers or achievements. \
If a detail is missing, leave it out; do not guess or fill gaps.
3. You may reword, shorten and reorder what is in the profile so it reads well and fits the job, \
as long as the meaning stays true.
4. Choose the experience, projects, skills and certifications most relevant to this job and put them first. \
Leave out items that add nothing for this role.
5. Write a 2-4 sentence professional summary aimed at this role, built only from facts in the profile.
6. Output a clean CV in Markdown: the candidate's name as a level-1 heading, contact details on one line, \
then sections with level-2 headings (Summary, Experience, Projects, Education, Skills, Certifications). \
Skip any section with nothing to put in it.
7. Output only the CV itself: no preface, no notes to the user, no code fences."""


class AIError(Exception):
    """Raised when the CV couldn't be generated. The message is safe to show to the user."""


def _client() -> anthropic.Anthropic:
    # Created per call, so tests and app startup never need an API key.
    # With no key in .env, the SDK falls back to the ANTHROPIC_API_KEY environment variable.
    return anthropic.Anthropic(api_key=settings.anthropic_api_key or None)


def generate_cv(profile_json: str, job_title: str, job_description: str | None) -> str:
    """Returns a Markdown CV tailored to the job, written only from the profile."""
    job = f"Job title: {job_title}\n\nJob description:\n{job_description or '(none provided; tailor to the job title)'}"
    user_message = (
        f"<profile>\n{profile_json}\n</profile>\n\n"
        f"<job>\n{job}\n</job>\n\n"
        "Write the tailored CV now."
    )

    try:
        response = _client().messages.create(
            model=settings.ai_model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AuthenticationError:
        raise AIError("The AI API key is invalid. Check ANTHROPIC_API_KEY in .env.")
    except TypeError as e:
        # The SDK raises a plain TypeError when no key is configured at all.
        if "authentication" not in str(e).lower():
            raise
        raise AIError("No AI API key is set. Add ANTHROPIC_API_KEY to .env and restart the server.")
    except anthropic.RateLimitError:
        raise AIError("The AI service is busy or out of credit. Try again shortly.")
    except anthropic.APIStatusError as e:
        raise AIError(f"The AI service returned an error ({e.status_code}).")
    except anthropic.APIConnectionError:
        raise AIError("Couldn't reach the AI service. Check your internet connection.")

    if response.stop_reason == "refusal":
        raise AIError("The AI declined to write this CV.")
    if response.stop_reason == "max_tokens":
        raise AIError("The CV was cut off before it finished. Try again.")

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise AIError("The AI returned an empty CV. Try again.")
    return text
