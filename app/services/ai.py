"""Talks to Claude. Everything AI-related lives here so the rest of the app stays simple to test."""
from contextlib import contextmanager

import anthropic
from pydantic import ValidationError

import re

from app.config import settings
from app.schemas import ClaimList, ClaimVerdict, GapAssessment, JobRequirements

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
4. Never claim a requirement the profile doesn't show. Gaps: if a <gap_sentence> is given, include it word for \
word, once, after the strengths, and write no other sentence about what the candidate lacks, hasn't done or \
is still learning. If there is no gap sentence, don't mention gaps at all.
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
- Refer to each missing requirement by its number in its list. Give it a short label that reads naturally \
after "I have not yet worked with" (e.g. "Windows Server", "KYC and AML procedures", "banking customers")."""

CLAIMS_SYSTEM_PROMPT = """You split a generated CV or cover letter into individual claims for fact-checking.

You receive numbered sources (the candidate's profile, the job ad, and sometimes the candidate's motivation notes) \
and the document. Rules:
- Cover every statement about the candidate. Split sentences that say several things into separate claims; \
merge nothing.
- Keep each claim short and self-contained: replace "there" or "this role" with what it refers to, and keep \
numbers, frequencies ("daily", "every") and qualifiers ("still building", "cross-functional") exactly as written.
- For each claim, list the IDs of every source it is based on. If no source backs it, give an empty list. \
Don't cite a source just because it's on a similar topic.
- Section headings, the candidate's name and contact line, greetings and sign-offs are kind "other".
- Statements about what the candidate hasn't done or is keen/eager/willing to learn are kind "gap", not \
"motivation". Only reasons for wanting this job are "motivation"."""

JUDGE_SYSTEM_PROMPT = """You check one claim from a generated CV or cover letter against its source text.

Answer with exactly one verdict:
- supported: everything the claim says is stated in, or directly follows from, the source. Rewording, \
shortening, summarising and combining sources are fine.
- stretch: mostly backed, but the claim adds or strengthens something the source doesn't state: scope \
("cross-functional", "every"), frequency ("daily"), method ("tracing a payment through its records"), \
tools, outcomes, seniority, extra detail or colour, or implied knowledge ("still building my knowledge of X" \
when the source shows none).
- unsupported: the claim's main content isn't in the source, or contradicts it.

Special cases:
- Numbers: if the numbers are the same, changing or dropping hedges such as "approximately", "about" or \
"around" is fine.
- Summaries: a statement that just groups or names experience shown in the sources (e.g. "combines customer \
service, administration and design") is supported.
- Understatement is not a stretch: saying less than the source (e.g. "basic design" for professional design \
work) is supported.
- Motivation claims: the only valid source is the candidate's motivation notes. A plain statement that the \
role matches experience shown in the sources counts as supported.
- Gap claims (what the candidate lacks or is keen to learn): supported if the specific named thing (a system, \
tool, procedure, regulation, industry) is absent from the profile, and the wording doesn't imply knowledge the \
profile doesn't show. Related general experience does NOT contradict a gap in a specific named thing. \
"Keen to learn X" about something missing is a gap claim, not motivation.
Judge only what is written; don't reward plausible guesses."""

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


def _feedback_part(feedback: str | None) -> list[str]:
    if not feedback:
        return []
    return [
        "<fact_check_feedback>\nA fact-check of your previous draft found these problems. Write the document again "
        "without them: remove or correct each flagged statement, keep everything else that was accurate, and don't "
        f"add new claims to make up for what you remove.\n{feedback}\n</fact_check_feedback>"
    ]


def generate_cv(
    profile_json: str,
    job_title: str,
    job_description: str | None,
    requirements: JobRequirements | None = None,
    feedback: str | None = None,
) -> str:
    """Returns a Markdown CV tailored to the job, written only from the profile."""
    parts = [
        f"<profile>\n{profile_json}\n</profile>",
        f"<job>\nJob title: {job_title}\n\nJob description:\n"
        f"{job_description or '(none provided; tailor to the job title)'}\n</job>",
    ]
    if requirements is not None:
        parts.append(f"<requirements>\n{requirements.model_dump_json(indent=2)}\n</requirements>")
    parts += _feedback_part(feedback)
    parts.append("Write the tailored CV now.")
    return _complete(CV_SYSTEM_PROMPT, "\n\n".join(parts), "CV")


LETTER_MIN_WORDS, LETTER_MAX_WORDS = 250, 350
MAX_GAPS_TO_MENTION = 2


def assess_gaps(full_profile_json: str, requirements: JobRequirements) -> GapAssessment:
    """Asks the model (with structured output) which requirements the whole profile shows no evidence for.

    The model decides *what* is missing (embedding distance couldn't: on 38 labelled requirements its best
    threshold got 87% right and called real experience "missing", against 95% for this check). Everything
    else (which gaps to mention, the wording) is decided in code, and the result is saved per application.
    """
    def numbered(items: list[str]) -> str:
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)) or "(none)"

    user_message = (
        f"<profile>\n{full_profile_json}\n</profile>\n\n"
        f"<required>\n{numbered(requirements.required_skills)}\n</required>\n\n"
        f"<nice_to_have>\n{numbered(requirements.nice_to_have_skills)}\n</nice_to_have>"
    )
    return _parse(GAP_SYSTEM_PROMPT, user_message, GapAssessment, "gap check")


def _parse(system: str, user_message: str, output_format, what: str):
    """One structured-output request; returns the validated object."""
    with _friendly_errors():
        try:
            response = _client().messages.parse(
                model=settings.ai_model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                output_format=output_format,
            )
        except ValidationError:
            raise AIError(f"The AI returned the {what} in an unexpected shape. Try again.")
    _check_finished(response)
    if response.parsed_output is None:
        raise AIError(f"The AI didn't return the {what}. Try again.")
    return response.parsed_output


def extract_claims(document: str, sources_text: str) -> ClaimList:
    """Splits a document into claims, each tagged with the IDs of the sources it came from."""
    return _parse(CLAIMS_SYSTEM_PROMPT,
                  f"<sources>\n{sources_text}\n</sources>\n\n<document>\n{document}\n</document>",
                  ClaimList, "claim list")


def judge_claim(claim: str, kind: str, source_text: str) -> ClaimVerdict:
    """Judges one claim against its source text: supported, stretch or unsupported.

    (Sampling settings such as temperature are rejected by current models, so consistency comes from the
    structured output with three allowed verdicts and the strict definitions in the prompt.)
    """
    return _parse(JUDGE_SYSTEM_PROMPT,
                  f"<source>\n{source_text or '(no source)'}\n</source>\n\n<claim kind=\"{kind}\">{claim}</claim>",
                  ClaimVerdict, "verdict")


def choose_gaps(assessment: GapAssessment, requirements: JobRequirements) -> list[str]:
    """At most two missing required items, in the order the ad lists them; if nothing required is missing, at
    most one nice-to-have. Ad order (not the model's ranking) keeps the choice the same on every run."""
    def valid(items: list, count: int) -> list:
        seen, out = set(), []
        for m in sorted(items, key=lambda m: m.index):
            if 1 <= m.index <= count and m.index not in seen and m.label.strip():
                seen.add(m.index)
                out.append(m.label.strip())
        return out

    required = valid(assessment.missing_required, len(requirements.required_skills))
    if required:
        return required[:MAX_GAPS_TO_MENTION]
    return valid(assessment.missing_nice_to_have, len(requirements.nice_to_have_skills))[:1]


def gap_sentence(gaps: list[str]) -> str | None:
    """The only wording a letter may use for gaps, so it can't drift into "still building my knowledge of X"."""
    if not gaps:
        return None
    return f"I have not yet worked with {' or '.join(gaps)}, but I am keen to learn."


def ensure_sentence(letter: str, sentence: str) -> str:
    """If the model left out the required sentence, adds it to the end of the second-to-last body paragraph."""
    if sentence in letter:
        return letter
    lines = letter.strip().split("\n")
    sign_off = next((i for i, l in enumerate(lines) if re.match(r"\s*(kind regards|best regards|regards|sincerely)\b",
                                                                 l, re.I)), len(lines))
    start = 1 if lines and lines[0].lower().startswith("dear") else 0
    paragraphs = [i for i in range(start, sign_off) if lines[i].strip() and (i == start or not lines[i - 1].strip())]
    if not paragraphs:
        return letter
    target = paragraphs[-2] if len(paragraphs) >= 2 else paragraphs[-1]
    end = target
    while end + 1 < sign_off and lines[end + 1].strip():
        end += 1
    lines[end] = lines[end].rstrip() + " " + sentence
    return "\n".join(lines)


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


def generate_cover_letter(context, company: str | None, motivation_notes: str | None,
                          feedback: str | None = None) -> str:
    """Returns a plain-text cover letter for the job, built from the same context as the CV.

    `context` is a GenerationContext; context.gaps holds the (saved) gaps to mention. Motivation comes only from
    `motivation_notes`; without notes the letter keeps its motivation general. Enforced in code because the
    model doesn't follow them reliably: the gap sentence (fixed wording, always present when there are gaps),
    the length (one rewrite if it's outside 250-350 words), and the greeting.
    """
    sentence = gap_sentence(getattr(context, "gaps", []) or [])

    parts = [
        f"<profile>\n{context.profile_json}\n</profile>",
        f"<job>\nJob title: {context.job_title}\nCompany: {company or '(not given)'}\n\nJob description:\n"
        f"{context.job_description or '(none provided)'}\n</job>",
    ]
    if context.requirements is not None:
        parts.append(f"<requirements>\n{context.requirements.model_dump_json(indent=2)}\n</requirements>")
    parts.append(f"<gap_sentence>{sentence}</gap_sentence>" if sentence
                 else "<gap_sentence>(none: don't mention gaps)</gap_sentence>")
    parts.append(
        f"<motivation_notes>\n{motivation_notes}\n</motivation_notes>" if motivation_notes
        else "<motivation_notes>(none given: keep the motivation general, as the rules say)</motivation_notes>"
    )
    parts.append(f"<greeting>{letter_greeting(company)}</greeting>")
    parts += _feedback_part(feedback)
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

    letter = set_greeting(letter, company)
    return ensure_sentence(letter, sentence) if sentence else letter


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
