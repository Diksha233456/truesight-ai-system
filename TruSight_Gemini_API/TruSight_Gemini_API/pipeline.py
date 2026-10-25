"""Compatibility functions backed by the hosted Gemini API.

The current FastAPI app imports predict.py and explanation.py directly. These
functions preserve the old pipeline.py entry points for any older callers,
without retaining the previous regex-only conflict detector.
"""
import json
from google.genai import types

from gemini_client import get_client, TEXT_MODEL
from explanation import generate_explanation as _generate_explanation


def find_conflicts(interview, report):
    """Return only explicit factual conflicts found in supplied evidence."""
    if isinstance(report, list):
        evidence_items = [
            {"page": item.get("page"), "text": item.get("text", "")}
            for item in report
        ]
    else:
        evidence_items = [{"page": None, "text": str(report or "")}]

    if not interview or not any(item["text"] for item in evidence_items):
        return []

    prompt = f"""
Identify explicit factual conflicts between the interview statement and the supplied report excerpts.
Do not infer a conflict from missing information, differences in wording, or unrelated facts.
Return JSON only as {{"conflicting_elements": [{{"type":"time/date/location/person/action/object/quantity/detail", "interview":"...", "evidence":"...", "page": 1}}]}}.
If no specific explicit conflict is visible, return an empty array. Never invent a page number.

INTERVIEW STATEMENT:
{interview}

REPORT EXCERPTS:
{json.dumps(evidence_items, ensure_ascii=False)}
"""
    response = get_client().models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        parsed = json.loads(response.text or "{}")
    except json.JSONDecodeError:
        return []
    conflicts = parsed.get("conflicting_elements", []) if isinstance(parsed, dict) else []
    return conflicts if isinstance(conflicts, list) else []


def generate_explanation(report, interview, result):
    """Generate a Gemini explanation while preserving the supplied result."""
    return _generate_explanation(report, interview, result)


def analyze_statement(interview, chunks, index, retrieve_fn):
    """Retrieve relevant evidence, classify the relationship, and explain it."""
    from predict import predict

    evidence = retrieve_fn(interview, chunks, index)
    result = predict(evidence, interview)
    explanation = _generate_explanation(evidence, interview, result)
    return {
        "transcript": interview,
        "evidence": evidence,
        "verdict": result.get("verdict", "Unknown"),
        "confidence": result.get("confidence", 0),
        "score_type": result.get(
            "score_type",
            "model confidence in evidence relationship, not lie probability",
        ),
        "explanation": explanation,
    }
