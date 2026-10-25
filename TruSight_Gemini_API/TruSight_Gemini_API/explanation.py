"""Generate evidence-grounded explanations through Groq."""
import json
from groq_client import groq_chat_json


def generate_explanation(report, interview, result):
    if isinstance(report, dict):
        report = [report]
    evidence_items = [
        {"page": item.get("page"), "text": item.get("text", ""), "distance": item.get("distance")}
        for item in (report or [])
    ]
    prompt = f"""
Write a concise, evidence-grounded explanation for a transcript-vs-case-report comparison.
Preserve the supplied verdict, confidence, and inconsistency score. Do not claim to prove lies, guilt, intent, or credibility. State only whether the statement is supported, contradicted, or unresolved by the supplied excerpts.
Only include conflicting elements when an explicit factual conflict is visible. Never invent details or page numbers. If no explicit conflict is visible, use an empty conflicting_elements array.
Return JSON with exactly: summary, interview_statement, evidence_statement, conflicting_elements.
Each conflicting_elements item must have type (time/date/location/person/action/object/quantity/detail), interview, evidence, page.

INTERVIEW STATEMENT:
{interview}

ANALYSIS RESULT (preserve it):
{json.dumps(result, ensure_ascii=False)}

RETRIEVED CASE EVIDENCE:
{json.dumps(evidence_items, ensure_ascii=False)}
"""
    parsed = groq_chat_json(prompt)
    if not isinstance(parsed, dict):
        parsed = {"summary": str(parsed), "conflicting_elements": []}
    parsed.setdefault("summary", "No summary returned.")
    parsed.setdefault("interview_statement", interview)
    parsed.setdefault("evidence_statement", "\n\n".join(x["text"] for x in evidence_items))
    if not isinstance(parsed.get("conflicting_elements"), list):
        parsed["conflicting_elements"] = []
    parsed["verdict"] = result.get("verdict", "Unknown")
    parsed["confidence"] = result.get("confidence", 0)
    parsed["deception_score"] = result.get("deception_score", 0)
    parsed["score_type"] = result.get("score_type", "Estimated inconsistency score; not a validated probability of deception.")
    return parsed
