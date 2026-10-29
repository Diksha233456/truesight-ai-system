"""Evidence comparison using Groq; returns a demo-facing inconsistency score."""
import json
from groq_client import groq_chat_json

VALID_LABELS = {"Entailment", "Neutral", "Contradiction"}
SCORE_TYPE = (
    "Estimated inconsistency score based on explicit contradictions in retrieved evidence; "
    "not a validated probability of deception."
)


def predict(retrieved_chunks, interview):
    if not interview:
        return {"verdict": "No Statement", "confidence": 0, "deception_score": 0,
                "inconsistency_index": 0.0, "evidence_results": [], "score_type": SCORE_TYPE}
    if not retrieved_chunks:
        return {"verdict": "No Evidence", "confidence": 0, "deception_score": 0,
                "inconsistency_index": 0.0, "evidence_results": [], "score_type": "Insufficient retrieved evidence to calculate an inconsistency score."}

    evidence_results = []
    for chunk in retrieved_chunks:
        evidence = str(chunk.get("text", ""))
        if not evidence.strip():
            continue
        prompt = f"""
Compare the interview statement with ONE case-report excerpt. Classify their textual relationship only, not the speaker's honesty.
Entailment means the excerpt supports the statement. Contradiction means an explicit conflict on a concrete fact. Neutral means neither is established.
Do not infer contradiction from missing information, wording differences alone, uncertainty, or unrelated details.
Return JSON with keys verdict (Entailment|Neutral|Contradiction), confidence (0-100, confidence in this classification), rationale, conflict (empty unless explicit conflict).

INTERVIEW STATEMENT:
{interview}

CASE REPORT EXCERPT (page {chunk.get('page')}):
{evidence}
"""
        parsed = groq_chat_json(prompt)
        label = parsed.get("verdict", "Neutral")
        if label not in VALID_LABELS:
            label = "Neutral"
        try:
            confidence = max(0.0, min(100.0, float(parsed.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence_results.append({
            "evidence": evidence,
            "page": chunk.get("page"),
            "distance": chunk.get("distance", 0),
            "verdict": label,
            "confidence": round(confidence, 2),
            "rationale": str(parsed.get("rationale", "")),
            "conflict": str(parsed.get("conflict", "")),
        })

    if not evidence_results:
        return {"verdict": "No Evidence", "confidence": 0, "deception_score": 0,
                "evidence_results": [], "score_type": "Insufficient retrieved evidence to calculate an inconsistency score."}

    contradictions = [r for r in evidence_results if r["verdict"] == "Contradiction"]
    strongest = max(contradictions or evidence_results, key=lambda item: item["confidence"])
    score = round(sum(item["confidence"] for item in contradictions) / len(evidence_results))
    if contradictions:
        verdict = "High Inconsistency" if score >= 65 else ("Moderate Inconsistency" if score >= 35 else "Possible Inconsistency")
    elif all(item["verdict"] == "Entailment" for item in evidence_results):
        verdict = "Consistent with Evidence"
    else:
        verdict = "No Clear Inconsistency"

    raw_deception_score = max(0, min(100, score))
    inconsistency_index = round(raw_deception_score / 100.0, 4)

    return {"verdict": verdict, "confidence": strongest["confidence"],
            "deception_score": raw_deception_score,
            "inconsistency_index": inconsistency_index,
            "evidence_results": evidence_results, "score_type": SCORE_TYPE}
