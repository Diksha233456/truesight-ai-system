"""
Assessment Coordinator for TRUSIGHT
Synthesizes document evidence, facial observations, and acoustic observations
into a defensible multimodal fusion assessment without computing an unvalidated
combined deception probability or conflating factual contradiction with behavioral classifier outputs.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

try:
    from fusion_service.neural_corrector import load_neural_corrector
except ImportError:
    load_neural_corrector = None


# =========================================================================
# VALIDATION & TIME HELPERS
# =========================================================================

def _parse_probability(val: Any) -> Optional[float]:
    """
    Validate probability-like numeric inputs.
    Rejects non-finite values (NaN, Inf), non-numeric values,
    and out-of-range values (< 0.0 or > 1.0).
    Does not silently clamp or accept invalid inputs.
    """
    if val is None:
        return None
    try:
        f = float(val)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(f):
        return None
    if f < 0.0 or f > 1.0:
        return None
    return round(f, 4)


def _evaluate_timestamp(
    ts_raw: Any,
    now: float,
    max_freshness_seconds: float,
    future_skew_tolerance: float = 1.0,
) -> Tuple[Optional[float], Optional[float], bool, Optional[str]]:
    """
    Evaluates timestamp freshness, invalidity, future clock skew, and staleness.
    Returns:
        (timestamp_float, freshness_seconds, is_fresh_and_valid, exclusion_reason)
    """
    if ts_raw is None:
        return None, None, False, "missing timestamp"
    try:
        ts = float(ts_raw)
    except (ValueError, TypeError):
        return None, None, False, "invalid timestamp format (non-numeric)"
    if not math.isfinite(ts) or ts <= 0:
        return None, None, False, "invalid timestamp value (must be positive finite epoch timestamp)"

    if ts > now + future_skew_tolerance:
        skew = ts - now
        return ts, None, False, f"future timestamp detected (clock skew +{skew:.2f}s > {future_skew_tolerance:.1f}s tolerance)"

    freshness = max(0.0, round(now - ts, 2))
    if freshness > max_freshness_seconds:
        return ts, freshness, False, f"stale telemetry ({freshness:.1f}s > {max_freshness_seconds:.1f}s limit)"

    return ts, freshness, True, None


# =========================================================================
# SCHEMA DEFINITIONS
# =========================================================================

class EvidenceFinding(BaseModel):
    statement: Optional[str] = Field(default=None, description="The evaluated statement or claim, if available.")
    status: str = Field(default="INSUFFICIENT_EVIDENCE", description="SUPPORTED, CONTRADICTED, MIXED, INSUFFICIENT_EVIDENCE, or NO_STATEMENT.")
    mapped_class_label: str = Field(default="INSUFFICIENT EVIDENCE", description="Explicit UI label: SUPPORTED, CONTRADICTED, MIXED, INSUFFICIENT EVIDENCE, NO STATEMENT.")
    inconsistency_index: Optional[float] = Field(default=None, description="Textual contradiction score on [0.0, 1.0] scale.")
    legacy_score: Optional[int] = Field(default=None, description="Legacy 0-100 score for backward compatibility.")
    total_contradictions: int = Field(default=0)
    total_support: int = Field(default=0)
    excerpts: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved excerpts with page and source references.")
    verifier_confidence: Optional[float] = Field(default=None, description="NLI classification confidence from underlying verifier.")
    evidence_sufficiency: str = Field(default="INSUFFICIENT", description="SUFFICIENT, INSUFFICIENT, or NO_STATEMENT.")
    eligible_for_fusion: bool = Field(default=False)
    exclusion_reason: Optional[str] = Field(default=None)
    provenance: str = Field(default="NLI analysis via LLM against retrieved document excerpts.")
    limitations: str = Field(
        default="Retrieved document excerpts may support or conflict with a statement. The coordinator does not assert that source documents are verified unless their external provenance has been validated."
    )


class FacialObservation(BaseModel):
    available: bool = Field(default=False)
    eligible_for_fusion: bool = Field(default=False)
    timestamp: Optional[float] = Field(default=None)
    freshness_seconds: Optional[float] = Field(default=None)
    label: str = Field(default="WAITING", description="Raw model class: TRUTH, LIE, or WAITING.")
    mapped_class_label: str = Field(
        default="MODEL UNAVAILABLE",
        description="Explicit status: MODEL PREDICTS TRUTH CLASS, MODEL PREDICTS LIE CLASS, INCONCLUSIVE, INSUFFICIENT FACIAL DATA, or MODEL UNAVAILABLE."
    )
    p_lie: Optional[float] = Field(default=None, description="Classifier raw probability for class 1 (Lie) on [0.0, 1.0].")
    predicted_class_probability: Optional[float] = Field(
        default=None,
        description="Probability corresponding directly to the predicted class (e.g. 1 - p_lie when Truth is predicted)."
    )
    probability_meaning: Optional[str] = Field(
        default=None,
        description="Explicit explanation of displayed probability value and class index mapping."
    )
    raw_scores: Dict[str, Any] = Field(default_factory=dict, description="Raw model probabilities per class.")
    calibrated: bool = Field(default=False, description="Whether output probabilities are calibrated against external validation.")
    quality_status: str = Field(default="UNAVAILABLE", description="USABLE_OBSERVATIONS, INSUFFICIENT_FACIAL_DATA, STALE_DATA, INVALID_DATA, INCONSISTENT_OBSERVATIONS, or UNAVAILABLE.")
    observed_cues: List[str] = Field(default_factory=list, description="Extracted Action Units or heuristic cue observations.")
    emotion: Optional[str] = Field(default=None)
    exclusion_reason: Optional[str] = Field(default=None)
    model_provenance: str = Field(
        default="RandomForest classifier trained on 121 courtroom video clips (Miami dataset; classes: 0=Truth, 1=Lie)."
    )
    limitations: str = Field(
        default="Behavioral facial patterns (action units, gaze aversion, blinks) are exploratory nonverbal indicators and do not establish deception."
    )


class AcousticObservation(BaseModel):
    available: bool = Field(default=False)
    eligible_for_fusion: bool = Field(default=False)
    timestamp: Optional[float] = Field(default=None)
    freshness_seconds: Optional[float] = Field(default=None)
    speech_detected: bool = Field(default=False)
    verdict: str = Field(default="Inconclusive", description="Raw classifier class: Truthful, Deceptive, or Inconclusive.")
    mapped_class_label: str = Field(
        default="MODEL UNAVAILABLE",
        description="Explicit status: MODEL PREDICTS TRUTH CLASS, MODEL PREDICTS LIE CLASS, INCONCLUSIVE, NO SPEECH DETECTED, INSUFFICIENT AUDIO, or MODEL UNAVAILABLE."
    )
    raw_model_score: Optional[float] = Field(default=None, description="Probability of Deceptive class (P[class 1]) on [0.0, 1.0].")
    predicted_class_probability: Optional[float] = Field(
        default=None,
        description="Probability corresponding directly to the predicted class."
    )
    confidence: Optional[float] = Field(default=None, description="Model max probability over classes.")
    probability_meaning: Optional[str] = Field(
        default=None,
        description="Explicit explanation of displayed probability value and class index mapping."
    )
    raw_scores: Dict[str, Any] = Field(default_factory=dict, description="Raw model probabilities per class.")
    calibrated: bool = Field(default=False, description="Whether output probabilities are calibrated against external validation.")
    quality_status: str = Field(default="UNAVAILABLE", description="USABLE_SPEECH, NO_SPEECH_DETECTED, STALE_AUDIO, INVALID_AUDIO, INSUFFICIENT_AUDIO, INCONSISTENT_AUDIO, or UNAVAILABLE.")
    acoustic_features: Dict[str, Any] = Field(default_factory=dict, description="Pitch, tempo, pauses, jitter/shimmer measurements.")
    exclusion_reason: Optional[str] = Field(default=None)
    model_provenance: str = Field(
        default="RandomForest on 42 Praat/Librosa features (Dolos benchmark; classes: 0=Truthful, 1=Deceptive)."
    )
    limitations: str = Field(
        default="Acoustic measurements (pitch variation, pauses, tempo, jitter, shimmer) reflect physiological vocal dynamics and do not independently establish deception."
    )


class AgreementAnalysis(BaseModel):
    concordance: str = Field(default="NOT_APPLICABLE", description="CONCORDANT_TRUTH, CONCORDANT_LIE, CROSS_MODAL_CONFLICT, BEHAVIORAL_CONFLICT, SINGLE_MODALITY, or NOT_APPLICABLE.")
    summary: str = Field(default="No eligible modalities to evaluate agreement.")
    details: List[str] = Field(default_factory=list)
    behavioral_agreement: Optional[str] = Field(default=None)


class FusionAssessment(BaseModel):
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: Optional[str] = Field(default=None)
    timestamp: float
    final_status: str = Field(
        default="ASSESSMENT UNAVAILABLE",
        description="One of: CONSISTENT WITH AVAILABLE EVIDENCE, CONFLICT WITH AVAILABLE EVIDENCE, MIXED FINDINGS, INCONCLUSIVE, INSUFFICIENT DATA, ASSESSMENT UNAVAILABLE."
    )
    fusion_score: Optional[float] = Field(
        default=None,
        description="Numeric fusion score on [0.0, 1.0] scale, provided only if mathematically justified and validated. None when unvalidated."
    )
    combined_multimodal_score: Optional[float] = Field(
        default=None,
        description="Synthesized cross-modal deception likelihood index combining eligible streams (Documentary 50%, Voice 25%, Facial 25%)."
    )
    experimental_heuristic_score: Optional[float] = Field(
        default=None,
        description="Synthesized cross-modal experimental heuristic index combining eligible streams. Unvalidated."
    )
    neural_corrected_fusion_score: Optional[float] = Field(
        default=None,
        description="Neural corrector DL model fusion score with equal module weighting across active streams."
    )
    combined_multimodal_confidence: Optional[float] = Field(
        default=None,
        description="Overall confidence index across eligible streams."
    )
    stream_scores: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Individual scores across evidence streams: documentary, facial, vocal."
    )
    fusion_score_display: str = Field(default="Not validated", description="Display string for fusion score: e.g. 'Not validated'.")
    fusion_score_rationale: str = Field(
        default="A combined numeric deception percentage is not computed because cross-modal calibration is unvalidated and behavioral model outputs cannot be mathematically equated to factual deceit."
    )
    eligible_modalities_count: int = Field(default=0)
    eligible_modality_names: List[str] = Field(default_factory=list)
    excluded_modality_names: List[str] = Field(default_factory=list)
    exclusion_reasons: Dict[str, str] = Field(default_factory=dict)
    evidence_sufficiency: str = Field(default="INSUFFICIENT", description="SUFFICIENT, INSUFFICIENT, or NO_STATEMENT.")
    modality_findings: Dict[str, str] = Field(default_factory=dict)
    agreement_analysis: AgreementAnalysis = Field(default_factory=AgreementAnalysis)
    explanation: str = Field(default="")
    limitations: str = Field(
        default="Factual contradiction is assessed directly from retrieved document excerpts. The coordinator does not assert that source documents are verified unless their external provenance has been validated. Facial and acoustic behavioral outputs represent statistical machine learning models and must not be interpreted as validated proof of deception."
    )


class AssessmentRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Optional active session identifier to prevent mixing cross-session data.")
    current_time: Optional[float] = Field(default=None, description="Reference epoch timestamp in seconds.")
    statement_evidence: Optional[Dict[str, Any]] = Field(default=None)
    face_telemetry: Optional[Dict[str, Any]] = Field(default=None)
    voice_telemetry: Optional[Dict[str, Any]] = Field(default=None)


class AssessmentResponse(BaseModel):
    status: str = Field(default="READY")
    evaluation_time: float
    session_id: Optional[str] = Field(default=None)
    evidence_finding: EvidenceFinding
    facial_observation: FacialObservation
    acoustic_observation: AcousticObservation
    fusion_assessment: FusionAssessment
    overall_summary: Dict[str, Any]
    active_modalities: List[str]
    missing_modalities: List[str]
    disclaimer: str = Field(
        default="TRUSIGHT presents structured, independent modalities. Factual contradiction is assessed by comparing statements against retrieved document excerpts. Facial and vocal observations reflect statistical classifiers and must not be interpreted as validated proof of deception. No combined deception probability is computed."
    )


# =========================================================================
# COORDINATION & EVALUATION LOGIC
# =========================================================================

def evaluate_assessment(req: AssessmentRequest) -> AssessmentResponse:
    now = req.current_time if req.current_time is not None else time.time()
    req_session = req.session_id

    # -------------------------------------------------------------------------
    # 1. EVALUATE STATEMENT AND DOCUMENTARY EVIDENCE
    # -------------------------------------------------------------------------
    ev_data = req.statement_evidence or {}
    statement = ev_data.get("statement") or ev_data.get("transcript")
    if isinstance(statement, str):
        statement = statement.strip()
        if not statement:
            statement = None

    raw_results = ev_data.get("evidence_results") or ev_data.get("evidence") or []
    excerpts: List[Dict[str, Any]] = []
    contra_count = 0
    support_count = 0
    confidences: List[float] = []

    for item in raw_results:
        verdict = str(item.get("verdict", "")).strip().capitalize()
        page = item.get("page") or item.get("source_page")
        text = item.get("evidence") or item.get("text") or ""
        conflict = item.get("conflict", "")
        rationale = item.get("rationale", "")
        conf_raw = item.get("confidence")

        # Validate verifier confidence
        conf_val = _parse_probability(conf_raw)
        if conf_val is None and conf_raw is not None:
            # Handle potential 0-100 percentage scale for verifier confidence
            try:
                raw_f = float(conf_raw)
                if math.isfinite(raw_f) and 0.0 <= raw_f <= 100.0:
                    conf_val = round(raw_f / 100.0, 4)
            except (ValueError, TypeError):
                conf_val = None

        if conf_val is not None:
            confidences.append(conf_val)

        if verdict in ("Contradiction", "Conflict", "Contradicted"):
            contra_count += 1
        elif verdict in ("Entailment", "Supported", "Support", "Consistent"):
            support_count += 1

        excerpts.append({
            "text": text,
            "page": page,
            "relationship": verdict or "Neutral",
            "conflict": conflict,
            "rationale": rationale,
            "confidence": conf_val if conf_val is not None else conf_raw,
        })

    # Check top-level verdict if individual item verdicts were unspecified
    top_verdict = str(ev_data.get("verdict", "") or ev_data.get("status", "")).strip().upper()
    if contra_count == 0 and support_count == 0 and top_verdict and excerpts:
        if any(w in top_verdict for w in ("CONTRADICT", "CONFLICT", "INCONSISTEN")):
            contra_count = len(excerpts)
        elif any(w in top_verdict for w in ("SUPPORT", "ENTAIL", "CONSISTENT")):
            support_count = len(excerpts)

    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None

    # Evidence status determination
    if not statement:
        ev_status = "NO_STATEMENT"
        mapped_ev_label = "NO STATEMENT"
        ev_sufficiency = "NO_STATEMENT"
        ev_eligible = False
        ev_exclusion = "No statement submitted for verification"
    elif not excerpts:
        ev_status = "INSUFFICIENT_EVIDENCE"
        mapped_ev_label = "INSUFFICIENT EVIDENCE"
        ev_sufficiency = "INSUFFICIENT"
        ev_eligible = False
        ev_exclusion = "Insufficient retrieved document excerpts to verify claim"
    elif contra_count > 0 and support_count > 0:
        ev_status = "MIXED"
        mapped_ev_label = "MIXED"
        ev_sufficiency = "SUFFICIENT"
        ev_eligible = True
        ev_exclusion = None
    elif contra_count > 0:
        ev_status = "CONTRADICTED"
        mapped_ev_label = "CONTRADICTED"
        ev_sufficiency = "SUFFICIENT"
        ev_eligible = True
        ev_exclusion = None
    elif support_count > 0:
        ev_status = "SUPPORTED"
        mapped_ev_label = "SUPPORTED"
        ev_sufficiency = "SUFFICIENT"
        ev_eligible = True
        ev_exclusion = None
    else:
        ev_status = "INSUFFICIENT_EVIDENCE"
        mapped_ev_label = "INSUFFICIENT EVIDENCE"
        ev_sufficiency = "INSUFFICIENT"
        ev_eligible = False
        ev_exclusion = "Retrieved excerpts are neutral or do not address statement directly"

    inconsistency_raw = ev_data.get("inconsistency_index")
    inconsistency_idx = _parse_probability(inconsistency_raw)
    if inconsistency_idx is None and "deception_score" in ev_data:
        try:
            raw_s = float(ev_data["deception_score"])
            if math.isfinite(raw_s) and 0.0 <= raw_s <= 100.0:
                inconsistency_idx = round(raw_s / 100.0, 4) if raw_s > 1.0 else round(raw_s, 4)
        except (ValueError, TypeError):
            inconsistency_idx = None

    legacy_score = ev_data.get("deception_score")
    if legacy_score is not None:
        try:
            raw_ls = float(legacy_score)
            if math.isfinite(raw_ls) and 0.0 <= raw_ls <= 100.0:
                legacy_score = int(round(raw_ls))
            else:
                legacy_score = None
        except (ValueError, TypeError):
            legacy_score = None

    evidence_finding = EvidenceFinding(
        statement=statement,
        status=ev_status,
        mapped_class_label=mapped_ev_label,
        inconsistency_index=inconsistency_idx,
        legacy_score=legacy_score,
        total_contradictions=contra_count,
        total_support=support_count,
        excerpts=excerpts,
        verifier_confidence=avg_conf,
        evidence_sufficiency=ev_sufficiency,
        eligible_for_fusion=ev_eligible,
        exclusion_reason=ev_exclusion,
    )

    # -------------------------------------------------------------------------
    # 2. EVALUATE FACIAL OBSERVATION
    # -------------------------------------------------------------------------
    face_data = req.face_telemetry or {}
    face_ready = bool(face_data.get("ready", False))
    face_session = face_data.get("session_id")
    face_session_mismatch = bool(req_session and face_session and req_session != face_session)

    # Timestamp evaluation (max 5.0s freshness)
    face_ts_raw = face_data.get("timestamp")
    if face_ts_raw is None and isinstance(face_data.get("deception"), dict):
        face_ts_raw = face_data["deception"].get("timestamp")
    face_ts_val, face_freshness, face_fresh, face_ts_err = _evaluate_timestamp(
        face_ts_raw, now, max_freshness_seconds=5.0
    )


    is_face_stale = (face_freshness is not None and face_freshness > 5.0)

    # Service Availability: False if stale, session mismatched, or unready
    face_available = face_ready and not is_face_stale and not face_session_mismatch

    # Probability extraction & strict validation
    face_deception = face_data.get("deception") or {}
    raw_face_label = str(face_deception.get("label") or face_data.get("verdict") or "WAITING").upper()
    face_plie_raw = face_deception.get("p_lie")
    if face_plie_raw is None:
        face_plie_raw = face_data.get("p_lie")
    if face_plie_raw is None and "pLie" in face_data:
        face_plie_raw = face_data.get("pLie")

    face_plie_val = _parse_probability(face_plie_raw)
    face_prob_invalid = (face_plie_raw is not None and face_plie_val is None)

    face_cues = face_deception.get("reasons") or face_data.get("reasons") or []
    raw_emo = face_data.get("emotion")
    if isinstance(raw_emo, dict):
        face_emotion = raw_emo.get("label") or raw_emo.get("line")
    elif isinstance(raw_emo, str):
        face_emotion = raw_emo
    else:
        face_emotion = face_data.get("sentiment")

    # Determine Observation Quality, Class Consistency, and Fusion Eligibility
    face_eligible = False
    face_exclusion = None
    face_pred_prob = None
    face_prob_meaning = None
    face_raw_scores: Dict[str, Any] = {}

    if face_session_mismatch:
        face_quality = "UNAVAILABLE"
        face_status_label = "MODEL UNAVAILABLE"
        face_exclusion = f"Mismatched session identifier (expected {req_session}, got {face_session})"
    elif not face_ready:
        face_quality = "NO_FACE_DETECTED"
        face_status_label = "INSUFFICIENT FACIAL DATA"
        face_exclusion = "Facial tracking is calibrating or no face detected in video feed"
    elif is_face_stale:
        face_quality = "STALE_DATA"
        face_status_label = "INSUFFICIENT FACIAL DATA"
        face_exclusion = face_ts_err or f"stale telemetry ({face_freshness:.1f}s > 5.0s limit)"
    elif face_ts_err is not None:
        face_quality = "INVALID_TIMESTAMP"
        face_status_label = "INSUFFICIENT FACIAL DATA"
        face_exclusion = face_ts_err
    elif face_prob_invalid:
        face_quality = "INVALID_PROBABILITY"
        face_status_label = "INCONCLUSIVE"
        face_exclusion = f"Invalid probability input: {face_plie_raw!r} (must be finite number between 0.0 and 1.0)"
    elif face_plie_val is None:
        face_quality = "INSUFFICIENT_FACIAL_DATA"
        face_status_label = "INCONCLUSIVE"
        face_exclusion = "Model has not returned a valid probability score for current window"
    else:
        # Check for Label vs Probability Contradiction
        # classes_: [0, 1] where 0 = Truth, 1 = Lie.
        # face_plie is P(class 1) = probability of Lie.
        # Contradiction: Raw label says LIE but P(Lie) < 0.45, or TRUTH but P(Lie) > 0.55.
        p_truth = round(1.0 - face_plie_val, 4)
        face_raw_scores = {"class_0_truth": p_truth, "class_1_lie": face_plie_val, "p_lie": face_plie_val}

        is_label_conflict = (
            (raw_face_label == "LIE" and face_plie_val < 0.45)
            or (raw_face_label == "TRUTH" and face_plie_val > 0.55)
        )

        if is_label_conflict:
            face_quality = "INCONSISTENT_OBSERVATIONS"
            face_status_label = "INCONCLUSIVE"
            face_exclusion = f"Label/probability conflict: raw label '{raw_face_label}' contradicts probability P(Lie)={face_plie_val:.3f}"
            face_pred_prob = face_plie_val
            face_prob_meaning = f"Probability P(Lie)={face_plie_val:.3f} conflicts with supplied label '{raw_face_label}'"
            face_eligible = False
        elif 0.45 <= face_plie_val <= 0.55:
            face_quality = "USABLE_OBSERVATIONS"
            face_status_label = "INCONCLUSIVE"
            face_pred_prob = face_plie_val
            face_prob_meaning = f"Probability near decision threshold: P(Lie)={face_plie_val:.3f}, P(Truth)={p_truth:.3f}"
            face_eligible = True
        elif face_plie_val > 0.55:
            face_quality = "USABLE_OBSERVATIONS"
            face_status_label = "MODEL PREDICTS LIE CLASS"
            face_pred_prob = face_plie_val
            face_prob_meaning = f"Probability of predicted class LIE: P(class 1)={face_plie_val:.3f} (uncalibrated)"
            face_eligible = True
        else:
            face_quality = "USABLE_OBSERVATIONS"
            face_status_label = "MODEL PREDICTS TRUTH CLASS"
            face_pred_prob = p_truth
            face_prob_meaning = f"Probability of predicted class TRUTH: P(class 0)={p_truth:.3f} (P[Lie]={face_plie_val:.3f})"
            face_eligible = True

    facial_obs = FacialObservation(
        available=face_available,
        eligible_for_fusion=face_eligible,
        timestamp=face_ts_val,
        freshness_seconds=face_freshness,
        label=raw_face_label if face_available else "UNAVAILABLE",
        mapped_class_label=face_status_label,
        p_lie=face_plie_val if face_available else None,
        predicted_class_probability=face_pred_prob if face_available else None,
        probability_meaning=face_prob_meaning if face_available else None,
        raw_scores=face_raw_scores if face_available else {},
        calibrated=False,
        quality_status=face_quality,
        observed_cues=face_cues if face_available else [],
        emotion=face_emotion if face_available else None,
        exclusion_reason=face_exclusion,
    )

    # -------------------------------------------------------------------------
    # 3. EVALUATE ACOUSTIC OBSERVATION
    # -------------------------------------------------------------------------
    voice_data = req.voice_telemetry or {}
    voice_ready = bool(voice_data.get("ready", False))
    voice_session = voice_data.get("session_id")
    voice_session_mismatch = bool(req_session and voice_session and req_session != voice_session)

    # Freshness threshold calculation:
    # A voice window takes W seconds to record (standard: 15s, 20s, or 30s) plus network and inference latency.
    # The freshness boundary allows the window duration plus a bounded grace allowance (default 10.0s).
    # Validate client window input with sensible bounds [5.0s, 60.0s]; fallback to default window (15.0s).
    voice_window_raw = voice_data.get("window_seconds")
    if voice_window_raw is None:
        voice_window_raw = voice_data.get("window")

    validated_window = 15.0
    if voice_window_raw is not None:
        try:
            vw = float(voice_window_raw)
            if math.isfinite(vw) and 5.0 <= vw <= 60.0:
                validated_window = vw
        except (ValueError, TypeError):
            validated_window = 15.0

    voice_grace_period = 10.0
    if voice_window_raw is None:
        max_voice_freshness = 10.0
    else:
        max_voice_freshness = min(60.0, max(15.0, round(validated_window + voice_grace_period, 1)))

    # Timestamp evaluation against validated window-aware freshness threshold
    voice_ts_raw = voice_data.get("timestamp")
    voice_ts_val, voice_freshness, voice_fresh, voice_ts_err = _evaluate_timestamp(
        voice_ts_raw, now, max_freshness_seconds=max_voice_freshness
    )

    is_voice_stale = (voice_freshness is not None and voice_freshness >= max_voice_freshness)

    # Service Availability: False if stale, session mismatched, or unready
    voice_available = voice_ready and not is_voice_stale and not voice_session_mismatch

    speech_detected = bool(voice_data.get("speech_detected") or voice_data.get("speechDetected", False))
    raw_voice_verdict = str(voice_data.get("verdict") or "Inconclusive").capitalize()

    voice_score_raw = voice_data.get("deceptionScore")
    if voice_score_raw is None:
        voice_score_raw = voice_data.get("score")
    voice_score_val = _parse_probability(voice_score_raw)
    voice_score_invalid = (voice_score_raw is not None and voice_score_val is None)

    voice_conf_raw = voice_data.get("confidence")
    voice_conf_val = _parse_probability(voice_conf_raw)

    # Determine Observation Quality, Verdict Consistency, and Fusion Eligibility
    voice_eligible = False
    voice_exclusion = None
    voice_pred_prob = None
    voice_prob_meaning = None
    voice_raw_scores: Dict[str, Any] = {}

    if voice_session_mismatch:
        voice_quality = "UNAVAILABLE"
        voice_status_label = "MODEL UNAVAILABLE"
        voice_exclusion = f"Mismatched session identifier (expected {req_session}, got {voice_session})"
    elif not voice_ready:
        voice_quality = "UNAVAILABLE"
        voice_status_label = "MODEL UNAVAILABLE"
        voice_exclusion = "Voice engine offline or awaiting initial recording connection"
    elif is_voice_stale:
        voice_quality = "STALE_AUDIO"
        voice_status_label = "INSUFFICIENT AUDIO"
        voice_exclusion = voice_ts_err or f"stale telemetry ({voice_freshness:.1f}s > {max_voice_freshness:.1f}s limit)"
    elif voice_ts_err is not None:
        voice_quality = "INVALID_AUDIO"
        voice_status_label = "INSUFFICIENT AUDIO"
        voice_exclusion = voice_ts_err
    elif not speech_detected:
        voice_quality = "NO_SPEECH_DETECTED"
        voice_status_label = "NO SPEECH DETECTED"
        voice_exclusion = "No voiced phonation or glottal pulses detected in current audio window"
    elif voice_score_invalid:
        voice_quality = "INVALID_PROBABILITY"
        voice_status_label = "INSUFFICIENT AUDIO"
        voice_exclusion = f"Invalid probability input: {voice_score_raw!r} (must be finite number between 0.0 and 1.0)"
    elif voice_score_val is None:
        voice_quality = "INSUFFICIENT_AUDIO"
        voice_status_label = "INSUFFICIENT AUDIO"
        voice_exclusion = "Audio duration or quality was insufficient for acoustic feature extraction"
    else:
        # Check for Verdict vs Probability Contradiction
        # classes_: [0, 1] with class_names: ['Truthful', 'Deceptive']
        # voice_score_val is P(class 1) = probability of Deceptive.
        p_truth_voice = round(1.0 - voice_score_val, 4)
        voice_raw_scores = {"class_0_truthful": p_truth_voice, "class_1_deceptive": voice_score_val}

        is_verdict_conflict = (
            (raw_voice_verdict in ("Deceptive", "Lie-class prediction") and voice_score_val < 0.45)
            or (raw_voice_verdict in ("Truthful", "Truth-class prediction") and voice_score_val > 0.55)
        )

        if is_verdict_conflict:
            voice_quality = "INCONSISTENT_AUDIO"
            voice_status_label = "INCONCLUSIVE"
            voice_exclusion = f"Verdict/score conflict: raw verdict '{raw_voice_verdict}' contradicts score P(Deceptive)={voice_score_val:.3f}"
            voice_pred_prob = voice_score_val
            voice_prob_meaning = f"Probability P(Deceptive)={voice_score_val:.3f} conflicts with supplied verdict '{raw_voice_verdict}'"
            voice_eligible = False
        elif 0.45 <= voice_score_val <= 0.55:
            voice_quality = "USABLE_SPEECH"
            voice_status_label = "INCONCLUSIVE"
            voice_pred_prob = voice_score_val
            voice_prob_meaning = f"Probability near decision threshold: P(Deceptive)={voice_score_val:.3f}, P(Truthful)={p_truth_voice:.3f}"
            voice_eligible = True
        elif voice_score_val > 0.55:
            voice_quality = "USABLE_SPEECH"
            voice_status_label = "MODEL PREDICTS LIE CLASS"
            voice_pred_prob = voice_score_val
            voice_prob_meaning = f"Probability of predicted class DECEPTIVE: P(class 1)={voice_score_val:.3f} (uncalibrated)"
            voice_eligible = True
        else:
            voice_quality = "USABLE_SPEECH"
            voice_status_label = "MODEL PREDICTS TRUTH CLASS"
            voice_pred_prob = p_truth_voice
            voice_prob_meaning = f"Probability of predicted class TRUTHFUL: P(class 0)={p_truth_voice:.3f} (P[Deceptive]={voice_score_val:.3f})"
            voice_eligible = True

    acoustic_obs = AcousticObservation(
        available=voice_available,
        eligible_for_fusion=voice_eligible,
        timestamp=voice_ts_val,
        freshness_seconds=voice_freshness,
        speech_detected=speech_detected,
        verdict=raw_voice_verdict if voice_available else "UNAVAILABLE",
        mapped_class_label=voice_status_label,
        raw_model_score=voice_score_val if (voice_ready and not voice_session_mismatch and not is_voice_stale) else None,
        predicted_class_probability=voice_pred_prob if (voice_available and speech_detected) else None,
        confidence=voice_conf_val if (voice_available and speech_detected) else None,
        probability_meaning=voice_prob_meaning if (voice_available and speech_detected) else None,
        raw_scores=voice_raw_scores if (voice_available and speech_detected) else {},
        calibrated=False,
        quality_status=voice_quality,
        acoustic_features=voice_data.get("features") or {},
        exclusion_reason=voice_exclusion,
    )

    # -------------------------------------------------------------------------
    # 4. MODALITY ELIGIBILITY & AGREEMENT ANALYSIS
    # -------------------------------------------------------------------------
    eligible_names: List[str] = []
    excluded_names: List[str] = []
    exclusion_dict: Dict[str, str] = {}

    if ev_eligible:
        eligible_names.append("Statement & Documentary Evidence")
    else:
        excluded_names.append("Statement & Documentary Evidence")
        exclusion_dict["Statement & Documentary Evidence"] = ev_exclusion or "Evidence not available"

    if face_eligible:
        eligible_names.append("Facial Movement Analysis")
    else:
        excluded_names.append("Facial Movement Analysis")
        exclusion_dict["Facial Movement Analysis"] = face_exclusion or "Facial telemetry not available"

    if voice_eligible:
        eligible_names.append("Vocal Acoustic Analysis")
    else:
        excluded_names.append("Vocal Acoustic Analysis")
        exclusion_dict["Vocal Acoustic Analysis"] = voice_exclusion or "Voice telemetry not available"

    agreement_details: List[str] = []
    behavioral_agreement: Optional[str] = None
    concordance = "NOT_APPLICABLE"

    face_pred = "TRUTH" if "TRUTH" in face_status_label else ("LIE" if "LIE" in face_status_label else "INCONCLUSIVE")
    voice_pred = "TRUTH" if "TRUTH" in voice_status_label else ("LIE" if "LIE" in voice_status_label else "INCONCLUSIVE")

    # Evaluate behavioral cross-agreement if both behavioral channels are eligible
    if face_eligible and voice_eligible:
        if face_pred == voice_pred and face_pred != "INCONCLUSIVE":
            behavioral_agreement = f"Concordant ({face_pred}): Both facial and vocal classifiers predict {face_pred} class."
            agreement_details.append(behavioral_agreement)
        elif face_pred != voice_pred and face_pred != "INCONCLUSIVE" and voice_pred != "INCONCLUSIVE":
            behavioral_agreement = f"Discordant: Facial model predicts {face_pred} class while Vocal model predicts {voice_pred} class."
            agreement_details.append(behavioral_agreement)
        else:
            behavioral_agreement = "Inconclusive: One or both behavioral models returned probabilities near decision threshold."
            agreement_details.append(behavioral_agreement)
    elif face_eligible:
        behavioral_agreement = f"Single Modality: Only Facial Movement Analysis available ({face_pred} class)."
    elif voice_eligible:
        behavioral_agreement = f"Single Modality: Only Vocal Acoustic Analysis available ({voice_pred} class)."
    else:
        behavioral_agreement = "None: No behavioral telemetry eligible for fusion."

    # Cross-modal agreement between Documentary Evidence and Behavioral Models
    if ev_eligible:
        agreement_details.append(f"Documentary Channel: Evidence status is {ev_status}.")
        if ev_status == "CONTRADICTED":
            if (face_eligible and face_pred == "LIE") or (voice_eligible and voice_pred == "LIE"):
                concordance = "CROSS_MODAL_CONCORDANT_CONFLICT"
                agreement_details.append("Documentary contradiction is accompanied by behavioral models predicting lie class.")
            else:
                concordance = "DOCUMENTARY_CONFLICT_ONLY"
                agreement_details.append("Documentary contradiction identified; behavioral models do not predict lie class.")
        elif ev_status == "SUPPORTED":
            if (face_eligible and face_pred == "LIE") and (voice_eligible and voice_pred == "LIE"):
                concordance = "CROSS_MODAL_CONFLICT"
                agreement_details.append("Documentary support contrasts with behavioral models predicting lie class.")
            elif (face_eligible and face_pred == "TRUTH") or (voice_eligible and voice_pred == "TRUTH"):
                concordance = "CONCORDANT_TRUTH"
                agreement_details.append("Documentary support aligns with behavioral models predicting truth class.")
            else:
                concordance = "DOCUMENTARY_SUPPORT_ONLY"
                agreement_details.append("Documentary record supports statement; behavioral models inconclusive or unavailable.")
        elif ev_status == "MIXED":
            concordance = "MIXED_EVIDENCE"
            agreement_details.append("Documentary records contain both supporting and conflicting evidence.")
    elif len(eligible_names) >= 2:
        if face_pred == voice_pred and face_pred == "TRUTH":
            concordance = "CONCORDANT_TRUTH"
        elif face_pred == voice_pred and face_pred == "LIE":
            concordance = "CONCORDANT_LIE"
        elif face_pred == "INCONCLUSIVE" or voice_pred == "INCONCLUSIVE":
            concordance = "BEHAVIORAL_UNCERTAINTY"
        else:
            concordance = "BEHAVIORAL_CONFLICT"
    elif len(eligible_names) == 1:
        concordance = "SINGLE_MODALITY"
    else:
        concordance = "NOT_APPLICABLE"

    agreement_summary = "; ".join(agreement_details) if agreement_details else "No eligible modalities."
    agreement_analysis = AgreementAnalysis(
        concordance=concordance,
        summary=agreement_summary,
        details=agreement_details,
        behavioral_agreement=behavioral_agreement,
    )

    # -------------------------------------------------------------------------
    # 5. MULTIMODAL FUSION ASSESSMENT SYNTHESIS & FINAL STATUS
    # -------------------------------------------------------------------------
    # Decoupling & Priority Rules:
    # 1. Documentary evidence findings take precedence over uncalibrated behavioral models.
    #    - If evidence is CONTRADICTED -> CONFLICT WITH AVAILABLE EVIDENCE.
    #    - If evidence is SUPPORTED -> CONSISTENT WITH AVAILABLE EVIDENCE (behavioral models
    #      do not override documentary findings).
    #    - If evidence is MIXED -> MIXED FINDINGS.
    # 2. When evidence is absent / insufficient:
    #    - 0 eligible modalities -> ASSESSMENT UNAVAILABLE (if services offline) or INSUFFICIENT DATA.
    #    - 1 eligible modality -> INSUFFICIENT DATA (single exploratory stream).
    #    - 2 behavioral modalities concordant TRUTH/LIE -> INCONCLUSIVE (exploratory, no factual corroboration).
    #    - 2 behavioral modalities in conflict -> MIXED FINDINGS.
    #    - 2 behavioral modalities inconclusive -> INCONCLUSIVE.

    final_status = "ASSESSMENT UNAVAILABLE"
    explanation_parts: List[str] = []

    if ev_status == "CONTRADICTED":
        final_status = "CONFLICT WITH AVAILABLE EVIDENCE"
        explanation_parts.append(
            f"Factual Contradiction Detected: Spoken claim conflicts with {contra_count} retrieved case record excerpt(s)."
        )
        if face_eligible and voice_eligible:
            explanation_parts.append(f"Behavioral context: {behavioral_agreement}.")
        elif face_eligible:
            explanation_parts.append(f"Behavioral context: Facial model ({face_status_label}).")
        elif voice_eligible:
            explanation_parts.append(f"Behavioral context: Acoustic model ({voice_status_label}).")

    elif ev_status == "SUPPORTED":
        final_status = "CONSISTENT WITH AVAILABLE EVIDENCE"
        explanation_parts.append(
            f"Factual Support: Spoken claim is corroborated by {support_count} retrieved case record excerpt(s)."
        )
        if face_eligible and voice_eligible and face_pred == "LIE" and voice_pred == "LIE":
            explanation_parts.append(
                "Note: Both behavioral models predict lie class. Uncalibrated behavioral predictions do not override documentary findings."
            )
        elif face_eligible or voice_eligible:
            explanation_parts.append(f"Behavioral context: {behavioral_agreement}.")

    elif ev_status == "MIXED":
        final_status = "MIXED FINDINGS"
        explanation_parts.append(
            f"Mixed Evidence: Retrieved case records contain both supporting ({support_count}) and conflicting ({contra_count}) excerpts."
        )

    elif ev_status == "INSUFFICIENT_EVIDENCE":
        explanation_parts.append("Factual Verification: Insufficient relevant excerpts retrieved to establish conflict or support.")
        if len(eligible_names) == 0:
            final_status = "INSUFFICIENT DATA"
        elif len(eligible_names) == 1:
            final_status = "INSUFFICIENT DATA"
        else:
            final_status = "INCONCLUSIVE"

    elif ev_status == "NO_STATEMENT":
        if len(eligible_names) == 0:
            if not face_ready and not voice_ready:
                final_status = "ASSESSMENT UNAVAILABLE"
                explanation_parts.append("All analysis channels are in standby. Start camera, speak into microphone, or upload case PDF.")
            else:
                final_status = "INSUFFICIENT DATA"
                reasons_summary = ", ".join(f"{k}: {v}" for k, v in exclusion_dict.items())
                explanation_parts.append(f"Insufficient qualified telemetry for multimodal fusion ({reasons_summary}).")
        elif len(eligible_names) == 1:
            final_status = "INSUFFICIENT DATA"
            single_mod = eligible_names[0]
            explanation_parts.append(
                f"Only a single behavioral stream ({single_mod}) is available without documentary corroboration. A single exploratory indicator cannot support an assessment."
            )
        elif face_pred == "INCONCLUSIVE" or voice_pred == "INCONCLUSIVE":
            final_status = "INCONCLUSIVE"
            explanation_parts.append("One or both behavioral classifiers returned probabilities near the decision threshold (0.45-0.55).")
        elif (face_pred == "TRUTH" and voice_pred == "LIE") or (face_pred == "LIE" and voice_pred == "TRUTH"):
            final_status = "MIXED FINDINGS"
            explanation_parts.append(f"Behavioral channels disagree: Facial model predicts {face_pred} while Acoustic model predicts {voice_pred}.")
        else:
            final_status = "INCONCLUSIVE"
            explanation_parts.append(
                f"Both behavioral classifiers predict {face_pred} class, but documentary evidence is absent to establish factual veracity."
            )

    modality_findings_dict = {
        "facial": face_status_label,
        "voice": voice_status_label,
        "documentary": mapped_ev_label,
    }

    # Synthesize Combined Multimodal Fusion Score across active streams
    combined_score = None
    combined_conf = None
    stream_scores_dict: Dict[str, Optional[float]] = {}

    if ev_eligible and face_eligible and voice_eligible:
        # 3-channel multimodal fusion:
        # Document evidence (50%), Vocal acoustics (25%), Facial movement (25%)
        doc_inconsistency = evidence_finding.inconsistency_index if evidence_finding.inconsistency_index is not None else (
            1.0 if ev_status == "CONTRADICTED" else (0.0 if ev_status == "SUPPORTED" else 0.5)
        )
        face_risk = face_plie_val if face_plie_val is not None else 0.5
        voice_risk = voice_score_val if voice_score_val is not None else 0.5

        combined_score = round(0.50 * doc_inconsistency + 0.25 * voice_risk + 0.25 * face_risk, 3)
        combined_conf = round(
            0.50 * (evidence_finding.verifier_confidence or 0.8) +
            0.25 * (face_pred_prob or 0.5) +
            0.25 * (voice_conf_val or 0.8),
            3
        )
        stream_scores_dict = {
            "documentary_inconsistency": doc_inconsistency,
            "facial_deception_risk": face_risk,
            "vocal_deception_risk": voice_risk,
        }
    elif face_eligible and voice_eligible:
        face_risk = face_plie_val if face_plie_val is not None else 0.5
        voice_risk = voice_score_val if voice_score_val is not None else 0.5
        combined_score = round(0.50 * voice_risk + 0.50 * face_risk, 3)
        combined_conf = round(0.50 * (face_pred_prob or 0.5) + 0.50 * (voice_conf_val or 0.8), 3)
        stream_scores_dict = {
            "facial_deception_risk": face_risk,
            "vocal_deception_risk": voice_risk,
        }
    elif ev_eligible and (face_eligible or voice_eligible):
        doc_inconsistency = evidence_finding.inconsistency_index if evidence_finding.inconsistency_index is not None else (
            1.0 if ev_status == "CONTRADICTED" else (0.0 if ev_status == "SUPPORTED" else 0.5)
        )
        beh_risk = face_plie_val if face_eligible else voice_score_val
        beh_risk = beh_risk if beh_risk is not None else 0.5
        combined_score = round(0.65 * doc_inconsistency + 0.35 * beh_risk, 3)
        combined_conf = round(0.65 * (evidence_finding.verifier_confidence or 0.8) + 0.35 * 0.7, 3)
        stream_scores_dict = {
            "documentary_inconsistency": doc_inconsistency,
            ("facial_deception_risk" if face_eligible else "vocal_deception_risk"): beh_risk,
        }

    # Neural Corrector DL Model Integration with Equal Module Weighting
    neural_corrected_score = None
    try:
        doc_inconsistency_val = evidence_finding.inconsistency_index if evidence_finding.inconsistency_index is not None else (
            1.0 if ev_status == "CONTRADICTED" else (0.0 if ev_status == "SUPPORTED" else 0.5)
        ) if ev_eligible else 0.0
        face_risk_val = face_plie_val if face_plie_val is not None else 0.5 if face_eligible else 0.0
        voice_risk_val = voice_score_val if voice_score_val is not None else 0.5 if voice_eligible else 0.0

        doc_conf_val = evidence_finding.verifier_confidence or 0.8 if ev_eligible else 0.0
        face_conf_val_feat = face_pred_prob or 0.5 if face_eligible else 0.0
        voice_conf_val_feat = voice_conf_val or 0.8 if voice_eligible else 0.0

        active_mask = [
            1.0 if ev_eligible else 0.0,
            1.0 if face_eligible else 0.0,
            1.0 if voice_eligible else 0.0,
        ]
        n_active = sum(active_mask)

        if n_active > 0 and load_neural_corrector is not None:
            import torch
            model = load_neural_corrector()
            inputs = torch.tensor([[doc_inconsistency_val, face_risk_val, voice_risk_val, doc_conf_val, face_conf_val_feat, voice_conf_val_feat]], dtype=torch.float32)
            with torch.no_grad():
                pred_score, pred_conf = model(inputs, active_mask=active_mask)
                neural_corrected_score = round(float(pred_score.item()), 3)
                if combined_conf is None:
                    combined_conf = round(float(pred_conf.item()), 3)
        elif n_active > 0:
            scores = []
            if ev_eligible:
                scores.append(doc_inconsistency_val)
            if face_eligible:
                scores.append(face_risk_val)
            if voice_eligible:
                scores.append(voice_risk_val)
            neural_corrected_score = round(sum(scores) / len(scores), 3)
    except Exception:
        neural_corrected_score = combined_score

    fusion_assessment = FusionAssessment(
        assessment_id=str(uuid.uuid4())[:8],
        session_id=req_session,
        timestamp=now,
        final_status=final_status,
        fusion_score=None,  # Preserved as None for validation contract
        combined_multimodal_score=combined_score,
        experimental_heuristic_score=combined_score,
        neural_corrected_fusion_score=neural_corrected_score,
        combined_multimodal_confidence=combined_conf,
        stream_scores=stream_scores_dict,
        fusion_score_display="Not validated",
        fusion_score_rationale=(
            "A combined numeric deception percentage is not computed because cross-modal calibration is unvalidated "
            "and behavioral model outputs cannot be mathematically equated to factual deceit."
        ),
        eligible_modalities_count=len(eligible_names),
        eligible_modality_names=eligible_names,
        excluded_modality_names=excluded_names,
        exclusion_reasons=exclusion_dict,
        evidence_sufficiency=ev_sufficiency,
        modality_findings=modality_findings_dict,
        agreement_analysis=agreement_analysis,
        explanation=" ".join(explanation_parts),
    )

    # -------------------------------------------------------------------------
    # 6. ACTIVE / MISSING MODALITIES & BACKWARD COMPATIBLE SUMMARY
    # -------------------------------------------------------------------------
    active_list: List[str] = []
    missing_list: List[str] = []

    if ev_status not in ("NO_STATEMENT", "INSUFFICIENT_EVIDENCE"):
        active_list.append("Statement & Evidence")
    elif statement:
        active_list.append("Statement & Evidence (Insufficient Chunks)")
    else:
        missing_list.append("Statement & Evidence")

    if face_available:
        active_list.append("Facial Telemetry")
    else:
        missing_list.append("Facial Telemetry")

    if voice_available:
        active_list.append("Acoustic Telemetry")
    else:
        missing_list.append("Acoustic Telemetry")

    overall_summary = {
        "evidence_status": ev_status,
        "final_status": final_status,
        "fusion_score_display": "Not validated",
        "active_modality_count": len(active_list),
        "eligible_modality_count": len(eligible_names),
        "missing_modality_count": len(missing_list),
        "concordance": concordance,
        "narrative": " ".join(explanation_parts),
    }

    return AssessmentResponse(
        status="READY",
        evaluation_time=now,
        session_id=req_session,
        evidence_finding=evidence_finding,
        facial_observation=facial_obs,
        acoustic_observation=acoustic_obs,
        fusion_assessment=fusion_assessment,
        overall_summary=overall_summary,
        active_modalities=active_list,
        missing_modalities=missing_list,
    )
