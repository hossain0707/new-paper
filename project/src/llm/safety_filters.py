"""
LEGACY DIAGNOSTICS ONLY. These regex helpers are not the revision release gate.
Use src.revision.grounding for typed validation and controlled rendering.

Historical description:
Safety filters applied to generated clinical narratives before they reach a user or
clinician: (1) numerical cross-check against the structured input (rejects hallucinated
glucose values), and (2) a FActScore-style factual-consistency stub.

The full FActScore pipeline (https://github.com/shmsw25/FActScore) decomposes a
narrative into atomic facts and checks each against a knowledge source using a
support-estimation model. That full pipeline is not wired up in this session (it
needs its own model weights/API budget); `factscore_stub` here does the cheaper,
fully local part -- numeric-value grounding -- and marks anything it cannot verify
as `unverified` rather than assuming it is correct.
"""

import re

# Only numbers immediately tagged with a clinical unit are treated as factual claims
# that must be grounded in the structured input. An early real test run (DeepSeek V4
# Flash, see revision_notes.md) showed that checking every digit in the text -- including
# carb-gram dosage advice ("15g"), re-check intervals ("15 minutes"), and risk tiers ("Tier
# 4") -- produces a high false-positive rate on clinically correct narratives. Those are
# general guidance/derived labels, not claims about this patient's measured data, so they
# should not be cross-checked against predicted_trajectory/vitals/SHAP values.
#
# Two defects in the first version of this pattern were found by auditing the filter's
# own output against the narratives it had scored, and are fixed here:
#   1. `°?C\b` under IGNORECASE matched the "1c" inside "HbA1c", injecting a spurious
#      1.0 "clinical claim" into 14 of 15 narratives. Celsius now requires the degree
#      sign, and a lookbehind forbids a preceding letter/digit so that identifiers like
#      "A1c" or "T2D" can never yield a numeric claim.
#   2. Numbers appearing in cited guideline thresholds ("ADA guidelines: above 250 mg/dL")
#      were flagged as ungrounded, although they are general clinical guidance rather
#      than assertions about this patient's measured state -- the same category that
#      dosages and time intervals were already excluded for. See GUIDELINE_CONTEXT.
UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(-?\d+\.?\d*)\s*(mg/dL|bpm|mmHg|°C|%)", re.IGNORECASE
)

# A number is treated as cited guidance, not a patient-specific claim, when its
# sentence carries one of these markers. This is a deliberately conservative
# heuristic: it can only ever suppress a flag, so it is reported as a known source
# of false negatives rather than presented as exact.
GUIDELINE_CONTEXT = re.compile(
    r"\b(ADA|ISPAD|IDF|guideline|guidelines|recommend\w*|threshold|target range|"
    r"generally|typically|usually)\b", re.IGNORECASE
)


def extract_numbers(text):
    """All numbers in the text (used only for factscore_stub's denominator)."""
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def _sentences(text):
    return re.split(r"(?<=[.!?])\s+|\n+", text)


def extract_grounded_claims(text, exclude_guideline_context=True):
    """Numbers explicitly tagged with a clinical unit (mg/dL, bpm, mmHg, °C, %) that
    assert something about *this patient's* state -- the claims that must trace back to
    the structured input. Numbers inside cited guideline statements are excluded when
    `exclude_guideline_context` is set (see GUIDELINE_CONTEXT)."""
    claims = []
    for sentence in _sentences(text):
        if exclude_guideline_context and GUIDELINE_CONTEXT.search(sentence):
            continue
        claims.extend(float(m.group(1)) for m in UNIT_PATTERN.finditer(sentence))
    return claims


def numerical_cross_check(narrative, case_input, tolerance=2.0):
    """Flags any unit-tagged clinical number in the narrative that is not within
    `tolerance` of a value present in predicted_trajectory, vitals, top_shap_features,
    or risk_score of the input JSON. Numbers without a clinical unit (dosage amounts,
    time intervals, tier labels) are intentionally not checked -- see UNIT_PATTERN note."""
    allowed_values = []
    traj = case_input.get("predicted_trajectory", {})
    for k, v in traj.items():
        if isinstance(v, (int, float)):
            allowed_values.append(float(v))
        elif isinstance(v, list):
            allowed_values.extend(float(x) for x in v)

    for feat in case_input.get("top_shap_features", []):
        if "value" in feat:
            allowed_values.append(float(feat["value"]))

    vitals = case_input.get("vitals", {})
    for v in vitals.values():
        if isinstance(v, (int, float)):
            allowed_values.append(float(v))

    risk = case_input.get("risk_score")
    if risk is not None:
        allowed_values.append(float(risk))

    flagged = []
    for n in extract_grounded_claims(narrative):
        if not any(abs(n - a) <= tolerance for a in allowed_values):
            flagged.append(n)

    return {"passed": len(flagged) == 0, "flagged_numbers": flagged, "allowed_values": allowed_values}


# Risk score tiers, matching the original paper's Table 1 exactly.
RISK_TIERS = [
    (0, 39, "Safe", 1),
    (40, 59, "Caution", 2),
    (60, 79, "Alert", 3),
    (80, 94, "Emergency", 4),
    (95, 100, "Critical", 5),
]

# Models state the tier either numerically ("Tier 4") or, far more often in practice,
# by its Table 1 label ("Caution-tier reading", "Critical tier"). The first version of
# this filter matched only the numeric form; auditing its output showed that form never
# occurred, so the filter silently verified nothing while reporting a 100% pass rate,
# and four genuinely mislabelled narratives passed. Both forms are matched now.
TIER_NUMBER_PATTERN = re.compile(r"\btier\s*[:\-]?\s*(\d)\b", re.IGNORECASE)
TIER_LABEL_PATTERN = re.compile(
    r"\b(Safe|Caution|Alert|Emergency|Critical)\b(?=[\s\-–—]*(?:tier|reading|level|state|status))"
    r"|(?:tier)[\s:\-–—]*\b(Safe|Caution|Alert|Emergency|Critical)\b",
    re.IGNORECASE,
)
LABEL_TO_TIER = {label.lower(): num for _, _, label, num in RISK_TIERS}

# A tier-shaped phrase whose label is not one of Table 1's five is itself a failure:
# the model has invented a severity category that does not exist in the alert protocol
# (observed in practice: "Warning-tier reading", where Table 1 defines no "Warning").
# Only the attributive position -- "<Label>-tier" / "<Label> tier" -- is inspected. The
# trailing position is deliberately not, because in "Caution-tier reading" the word
# following "tier" is an ordinary noun, and treating it as a label produced false
# positives on correctly-labelled narratives.
TIER_SHAPED_PATTERN = re.compile(r"\b([A-Za-z]+)[\s\-–—]*tier\b", re.IGNORECASE)
_TIER_STOPWORDS = {
    "the", "a", "an", "this", "that", "risk", "score", "high", "low", "same", "each",
    "per", "its", "of", "in", "at", "and", "or", "is", "was", "current", "corresponding",
}


def correct_tier(risk_score):
    """Continuous, half-open score intervals; invalid scores have no tier."""
    if not isinstance(risk_score, (int, float)) or not 0 <= risk_score <= 100:
        return None
    for threshold, label, tier in [(40, "Safe", 1), (60, "Caution", 2),
                                   (80, "Alert", 3), (95, "Emergency", 4),
                                   (101, "Critical", 5)]:
        if risk_score < threshold:
            return {"tier": tier, "label": label}


def extract_stated_tiers(narrative):
    """Returns the tier numbers asserted by the narrative, whether written numerically
    ('Tier 4') or as a Table 1 label ('Emergency tier', 'Caution-tier reading')."""
    tiers = [int(m.group(1)) for m in TIER_NUMBER_PATTERN.finditer(narrative)]
    for m in TIER_LABEL_PATTERN.finditer(narrative):
        label = (m.group(1) or m.group(2) or "").lower()
        if label in LABEL_TO_TIER:
            tiers.append(LABEL_TO_TIER[label])
    return tiers


def extract_invented_tier_labels(narrative):
    """Attributive tier labels ('<Label>-tier') that are not one of Table 1's five."""
    invented = []
    for m in TIER_SHAPED_PATTERN.finditer(narrative):
        word = (m.group(1) or "").lower()
        if not word or word in _TIER_STOPWORDS or word.isdigit():
            continue
        if word not in LABEL_TO_TIER:
            invented.append(word)
    return invented


def tier_consistency_check(narrative, case_input):
    """Checks that any risk tier the narrative states -- numerically, or by Table 1
    label -- matches the tier implied by risk_score, and that it does not invent a
    severity label outside Table 1.

    `checked` distinguishes a real verification from a vacuous pass: when the narrative
    states no tier at all there is nothing to verify, and such cases must be excluded
    from any reported pass rate rather than counted as successes."""
    risk_score = case_input.get("risk_score")
    if risk_score is None:
        return {"checked": False, "passed": None, "note": "no risk_score in input"}

    expected = correct_tier(risk_score)
    if expected is None:
        return {"checked": False, "passed": False, "expected": None,
                "note": "invalid risk score"}
    stated = extract_stated_tiers(narrative)
    invented = extract_invented_tier_labels(narrative)

    if not stated and not invented:
        return {"checked": False, "passed": None, "expected": expected, "stated_tiers": [],
                "invented_labels": [], "note": "narrative states no tier -- nothing to verify"}

    passed = bool(stated) and not invented and all(t == expected["tier"] for t in stated)
    return {"checked": True, "passed": passed, "expected": expected,
            "stated_tiers": stated, "invented_labels": invented}


def factscore_stub(narrative, case_input):
    """Cheap local proxy for factual consistency: fraction of unit-tagged clinical
    claims (mg/dL, bpm, mmHg, %, C) that are numerically grounded in the structured
    input. Real FActScore requires the full decomposition+support pipeline; treat
    this as a pre-filter, not a replacement, and report it as such in the manuscript."""
    check = numerical_cross_check(narrative, case_input)
    total_numbers = len(extract_grounded_claims(narrative))
    if total_numbers == 0:
        return {"proxy_factscore": None, "note": "no unit-tagged clinical claims to check"}
    grounded = total_numbers - len(check["flagged_numbers"])
    return {
        "proxy_factscore": grounded / total_numbers,
        "flagged_numbers": check["flagged_numbers"],
        "note": "local numeric-grounding proxy only, NOT the full FActScore pipeline",
    }
