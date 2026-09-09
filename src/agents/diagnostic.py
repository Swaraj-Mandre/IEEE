import re

# Phrases are matched whole-word and longest-first, so "not clear" is consumed
# before plain "clear" gets a chance to match the same words.
CONFUSION_SIGNALS = [
    "i do not understand", "i don't understand", "i dont understand",
    "do not understand", "don't understand", "dont understand",
    "does not make sense", "doesn't make sense", "makes no sense",
    "nothing is clear", "not very clear", "not clear", "unclear",
    "i don't get it", "i dont get it", "don't get", "dont get",
    "what do you mean", "can you explain", "explain again", "say it again",
    "one more time", "no idea", "not sure",
    "confusing", "confused", "too hard", "difficult",
    "huh", "nope", "no",
]

UNDERSTANDING_SIGNALS = [
    "makes sense", "make sense",
    "i understand", "understand now", "understood",
    "i get it", "get it now", "got it",
    "already know", "i know", "i knew", "i see",
    "thank you", "thanks",
    "crystal clear", "very clear", "clear",
    "okay", "yes", "yeah", "yep", "sure", "easy", "ok",
]


def compute_gap_score(mastered: int, total: int) -> float:
    if total == 0:
        return 0.0
    score = 1 - (mastered / total)
    return round(score, 4)


def _count_signals(text: str, signals: list[str]) -> tuple[int, str]:
    """Count whole-word matches and blank each one out so shorter phrases cannot re-match it."""
    count = 0
    for signal in sorted(signals, key=len, reverse=True):
        text, hits = re.subn(rf"\b{re.escape(signal)}\b", " ", text)
        count += hits
    return count, text


def analyse_response(student_reply: str, current_concept: str) -> dict:
    reply_lower = student_reply.lower().strip()

    # Confusion runs first so a negated phrase wins over the positive word inside it.
    confusion_count, remaining = _count_signals(reply_lower, CONFUSION_SIGNALS)
    understanding_count, _ = _count_signals(remaining, UNDERSTANDING_SIGNALS)

    if confusion_count > understanding_count:
        verdict = "confused"
    elif understanding_count > 0:
        verdict = "understood"
    else:
        verdict = "unclear"

    return {
        "verdict": verdict,
        "confusion_signals_found": confusion_count,
        "understanding_signals_found": understanding_count,
        "student_reply": student_reply,
        "concept": current_concept,
    }


def should_backtrack(
    diagnostic_result: dict,
    gap_score: float,
    gap_threshold: float = 0.4,
) -> bool:
    if diagnostic_result["verdict"] == "confused":
        return True
    if gap_score > gap_threshold and diagnostic_result["verdict"] == "unclear":
        return True
    return False


def should_advance(diagnostic_result: dict) -> bool:
    return diagnostic_result["verdict"] == "understood"


def should_simplify(diagnostic_result: dict, backtrack_count: int) -> bool:
    return diagnostic_result["verdict"] == "confused" and backtrack_count >= 2
