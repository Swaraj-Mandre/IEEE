def compute_gap_score(mastered: int, total: int) -> float:
    if total == 0:
        return 0.0
    score = 1 - (mastered / total)
    return round(score, 4)


def analyse_response(student_reply: str, current_concept: str) -> dict:
    reply_lower = student_reply.lower().strip()

    confusion_signals = [
        "don't understand",
        "dont understand",
        "not clear",
        "confused",
        "what do you mean",
        "i don't get",
        "i dont get",
        "can you explain",
        "what is",
        "huh",
        "?",
        "no",
        "nope",
    ]

    understanding_signals = [
        "i understand",
        "i get it",
        "got it",
        "makes sense",
        "okay",
        "ok",
        "yes",
        "understood",
        "clear",
        "thanks",
        "thank you",
        "i see",
    ]

    confusion_count = sum(1 for signal in confusion_signals if signal in reply_lower)
    understanding_count = sum(1 for signal in understanding_signals if signal in reply_lower)

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