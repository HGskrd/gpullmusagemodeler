"""Task contracts independent of model families and execution profiles."""

TASK_PRIMITIVES = {
    "choice": "Choose",
    "boolean": "Decide",
    "score": "Score",
    "span_extraction": "Extract spans",
}


def normalize_task_primitive(value):
    return value if isinstance(value, str) and value in TASK_PRIMITIVES else "choice"


def normalize_task_count(value, default=1):
    try:
        return max(1, min(128, int(value)))
    except (ValueError, TypeError, OverflowError):
        return default


def supports_task(model, primitive):
    profile = model.decision_profile
    return profile is not None and primitive in profile.task_primitives
