import re


def decide_escalation(customer_message: str, intent: str) -> tuple[str, str]:
    text = customer_message.lower().strip()

    # Strong escalation signals.
    strong_patterns = [
        r"\bfix (this|it|this issue)\b",
        r"\bplease fix\b",
        r"\bneed (help|assistance) urgently\b",
        r"\bimmediately\b",
        r"\bspeak to (a|an) (human|person|agent)\b",
        r"\breal person\b",
        r"\bsupervisor\b",
        r"\bmanager\b",
        r"\bescalat(e|ion)\b",
        r"\bno one is responding\b",
        r"\bstill waiting\b",
        r"\bcan't use my (iphone|phone|device)\b",
        r"\bcannot use my (iphone|phone|device)\b",
        r"\bphone is (disabled|dead|unusable)\b",
        r"\biphone is (disabled|dead|unusable)\b",
        r"\bwon't turn on\b",
        r"\bwill not turn on\b",
        r"\bexplod(e|ed|ing)\b",
    ]

    strong_hits = sum(bool(re.search(p, text)) for p in strong_patterns)

    if strong_hits > 0:
        return "yes", "Strong escalation signal detected."

    # Repeated failure / unresolved issue.
    failure_patterns = [
        "tried everything",
        "tried all",
        "nothing works",
        "nothing helps",
        "still not working",
        "still doesn't work",
        "still won't work",
        "does not help",
        "doesn't help",
        "keeps happening",
        "keeps restarting",
        "multiple times",
        "several times",
        "for hours",
        "unable to resolve",
        "cannot resolve",
        "can't resolve",
        "already contacted",
        "already called",
        "already spoke",
        "already tried",
        "restarted the phone",
        "restarted my phone",
    ]

    failure_hits = sum(signal in text for signal in failure_patterns)

    # Strong operational/device failure.
    severe_patterns = [
        "black all day",
        "blinking apple",
        "completely unusable",
        "severe",
        "crash",
        "crashing",
        "constant reset",
        "constant soft reset",
        "freezes all the time",
        "frozen",
        "won't charge",
        "will not charge",
        "won't work",
        "will not work",
        "can't access",
        "cannot access",
        "locked out",
        "account locked",
        "forgotten password",
        "wrong dob",
    ]

    severe_hits = sum(signal in text for signal in severe_patterns)

    # Frustration / service-failure signals.
    frustration_patterns = [
        "frustrating",
        "frustrated",
        "terrible",
        "horrible",
        "ridiculous",
        "unacceptable",
        "disappointed",
        "annoying",
        "annoyed",
        "beyond annoyed",
        "wtf",
        "fix it",
        "fix this",
        "please help",
        "help !!!!!",
        "help!!!",
        "help!",
    ]

    frustration_hits = sum(signal in text for signal in frustration_patterns)

    # Account/security issues are generally high-touch.
    security_signals = [
        "apple id",
        "icloud",
        "password",
        "security questions",
        "locked out",
        "account",
    ]

    security_hits = sum(signal in text for signal in security_signals)

    # Purchase/payment cases often need account-specific intervention.
    purchase_signals = [
        "charged",
        "payment",
        "refund",
        "visa card",
        "buy in app",
        "purchase",
        "reservation",
        "order",
    ]

    purchase_hits = sum(signal in text for signal in purchase_signals)

    # Score rather than requiring a single exact phrase.
    score = 0

    if failure_hits >= 1:
        score += 2

    if severe_hits >= 1:
        score += 2

    if frustration_hits >= 1:
        score += 1

    if security_hits >= 1:
        score += 1

    if purchase_hits >= 1:
        score += 1

    # Intent-specific boost.
    if intent in {
        "account_icloud_issue",
        "purchase_order_support",
    }:
        score += 1

    # Two moderate signals are enough to escalate.
    if score >= 2:
        return "yes", "Multiple unresolved, severe, or service-related signals detected."

    return "no", "Issue appears suitable for normal support troubleshooting."