import re
import time

import ollama
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


MODEL_NAME = "qwen3:4b"
GOLDEN_SET = "data/gold/golden_set_200_frozen.csv"
OUTPUT_FILE = "data/processed/escalation_predictions.csv"


def classify_escalation(customer_message: str, intent: str) -> str:
    prompt = f"""
You are an escalation classifier for a customer-support system.

Decide whether this customer conversation should be escalated to
a human/support specialist.

ESCALATE = yes when the customer:
- explicitly asks for a human, agent, manager, or escalation
- reports repeated unsuccessful attempts or says nothing works
- cannot use, access, charge, start, or otherwise use the device/account
- reports a severe or potentially damaging hardware problem
- reports a prolonged unresolved support problem
- clearly requires account-specific, order-specific, payment-specific,
  or other intervention that normal troubleshooting cannot safely resolve

DO NOT ESCALATE = no when the message is simply:
- a normal how-to question
- a routine troubleshooting question
- a feature question
- a complaint that can still be handled with ordinary troubleshooting
- a vague statement with no evidence that specialist intervention is needed

Use the customer message and intent together.
Do not infer escalation solely from angry language.

Return exactly one word:
yes
or
no

INTENT:
{intent}

CUSTOMER MESSAGE:
{customer_message}
""".strip()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        think=False,
        stream=False,
        options={
            "temperature": 0,
        },
    )

    text = response.message.content.strip().lower()

    # Extract only a yes/no decision.
    match = re.search(r"\b(yes|no)\b", text)

    if match:
        return match.group(1)

    # Conservative fallback.
    return "no"


def main():
    df = pd.read_csv(GOLDEN_SET)

    predictions = []

    print("=" * 70)
    print("LLM ESCALATION CLASSIFIER")
    print("=" * 70)
    print(f"Examples: {len(df)}")
    print(f"Model: {MODEL_NAME}")
    print()

    for i, row in df.iterrows():
        message = str(row["customer_message"])
        intent = str(row["intent"])

        prediction = classify_escalation(
            message,
            intent,
        )

        predictions.append(prediction)

        print(
            f"[{i + 1:3d}/{len(df)}] "
            f"predicted={prediction:<3} "
            f"intent={intent}"
        )

    actual = (
        df["should_escalate"]
        .astype(str)
        .str.lower()
        .tolist()
    )

    df["predicted_escalation"] = predictions

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    accuracy = accuracy_score(
        actual,
        predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predictions,
        labels=["yes"],
        average="binary",
        pos_label="yes",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predictions,
            labels=["no", "yes"],
            zero_division=0,
        )
    )

    print(f"\nSaved predictions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()