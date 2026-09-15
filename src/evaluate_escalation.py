import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from escalation import decide_escalation


GOLDEN_SET = "data/gold/golden_set_200_frozen.csv"


def main():
    df = pd.read_csv(GOLDEN_SET)

    predictions = []

    for _, row in df.iterrows():
        predicted, reason = decide_escalation(
            str(row["customer_message"]),
            str(row["intent"]),
        )

        predictions.append(predicted)

    actual = df["should_escalate"].astype(str).str.lower().tolist()

    print("\n" + "=" * 70)
    print("ESCALATION POLICY EVALUATION")
    print("=" * 70)

    print("\nConfusion-style counts:")

    for label in ["yes", "no"]:
        print(
            f"Actual {label}: {actual.count(label):3d} | "
            f"Predicted {label}: {predictions.count(label):3d}"
        )

    accuracy = accuracy_score(actual, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predictions,
        labels=["yes"],
        average="binary",
        pos_label="yes",
        zero_division=0,
    )

    print("\nOverall metrics:")
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


if __name__ == "__main__":
    main()