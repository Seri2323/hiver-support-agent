import pandas as pd

from escalation import decide_escalation


GOLDEN_SET = "data/gold/golden_set_200_frozen.csv"


df = pd.read_csv(GOLDEN_SET)

false_negatives = []

for _, row in df.iterrows():
    customer_message = str(row["customer_message"])
    intent = str(row["intent"])
    actual = str(row["should_escalate"]).lower()

    predicted, reason = decide_escalation(
        customer_message,
        intent,
    )

    if actual == "yes" and predicted == "no":
        false_negatives.append(
            {
                "intent": intent,
                "customer_message": customer_message,
                "reason": reason,
            }
        )

print("\n" + "=" * 80)
print("ESCALATION FALSE NEGATIVES")
print("=" * 80)

print(f"\nMissed escalations: {len(false_negatives)}\n")

for i, item in enumerate(false_negatives, start=1):

    print("-" * 80)
    print(f"CASE {i}")
    print("Intent:", item["intent"])
    print("Customer:", item["customer_message"])
    print("Current reason:", item["reason"])