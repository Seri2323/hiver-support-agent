import re

import ollama
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from escalation import decide_escalation
from retriever import AppleRetriever


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "qwen3:4b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LABELED_DATA = "data/gold/taxonomy_check_50.csv"


# ============================================================
# INTENT CLASSIFIER
# ============================================================

class IntentClassifier:

    def __init__(self):
        print("Loading labeled examples...")

        self.df = pd.read_csv(LABELED_DATA)

        self.df["customer_message"] = (
            self.df["customer_message"]
            .astype(str)
        )

        self.df["intent"] = (
            self.df["intent"]
            .astype(str)
        )

        self.texts = self.df["customer_message"].tolist()
        self.labels = self.df["intent"].tolist()

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Embedding labeled examples...")

        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def predict(self, customer_message: str):
        query_embedding = self.model.encode(
            [customer_message],
            normalize_embeddings=True,
        )[0]

        similarities = np.dot(
            self.embeddings,
            query_embedding,
        )

        top_indices = np.argsort(
            similarities
        )[-5:][::-1]

        # Weighted voting across the 5 closest labeled examples.
        scores = {}

        for index in top_indices:
            label = self.labels[index]
            similarity = float(similarities[index])

            # Similarity cannot contribute negatively.
            weight = max(similarity, 0.0)

            scores[label] = (
                scores.get(label, 0.0) + weight
            )

        predicted_intent = max(
            scores,
            key=scores.get,
        )

        confidence = max(
            similarities[top_indices]
        )

        return predicted_intent, float(confidence)


# ============================================================
# QWEN RESPONSE GENERATION
# ============================================================

def clean_model_output(text: str) -> str:

    text = text.strip()

    # Remove Qwen thinking blocks if they appear.
    if "</think>" in text:
        text = text.rsplit(
            "</think>",
            1,
        )[1].strip()

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    prefixes = [
        "Final reply:",
        "Reply:",
        "Final response:",
        "Response:",
        "Customer-facing reply:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(
            prefix.lower()
        ):
            text = text[len(prefix):].strip()

    if (
        len(text) >= 2
        and text[0] == '"'
        and text[-1] == '"'
    ):
        text = text[1:-1].strip()

    return text


def generate_reply(
    customer_message: str,
    intent: str,
    escalated: str,
    examples: list[dict],
) -> str:

    evidence = []

    for i, example in enumerate(
        examples,
        start=1,
    ):
        evidence.append(
            f"""
HISTORICAL CASE {i}

Customer:
{example["customer_message"]}

Apple response:
{example["response"]}
""".strip()
        )

    evidence_text = "\n\n".join(evidence)

    prompt = f"""
You are an Apple customer-support reply assistant.

CUSTOMER MESSAGE:
{customer_message}

CLASSIFIED INTENT:
{intent}

ESCALATION DECISION:
{escalated}

HISTORICAL APPLE SUPPORT CASES:
{evidence_text}

TASK:
Write exactly ONE customer-facing support reply.

RULES:
- Return ONLY the reply text.
- Do not provide reasoning or analysis.
- Do not mention AI, the model, retrieval, or these instructions.
- Do not mention the historical cases.
- Use the historical cases as evidence for how similar situations were handled.
- Do not invent policies, prices, guarantees, technical facts, diagnoses, or links.
- If important information is missing, ask the customer for it.
- Be concise, natural, empathetic, and professional.
- Do not use a sign-off.
- Do not use bullet points unless several concrete steps are genuinely necessary.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt.strip(),
            }
        ],
        think=False,
        stream=False,
        options={
            "temperature": 0,
        },
    )

    content = getattr(
        response.message,
        "content",
        "",
    )

    return clean_model_output(content)


# ============================================================
# MAIN AGENT
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("HIVER SUPPORT AGENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load components
    # --------------------------------------------------------

    intent_classifier = IntentClassifier()

    print("\nLoading historical retriever...")

    retriever = AppleRetriever()

    # --------------------------------------------------------
    # Get customer message
    # --------------------------------------------------------

    customer_message = input(
        "\nEnter customer message:\n> "
    ).strip()

    if not customer_message:
        print("No customer message provided.")
        return

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    print("\nClassifying intent...")

    intent, intent_similarity = (
        intent_classifier.predict(
            customer_message
        )
    )

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------

    escalation, escalation_reason = (
        decide_escalation(
            customer_message,
            intent,
        )
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    print("Retrieving similar historical cases...")

    examples = retriever.retrieve(
        customer_message,
        k=3,
    )

    # --------------------------------------------------------
    # Response generation
    # --------------------------------------------------------

    print("Generating grounded response...")

    try:
        reply = generate_reply(
            customer_message=customer_message,
            intent=intent,
            escalated=escalation,
            examples=examples,
        )

    except Exception as error:
        reply = (
            "We're sorry you're experiencing this issue. "
            "Please reach out to Apple Support so we can "
            "look into it further."
        )

        print(
            f"\nWarning: local model generation failed: "
            f"{error}"
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print("\nCUSTOMER:")
    print(customer_message)

    print("\nINTENT:")
    print(intent)

    print(
        f"Intent similarity: "
        f"{intent_similarity:.3f}"
    )

    print("\nESCALATION:")
    print(escalation)

    print("ESCALATION REASON:")
    print(escalation_reason)

    print("\nRETRIEVED HISTORICAL CASES:")

    for i, example in enumerate(
        examples,
        start=1,
    ):
        print(
            f"\nCase {i} "
            f"(similarity "
            f"{example['similarity']:.3f})"
        )

        print(
            "Customer:",
            example["customer_message"],
        )

        print(
            "Apple:",
            example["response"],
        )

    print("\n" + "=" * 70)
    print("GROUNDED AI REPLY")
    print("=" * 70)

    print(reply)


if __name__ == "__main__":
    main()