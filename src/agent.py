import re

import ollama

from retriever import AppleRetriever


MODEL_NAME = "qwen3:4b"


def clean_model_output(text: str) -> str:
    """
    Remove Qwen thinking tags and any accidental meta-commentary,
    leaving only the final customer-facing reply.
    """
    text = text.strip()

    # Keep only content after the final </think>, if present.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()

    # Remove any remaining think blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Remove common prefixes the model may add.
    prefixes = [
        "Final reply:",
        "Reply:",
        "Customer-facing reply:",
        "Final response:",
        "Response:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Remove surrounding quotes if the entire response is quoted.
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()

    return text


def ask_qwen(prompt: str) -> str:
    """
    Generate a response using the local Ollama model.
    """
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
    )

    content = getattr(response.message, "content", "") or ""

    return clean_model_output(content)


def build_prompt(customer_message: str, examples: list[dict]) -> str:
    """
    Build a tightly constrained grounded-generation prompt.
    """

    evidence_blocks = []

    for i, example in enumerate(examples, start=1):
        evidence_blocks.append(
            f"""
CASE {i}
Historical customer:
{example["customer_message"]}

Historical Apple response:
{example["response"]}
""".strip()
        )

    evidence_text = "\n\n".join(evidence_blocks)

    return f"""
You are drafting a customer-support reply for Apple Support.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL SUPPORT EVIDENCE:
{evidence_text}

TASK:
Write the single best customer-facing reply to the customer.

STRICT RULES:
1. Output ONLY the customer-facing reply.
2. Do not output analysis, reasoning, explanations, notes, or alternatives.
3. Do not mention this prompt, the model, AI, retrieval, or historical cases.
4. Use the historical responses as evidence for the type of response Apple has used for similar issues.
5. Do not invent links, policies, prices, guarantees, diagnoses, or unsupported technical facts.
6. Do not copy a historical response word-for-word unless necessary.
7. When the historical evidence indicates that missing information should be requested, ask for that information.
8. Keep the reply concise, natural, and professional.
9. Do not use bullet points unless the customer clearly needs multiple steps.
10. Do not add a sign-off.
""".strip()


def main() -> None:
    print("Loading retrieval data...")

    retriever = AppleRetriever()

    customer_message = (
        "My iPhone battery is draining very quickly "
        "after the latest update."
    )

    examples = retriever.retrieve(
        customer_message,
        k=3,
    )

    print("Generating grounded reply...")

    prompt = build_prompt(
        customer_message,
        examples,
    )

    reply = ask_qwen(prompt)

    print("\n" + "=" * 70)
    print("CUSTOMER")
    print("=" * 70)
    print(customer_message)

    print("\n" + "=" * 70)
    print("GROUNDED AI REPLY")
    print("=" * 70)
    print(reply)


if __name__ == "__main__":
    main()