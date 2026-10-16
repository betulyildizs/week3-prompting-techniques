"""K-shot prompting — ticket classification.

Task: classify a customer message into one of: billing, shipping, returns,
technical, other. Use in-context EXAMPLES (k-shot) so the model outputs exactly the
label, lowercase, with no extra text.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python k_shot_prompting.py
"""
from ollama import chat

MODEL = "llama3.1:8b"
NUM_RUNS = 5

YOUR_SYSTEM_PROMPT = """Classify each customer message into exactly one category:
billing, shipping, returns, or technical.
Use the examples to match the intent of the message. Reply with only the
lowercase category name, with no punctuation or explanation.

Examples:
Message: My payment went through twice.
Category: billing

Message: I don't see a charge for my recent order.
Category: billing

Message: My delivery hasn't arrived yet.
Category: shipping

Message: Can I get a tracking update?
Category: shipping

Message: I need to exchange this item for a refund.
Category: returns

Message: The product arrived damaged and I want to send it back.
Category: returns

Message: The checkout page gives me an error.
Category: technical

Message: I can't log in to the store app.
Category: technical"""

CASES = [
    ("My card was charged twice for order 4412.", "billing"),
    ("Where is my package? It's been two weeks.", "shipping"),
    ("I want to send the shoes back for a refund.", "returns"),
    ("The app crashes every time I open my cart.", "technical"),
]


def classify(system_prompt: str, message: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for message, expected in CASES:
        got = classify(YOUR_SYSTEM_PROMPT, message)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {message!r} -> got {got!r}, expected {expected!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
