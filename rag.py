"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


def YOUR_RETRIEVER(question: str, chunks: list[str]) -> list[str]:
    stop_words = {
        "a", "an", "and", "are", "do", "does", "for", "have", "how",
        "i", "in", "is", "it", "my", "of", "the", "to", "we", "what",
        "when", "where", "you",
    }

    def terms(text: str) -> set[str]:
        words = {
            word.strip(".,?!'\"():;").lower()
            for word in text.split()
        }
        normalized = set()
        for word in words:
            if not word or word in stop_words:
                continue
            if word.endswith("ing") and len(word) > 5:
                word = word[:-3]
                if len(word) > 2 and word[-1] == word[-2]:
                    word = word[:-1]
            elif word.endswith("ly") and len(word) > 5:
                word = word[:-2]
            elif word.endswith("s") and len(word) > 3:
                word = word[:-1]
            normalized.add(word)
        return normalized

    question_terms = terms(question)
    ranked_chunks = []
    for index, chunk in enumerate(chunks):
        overlap = len(question_terms & terms(chunk))
        if overlap:
            ranked_chunks.append((overlap, index, chunk))
    ranked_chunks.sort(key=lambda result: (-result[0], result[1]))
    return [chunk for _, _, chunk in ranked_chunks]


YOUR_SYSTEM_PROMPT = """Answer the customer's question using only facts directly
supported by the supplied help-center context. Do not add outside knowledge or
guess. If the context is empty or does not contain the answer, politely say
"I don't know based on the help-center information provided." Keep the answer
brief and relevant."""

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
