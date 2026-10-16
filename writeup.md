# Week 3 — Prompting Techniques Write-up

Model used for all four checks: `llama3.1:8b` (temperature `0.0`).

## 1. Few-shot ticket classification

**Final system prompt**

```text
Classify each customer message into exactly one category:
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
Category: technical
```

**Observed sample:** `My card was charged twice for order 4412.` → `billing`

**Observed check:** `python k_shot_prompting.py` — 4/4 correct.

**What I learned:** Labeled examples demonstrate both the category boundaries and the exact one-word output format.

## 2. Chain-of-thought refund decision

**Final system prompt**

```text
Evaluate the order against every requirement in the refund
policy. Reason through the purchase date, item condition, and final-sale status
step by step. A refund is eligible only if it was purchased within 30 days, is
unused and in original condition, and is not final sale. If any requirement
fails, decide NO. End your response with exactly one final line:
DECISION: YES
or
DECISION: NO
Do not write anything after the decision line.
```

**Observed sample:** `Bought 10 days ago, unused, not final sale.` → `YES`

**Observed check:** `python chain_of_thought.py` — 3/3 correct. The script prints the parsed decision, not the model's full reasoning text.

**What I learned:** Breaking eligibility into explicit policy criteria and constraining the final line makes the decision machine-checkable.

## 3. Tool calling

**Final system prompt**

```text
You are a concise store support assistant. When a customer
asks about an order, extract its order ID and call the get_order_status tool
before answering. Use the status returned by the tool in a short, clear reply.
Do not guess or invent an order status.
```

**Tool description:** `Look up the current status of a store order. Call this whenever the customer asks about an order's status or delivery.`

**Parameter:** `order_id` is a required string: `The order ID exactly as provided by the customer.`

**Observed sample:** `Hi, can you check on order 4412 for me?` → tool called with `order_id='4412'`; the script then printed `shipped, arriving Friday`.

**Observed check:** `python tool_calling.py` — PASS, tool called with the expected order ID. The supplied check prints the local fake-database result after validating the tool call; it does not send that result back to the model for a second response, so the observed check does not independently verify a model-generated answer using the result.

**What I learned:** A clear tool description and parameter description guide the model to call the function with the extracted identifier instead of guessing.

## 4. Retrieval-augmented generation

**Final system prompt**

```text
Answer the customer's question using only facts directly
supported by the supplied help-center context. Do not add outside knowledge or
guess. If the context is empty or does not contain the answer, politely say
"I don't know based on the help-center information provided." Keep the answer
brief and relevant.
```

**Retriever:** Splits the query and documents into lowercase terms, removes common stop words, normalizes simple `-ing`, `-ly`, and plural `-s` endings, then returns only chunks with term overlap, ranked by overlap while preserving source order for ties.

**Observed samples:**
- `How long do I have to return an item?` → `you may return most items within 30 days of delivery for a full refund.`
- `What's your CEO's home address?` → `i don't know based on the help-center information provided.`

**Observed check:** `python rag.py` — 3/3 correct.

**What I learned:** Retrieval provides relevant source text, while an explicit instruction to abstain on unsupported questions helps limit hallucinations.

## Overall reflection

These checks show that LLMs can map language to structured categories, apply a small set of policy criteria, and select a tool when the task and output contract are explicit. They still need scaffolding: examples clarify classification labels, a decision format makes policy results testable, tool schemas specify callable actions and arguments, and retrieval plus an abstention rule grounds answers in available evidence. The tool-calling check is limited to verifying the call and locally printing the fake result; it does not test a second model response.

## Commands and observed results

Commands were run from `assignments/03-anatomy-of-coding-agents/prompting-techniques/starter/`. Final check output:

```text
--- k_shot_prompting.py ---
[PASS] 'My card was charged twice for order 4412.' -> got 'billing', expected 'billing'
[PASS] "Where is my package? It's been two weeks." -> got 'shipping', expected 'shipping'
[PASS] 'I want to send the shoes back for a refund.' -> got 'returns', expected 'returns'
[PASS] 'The app crashes every time I open my cart.' -> got 'technical', expected 'technical'
4/4 correct
--- chain_of_thought.py ---
[PASS] 'Bought 10 days ago, unused, not final sale.' -> YES, expected YES
[PASS] 'Bought 45 days ago, unused.' -> NO, expected NO
[PASS] 'Bought 5 days ago but it was a final-sale item.' -> NO, expected NO
3/3 correct
--- tool_calling.py ---
[PASS] tool called with order_id='4412'
tool result: shipped, arriving Friday
--- rag.py ---
[PASS] 'How long do I have to return an item?' -> 'you may return most items within 30 days of delivery for a full refund.'
[PASS] 'Do you ship internationally?' -> 'yes, we offer international shipping to most countries.'
[PASS] "What's your CEO's home address?" -> "i don't know based on the help-center information provided."
3/3 correct
```
