# Ungrounded policy refusal conversation

Mode: `MOCK_LLM_MODE=True` (local deterministic mode; no API key or live LLM).

## Turn 1
**User:** What is the executive bonus policy?

**Intent route:** `policy`

**Routing evidence:** Applied few-shot routing example: User: "When will my COD refund arrive?" → Intent: policy

**Retrieved policy chunks:**
- `POL-12-S3` → `POL-12`; similarity `0.2644`
- `POL-05-S2` → `POL-05`; similarity `0.1734`
- `POL-06-S3` → `POL-06`; similarity `0.1630`

**Groundedness check:** top similarity `0.2644` vs threshold `0.35` → `refuse`.

**Assistant JSON response:**
```json
{
  "answer": "I’m sorry, but I do not have sufficiently grounded Flipkart policy information to answer that question.",
  "source": "policy_kb",
  "confidence": 0.0
}
```
