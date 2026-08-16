# Policy conversation: footwear return window

Mode: `MOCK_LLM_MODE=True` (local deterministic mode; no API key or live LLM).

## Turn 1
**User:** How many days do I have to return shoes that do not fit?

**Intent route:** `policy`

**Routing evidence:** Applied few-shot routing example: User: "When will my COD refund arrive?" → Intent: policy

**Retrieved policy chunks:**
- `POL-01-S1` → `POL-01`; similarity `0.6199`
- `POL-03-S1` → `POL-03`; similarity `0.5106`
- `POL-02-S1` → `POL-02`; similarity `0.4624`

**Groundedness check:** top similarity `0.6199` vs threshold `0.35` → `pass`.

**Assistant JSON response:**
```json
{
  "answer": "According to the policy knowledge base: Apparel and footwear items may be returned within 10 days of delivery when they are unused, unwashed, and returned with original tags.",
  "source": "policy_kb",
  "confidence": 0.6199
}
```
