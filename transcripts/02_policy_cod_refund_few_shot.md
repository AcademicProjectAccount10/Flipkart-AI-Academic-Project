# Policy conversation: COD refund timeline

Mode: `MOCK_LLM_MODE=True` (local deterministic mode; no API key or live LLM).

## Turn 1
**User:** When will my COD refund arrive after the pickup?

**Intent route:** `policy`

**Routing evidence:** Applied few-shot routing example: User: "When will my COD refund arrive?" → Intent: policy

**Retrieved policy chunks:**
- `POL-06-S3` → `POL-06`; similarity `0.5006`
- `POL-05-S1` → `POL-05`; similarity `0.4922`
- `POL-06-S1` → `POL-06`; similarity `0.4762`

**Groundedness check:** top similarity `0.5006` vs threshold `0.35` → `pass`.

**Assistant JSON response:**
```json
{
  "answer": "According to the policy knowledge base: For a successful cash-on-delivery return, the refund is initiated after the returned item passes the seller or warehouse quality check.",
  "source": "policy_kb",
  "confidence": 0.5006
}
```
