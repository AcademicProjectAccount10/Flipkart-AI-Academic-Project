# Fresh-conversation reset example

Mode: `MOCK_LLM_MODE=True` (local deterministic mode; no API key or live LLM).

**Conversation-state evidence:** This is a new session, so it has no `last_order_features`; unlike transcript 05 it asks for order features instead of scoring an inherited order.

## Turn 1
**User:** Is it likely to be returned?

**Intent route:** `return_risk`

**Routing evidence:** Matched the return-risk wording from the few-shot routing example; the fresh session correctly has no order features to score.

**Real tool output:**
```json
{
  "error": "Return-risk scoring needs order features in this turn or an earlier turn of this conversation."
}
```

**Assistant JSON response:**
```json
{
  "answer": "Return-risk scoring needs order features in this turn or an earlier turn of this conversation.",
  "source": "return_risk_tool",
  "confidence": 0.0
}
```
