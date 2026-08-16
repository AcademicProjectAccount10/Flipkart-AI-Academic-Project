# Part 3 Tasks 7--8: deterministic mock responses and guardrails

`src/part3/support_graph.py` runs with `MOCK_LLM_MODE = True`. Its response
composer is a rule-based local function, not an API client: it uses the highest
scoring locally retrieved policy sentence for policy questions, and formats the
real saved-model result for return-risk and image-classification questions. It
does not read API keys or make outbound network calls.

Input is blocked before RAG retrieval or tool calling when it contains one of
these case-insensitive prompt-injection patterns: `ignore previous
instructions`, `ignore all rules`, or `pretend you are`. The agent returns the
fixed JSON schema with a concise deflection instead of complying.

For a policy answer, the top normalized all-MiniLM-L6-v2 cosine similarity must
be at least **0.35**. The graph records the score, threshold, and pass/fail
state. If the threshold is not met, it returns a `policy_kb` JSON response with
confidence `0.0` and refuses to invent a policy answer.

No Task 9 transcripts or Task 10 retrieval-evaluation artifacts are included
in this task.
