# Part 3 — Tasks 5–6 Graph Design

## Nodes and routes

The compiled LangGraph workflow has four nodes: `intent`, `rag_retrieval`, `tool_calling`, and `response_generation`.

`START → intent` branches conditionally: policy requests route to `rag_retrieval`, while return-risk and image-classification requests route to `tool_calling`. Both routes then converge at `response_generation` and finish at `END`.

## Conversation state

`ConversationSession` retains a per-conversation `history`, `last_order_features`, and `last_image_path`. A follow-up can therefore reuse order features or an image path supplied earlier in that same session. `new_conversation_state()` and a newly constructed `ConversationSession` start with an empty history and no retained inputs.

## Prompt structure and response contract

The role-based system prompt is annotated for Specific, Short, Surround, and Single (4S). It requires exactly one JSON object conforming to the `FinalResponse` schema: `answer`, `source`, and `confidence`. The intent prompt includes three few-shot examples, including one policy, one return-risk, and one image-classification request.

Task 7's deterministic response composer and Tasks 8–10 features are intentionally not included in this graph-design stage.
