# Part 3 Task 10: document-level retrieval evaluation

The local FAISS index is searched over all sentence chunks. Chunks are mapped to their `parent_document_id` and deduplicated in rank order before the first three unique documents are scored. This makes both metrics document-level P@3 and R@3.

| Query | Relevant documents | Retrieved unique documents @3 | Hits | Precision@3 arithmetic | Recall@3 arithmetic |
|---|---|---|---:|---:|---:|
| Q1 | POL-01 | POL-01, POL-03, POL-02 | 1 | 1/3 = 0.3333 | 1/1 = 1.0000 |
| Q2 | POL-05, POL-10 | POL-06, POL-05, POL-10 | 2 | 2/3 = 0.6667 | 2/2 = 1.0000 |
| Q3 | POL-02 | POL-02, POL-04, POL-08 | 1 | 1/3 = 0.3333 | 1/1 = 1.0000 |
| Q4 | POL-09 | POL-09, POL-10, POL-05 | 1 | 1/3 = 0.3333 | 1/1 = 1.0000 |
| Q5 | POL-07 | POL-07, POL-08, POL-09 | 1 | 1/3 = 0.3333 | 1/1 = 1.0000 |
| Q6 | POL-04 | POL-04, POL-12, POL-03 | 1 | 1/3 = 0.3333 | 1/1 = 1.0000 |

**Average Precision@3:** (0.3333 + 0.6667 + 0.3333 + 0.3333 + 0.3333 + 0.3333) / 6 = **0.3889**

**Average Recall@3:** (1.0000 + 1.0000 + 1.0000 + 1.0000 + 1.0000 + 1.0000) / 6 = **1.0000**
