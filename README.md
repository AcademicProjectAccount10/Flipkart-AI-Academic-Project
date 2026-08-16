# Flipkart Order Intelligence & Support Assistant

This repository combines three local components: a return-risk model, a
Fashion-MNIST product-image classifier, and a LangGraph support assistant that
uses both models plus a grounded policy knowledge base.

All commands below are PowerShell commands and must be run from the repository
root. No API key is required for the default Part 3 mock mode.

## Setup for a fresh Windows clone

Install a current CPython version supported by PyTorch, then create a local
virtual environment and install the pinned direct dependencies. The `.venv`
folder is intentionally local and is not committed. If `python` is not on your
PATH, substitute the full path to the desired Python executable in the commands.

```powershell
# Create the environment once after cloning.
python -m venv .venv

# Install the dependencies declared by this repository.
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install -r .\requirements.txt
```

After setup, use the same `$PY` variable for every command below.

## Quick verification of the committed implementation

These commands load or inspect the existing artifacts; they do not retrain a
model. They are the fastest way to validate the committed project.

```powershell
& $PY .\src\part1\analyze_generated_orders.py
& $PY .\src\part2\verify_saved_product_classifier.py
& $PY .\src\part3\verify_model_tools.py
```

Repository navigation:

- `src/part1`, `src/part2`, `src/part3`: implementation and reproducible workflows
- `models/`: the fitted Random Forest and ResNet-18 checkpoint used by Part 3
- `outputs/`: metrics, reports, confusion matrix, and retrieval evaluation
- `data/sample_images/`: five verified Fashion-MNIST official-test PNGs
- `kb/`, `data/part3_policy_index/`, and `transcripts/`: Part 3 knowledge-base and agent evidence

## Part 1 - return-risk scoring

The exact seeded generator is [generate_orders.py](generate_orders.py). It
creates `orders_dataset.csv`; the data analysis verifies 6,000 rows and 13
columns. The training scripts save reports under `outputs/` and the final
preprocessing-plus-Random-Forest pipeline at `models/return_risk_model.pkl`.

```powershell
# Regenerates the deterministic dataset and retrains the Part 1 models.
& $PY .\generate_orders.py
& $PY .\src\part1\analyze_generated_orders.py
& $PY .\src\part1\train_tasks_3_to_5.py
& $PY .\src\part1\train_tasks_6_to_9.py
```

Current data verification: return rate 22.75%; `rating_given` missingness
13.05%; and missingness is MAR because it depends on the observed
`payment_method` column (COD 22.83% vs non-COD 6.06%). The saved Random Forest
uses `t*_rf = 0.50`; its low/medium/high cut points are below 0.50,
0.50-0.65, and at least 0.65.

Key reports:

- [Tasks 3-5 report](outputs/part1_tasks_3_5_report.md)
- [Tasks 6-9 report](outputs/part1_tasks_6_9_report.md)
- [Random Forest model metrics](outputs/part1_tasks_6_9_model_metrics.csv)

## Part 2 - Fashion-MNIST image classifier

Part 2 uses the official Fashion-MNIST dataset through
`torchvision.datasets.FashionMNIST(..., download=True)`. It creates a 55,000 /
5,000 stratified training/validation split from the official training split and
keeps the official 10,000-image test split untouched for final evaluation.

The chosen transfer-learning backbone is ImageNet-pretrained ResNet-18. Each
grayscale image is replicated to three channels, resized to 224 x 224, and
normalized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
`(0.229, 0.224, 0.225)`.

```powershell
# Download/load the data, create the stratified split, and write the setup report.
& $PY .\src\part2\setup_fashion_mnist.py

# Recreate the Tasks 3-6 feature-extraction training/evaluation reports.
& $PY .\src\part2\train_tasks_3_to_6.py

# Optional artifact-recovery workflow: regenerates the persisted checkpoint and
# five verified official-test PNGs. It repeats cached feature extraction and
# head-only training, so it is not needed merely to inspect the committed model.
& $PY .\src\part2\recover_and_complete_tasks_7_8.py

# Load the checkpoint in a fresh process and run single-image inference.
& $PY .\src\part2\verify_saved_product_classifier.py
```

The saved checkpoint is `models/product_classifier.pt`; its reusable loading
and prediction functions are in [product_classifier.py](src/part2/product_classifier.py).
The final held-out test accuracy is 88.38%. Five pixel-verified official test
samples are in `data/sample_images/`.

Key reports:

- [Tasks 1-2 setup](outputs/part2_tasks_1_2_setup_report.md)
- [Tasks 3-6 training summary](outputs/part2_tasks_3_6_training_summary.md)
- [Tasks 7-8 artifact guide](outputs/part2_tasks_7_8_artifact_guide.md)

## Part 3 - grounded support agent

The agent uses 12 authored Flipkart-style policy documents, sentence chunks,
the local `sentence-transformers/all-MiniLM-L6-v2` embeddings, and a FAISS
`IndexFlatIP` index. Its two model tools load the real saved Part 1 and Part 2
artifacts. `MOCK_LLM_MODE=True` is the default: it is deterministic, requires
zero API keys, and does not use a live LLM or outbound network calls at runtime.

```powershell
# Rebuild the local policy chunks and FAISS index, if needed.
& $PY .\src\part3\build_policy_index.py

# Verify both real saved-model tools against direct model calls.
& $PY .\src\part3\verify_model_tools.py

# Run the default MOCK_LLM agent suite and write the Task 9 transcripts plus Task 10 evaluation.
& $PY .\src\part3\run_tasks_9_10.py
```

`build_policy_index.py` is needed only when rebuilding the committed local
index. The normal mock-agent and verification paths load the existing index
locally; they do not use a live LLM or require an API key.

For a single local mock-agent turn:

```powershell
& $PY -c "import sys; sys.path.insert(0, 'src/part3'); from support_graph import ConversationSession; print(ConversationSession('demo').prepare_turn('How many days do I have to return shoes that do not fit?')['final_response'])"
```

### Full MOCK_LLM example

User: `How many days do I have to return shoes that do not fit?`

- Route: `policy` (via the policy few-shot routing example)
- Top retrieved policy chunk: `POL-01-S1`, similarity `0.6199`
- Groundedness check: `0.6199 >= 0.35`, so the answer is permitted

```json
{
  "answer": "According to the policy knowledge base: Apparel and footwear items may be returned within 10 days of delivery when they are unused, unwashed, and returned with original tags.",
  "source": "policy_kb",
  "confidence": 0.6199
}
```

The complete recorded version is [the footwear-return transcript](transcripts/01_policy_footwear_return.md).

### Graded MOCK_LLM transcripts

- [Policy: footwear return window](transcripts/01_policy_footwear_return.md)
- [Policy: COD refund timeline and few-shot routing](transcripts/02_policy_cod_refund_few_shot.md)
- [Return-risk tool and few-shot routing](transcripts/03_return_risk_few_shot.md)
- [Image-classification tool](transcripts/04_image_classification_tool.md)
- [Multi-turn state carryover](transcripts/05_multi_turn_state.md)
- [Fresh-conversation state reset](transcripts/06_fresh_conversation_reset.md)
- [Prompt-injection guardrail](transcripts/07_prompt_injection_blocked.md)
- [Ungrounded-policy refusal](transcripts/08_ungrounded_policy_refusal.md)

The document-level six-query retrieval evaluation, including per-query
Precision@3 and Recall@3 arithmetic, is in
[part3_task_10_retrieval_evaluation.md](outputs/part3_task_10_retrieval_evaluation.md).
