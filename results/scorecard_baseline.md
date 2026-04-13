# Scorecard: baseline_dense
Generated: 2026-04-13 15:56

## Summary — Simple LLM-as-Judge (gpt-4o-mini, thang 1–5)

| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.70/5 |
| Relevance | 4.80/5 |
| Context Recall | 5.00/5 |
| Completeness | 4.20/5 |

## Per-Question Results

| ID | Category | Faithful | Relevant | Recall | Complete | Notes |
|----|----------|----------|----------|--------|----------|-------|
| q01 | SLA | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q02 | Refund | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q03 | Access Control | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q04 | Refund | 4 | 5 | 5 | 3 | The answer accurately reflects the refund policy for digital |
| q05 | IT Helpdesk | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q06 | SLA | 4 | 5 | 5 | 5 | The answer accurately describes the escalation process for P |
| q07 | Access Control | 5 | 5 | 5 | 3 | Every claim in the answer is directly supported by the retri |
| q08 | HR Policy | 5 | 5 | 5 | 5 | The answer accurately reflects the retrieved context regardi |
| q09 | Insufficient Context | 4 | 5 | None | 3 | The answer is mostly grounded in the retrieved context, but  |
| q10 | Refund | 5 | 3 | 5 | 3 | The answer accurately reflects that there is no information  |

## Config

```
label          : baseline_dense
retrieval_mode : baseline_dense
```
