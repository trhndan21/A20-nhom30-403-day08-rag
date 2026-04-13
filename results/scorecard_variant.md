# Scorecard: variant_hybrid_rerank
Generated: 2026-04-13 15:57

## Summary — Simple LLM-as-Judge (gpt-4o-mini, thang 1–5)

| Metric | Average Score |
|--------|--------------|
| Faithfulness | 4.70/5 |
| Relevance | 4.50/5 |
| Context Recall | 5.00/5 |
| Completeness | 4.20/5 |

## Per-Question Results

| ID | Category | Faithful | Relevant | Recall | Complete | Notes |
|----|----------|----------|----------|--------|----------|-------|
| q01 | SLA | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q02 | Refund | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q03 | Access Control | 5 | 5 | 5 | 5 | The answer accurately reflects the approval requirements for |
| q04 | Refund | 4 | 5 | 5 | 3 | The answer accurately reflects the retrieved context regardi |
| q05 | IT Helpdesk | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q06 | SLA | 5 | 5 | 5 | 5 | Every claim in the answer is directly supported by the retri |
| q07 | Access Control | 5 | 5 | 5 | 3 | Every claim in the answer is directly supported by the retri |
| q08 | HR Policy | 5 | 5 | 5 | 5 | The answer accurately reflects the retrieved context regardi |
| q09 | Insufficient Context | 3 | 4 | None | 3 | The answer provides a general explanation of the ERR-403-AUT |
| q10 | Refund | 5 | 1 | 5 | 3 | The answer accurately reflects that there is no information  |

## Config

```
label          : variant_hybrid_rerank
retrieval_mode : variant_hybrid_rerank
```
