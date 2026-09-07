# Sampling strategies

We explore various negative mining strategies for enhancing model performance. 
The scripts included in this directory implement different strategies for mining negatives:

1. `hn-from-coverage.py`: Mines negatives based on coverage (per-document). We first binarize the multi-asepct judge into binary labels, then select the documents based on the coverage metric:
    - **half-coverage (0.5 <= cov < 0.75)**,
    - **low-coverage (0.0 < cov < 0.25)**, 
    - **zero-coverage (cov == 0.0)**

2. `hn-from-rank-ratio.py`: Mines negatives based on rank ratio: log RR(d)/R(d). 
As the reranked ranking (i.e., RR(d)) is supposed to be more accurate than the initial retrieval ranking (i.e., R(d)). We compute the rank ratio and select documents with high log rank ratio as negatives. It means the document ranked higher (smaller) in the initial retrieval but ranked lower (larger) in the reranked ranking.

3. `hn-mining-from-qwen3-rerank`: Mine negatives using Qwen-3 model reranking. We directly select the documents ranked from 50 - 100 as negative.

4. `hn-mining-from-qwen3-0.6b.py`: Mine negatives using Qwen-3-0.6B model. We re-encode the 3M clueweb documents using Qwen-3-0.6B and use it to mine negatives following the same procedure as in single-view negative mining.

## Coverage-Based Sampling Methods (`sample-from-coverage.py`)

This script implements various positive/negative sampling strategies based on coverage metrics. Coverage is defined as the fraction of answerable subtopics that a document addresses (with rating >= tau threshold).

### Coverage Buckets

Documents from the top 20 results are classified into buckets based on their coverage score `c`:
- **high (0.75)**: c >= 0.75 (covers at least 75% of answerable subtopics)
- **half (0.5)**: 0.5 <= c < 0.75 (covers 50% to <75% of answerable subtopics)
- **quarter (0.25)**: 0.25 <= c < 0.5 (covers 25% to <50% of answerable subtopics)
- **low (0.0)**: 0 <= c < 0.25 (covers <25% of answerable subtopics)
- **zero (-1)**: c == 0 exactly (covers no answerable subtopics)
- **unjudged (-2)**: Documents ranked after position 50 (not judged)

### Sampling Strategies

#### 1. Single-list Sampling (`pos_20.neg_51`)
- **Positives**: Top 20 documents from the ranking
- **Negatives**: Documents ranked 51 and beyond
- Simple rank-based approach without coverage filtering

#### 2. Single-list Sampling with Coverage Filtering (`pos_20.neg_51.filtered`)
- **Positives**: Top 20 documents, excluding those with zero coverage
- **Negatives**: Documents ranked 51 and beyond
- Filters out documents that don't answer any subtopic

#### 3. Coverage-based Sampling Pairs
The following pairs sample positives and negatives based on coverage thresholds:

| Strategy | Positive Selection | Negative Selection |
|----------|-------------------|-------------------|
| `pos_high.neg_zero` | High coverage (c >= 0.75) | Zero coverage (c == 0) |
| `pos_high.neg_low` | High coverage (c >= 0.75) | Low coverage (0 <= c < 0.25) |
| `pos_high.neg_quarter` | High coverage (c >= 0.75) | Quarter + Low coverage (0 <= c < 0.5) |
| `pos_high.neg_half` | High coverage (c >= 0.75) | Half + Quarter + Low coverage (0 <= c < 0.75) |
| `pos_half.neg_zero` | Half coverage (0.5 <= c < 0.75) | Zero coverage (c == 0) |
| `pos_low.neg_zero` | Low coverage (0 <= c < 0.25) | Zero coverage (c == 0) |

All coverage-based strategies pad negatives with unjudged documents (ranked 51+) up to 16 total negatives.

### Known Issues / Potential Bugs (Fixed)

1. **Environment Variable Handling (Line 21)**: ~~Uses `os.environ["CRUX_ROOT"]` which raises `KeyError` if not set.~~ **Fixed**: Now uses `os.environ.get("CRUX_ROOT", '/exp/scale25/artifacts/crux')`.

2. **Division by Zero (Line 38)**: ~~When `n_ans == 0` (no answerable subtopics), the coverage calculation `c = sum(...) / n_ans` will raise a `ZeroDivisionError`.~~ **Fixed**: Now checks `if n_ans > 0 else 0`.

3. **Ambiguous 'low' Category**: The `pos_low` category includes documents with coverage >= 0.0, which also includes zero-coverage documents. This overlap with `neg_zero` may be intentional for contrastive learning but could be confusing.
