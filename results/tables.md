## Main comparison

| Condition | Contains | EM | F1 | Quote rate | Grounded | Fabricated | Gold cited | Verified correct | Tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 Base, closed-book | 42.6 | 28.0 | 39.8 | 39.6 | 0.0 | 39.6 | 0.0 | 0.0 | 30 | 0.37 |
| C2 Base + RAG | 69.2 | 38.0 | 50.7 | 8.6 | 3.6 | 5.0 | 2.0 | 1.4 | 28 | 0.63 |
| C3 RAFT, closed-book | 34.0 | 30.0 | 36.6 | 0.6 | 0.0 | 0.6 | 0.0 | 0.0 | 45 | 0.79 |
| C4 RAFT + RAG | 72.4 | 67.8 | 74.4 | 99.6 | 99.6 | 0.0 | 47.8 | 40.4 | 54 | 0.87 |
| C5 RAFT, oracle context | 74.6 | 69.8 | 77.1 | 99.8 | 99.8 | 0.0 | 68.4 | 57.6 | 54 | 0.87 |

## Ablations

| Condition | Contains | Verified correct | Gold cited | Distractor cited | Fabricated |
|---|---|---|---|---|---|
| C4 RAFT + RAG | 72.4 | 40.4 | 47.8 | 51.8 | 0.0 |
| A1 SFT control (gold only) | 76.0 | 50.0 | 59.2 | 39.8 | 0.8 |
| A2 RAFT without CoT | 78.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| A3 RAFT, random distractors | 76.6 | 48.2 | 56.8 | 42.0 | 1.0 |

## Retrieval success split

| Condition | n (gold retrieved) | Contains | n (gold missing) | Contains |
|---|---|---|---|---|
| C2 Base + RAG | 429 | 72.7 | 71 | 47.9 |
| C4 RAFT + RAG | 429 | 77.4 | 71 | 42.2 |
| C5 RAFT, oracle context | 500 | 74.6 | 0 | — |

## Top-k sweep

| k | Contains | Grounded | Gold cited | Verified correct | Fabricated | Tokens |
|---|---|---|---|---|---|---|
| 1 | 69.8 | 88.4 | 57.2 | 47.6 | 4.6 | 51 |
| 3 | 72.6 | 99.0 | 53.6 | 43.8 | 0.6 | 52 |
| 8 | 68.6 | 92.2 | 43.8 | 37.4 | 0.6 | 58 |

## Learner profile

| Profile | Contains | Avg output tokens | Verified correct |
|---|---|---|---|
| advanced | 72.2 | 54 | 38.4 |
| novice | 72.0 | 54 | 39.6 |