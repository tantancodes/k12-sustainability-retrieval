from src.evaluation.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
)


all_results = [
    # Query 1: correct document is rank 1
    [
        {"doc_id": "bus"},
        {"doc_id": "solar"},
        {"doc_id": "garden"},
    ],

    # Query 2: correct document is rank 2
    [
        {"doc_id": "math"},
        {"doc_id": "garden"},
        {"doc_id": "solar"},
    ],

    # Query 3: correct document is rank 3
    [
        {"doc_id": "bus"},
        {"doc_id": "garden"},
        {"doc_id": "solar"},
    ],
]


all_relevant = [
    {"bus"},
    {"garden"},
    {"solar"},
]


print(
    "Recall@1:",
    mean_recall_at_k(
        all_results,
        all_relevant,
        1
    )
)

print(
    "Recall@3:",
    mean_recall_at_k(
        all_results,
        all_relevant,
        3
    )
)

print(
    "MRR:",
    mean_reciprocal_rank(
        all_results,
        all_relevant
    )
)