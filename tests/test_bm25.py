from src.retrieval.bm25 import BM25Retriever
from src.evaluation.metrics import recall_at_k, reciprocal_rank


documents = [
    {
        "doc_id": "bus",
        "text": "The district purchased electric school buses for its transportation fleet."
    },
    {
        "doc_id": "garden",
        "text": "Students planted California native plants in the school garden."
    },
    {
        "doc_id": "solar",
        "text": "Solar panels were installed on several school buildings."
    },
    {
        "doc_id": "math",
        "text": "The district adopted a new mathematics curriculum."
    }
]


retriever = BM25Retriever(documents)

results = retriever.search(
    "electric school bus fleet",
    k=4
)


print("RANKING")

for rank, result in enumerate(results, start=1):
    print(
        rank,
        result["doc_id"],
        round(result["score"], 3)
    )


relevant = {"bus"}

print()
print("EVALUATION")
print("Recall@1:", recall_at_k(results, relevant, 1))
print("Recall@3:", recall_at_k(results, relevant, 3))
print("Reciprocal Rank:", reciprocal_rank(results, relevant))