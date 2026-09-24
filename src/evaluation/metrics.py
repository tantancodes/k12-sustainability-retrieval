def recall_at_k(results, relevant_doc_ids, k):
    """
    Measure what fraction of relevant documents
    appear in the top-k retrieved results.
    """

    top_k_ids = [
        result["doc_id"]
        for result in results[:k]
    ]

    relevant_doc_ids = set(relevant_doc_ids)

    found = relevant_doc_ids.intersection(top_k_ids)

    return len(found) / len(relevant_doc_ids)


def reciprocal_rank(results, relevant_doc_ids):
    """
    Return 1 / rank of the first relevant document.

    Rank 1 -> 1.0
    Rank 2 -> 0.5
    Rank 3 -> 0.333...
    """

    relevant_doc_ids = set(relevant_doc_ids)

    for rank, result in enumerate(results, start=1):

        if result["doc_id"] in relevant_doc_ids:
            return 1 / rank

    return 0.0

def mean_reciprocal_rank(all_results, all_relevant_doc_ids):
    """
    Compute Mean Reciprocal Rank across multiple queries.
    """

    reciprocal_ranks = []

    for results, relevant_doc_ids in zip(
        all_results,
        all_relevant_doc_ids
    ):
        rr = reciprocal_rank(results, relevant_doc_ids)
        reciprocal_ranks.append(rr)

    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def mean_recall_at_k(all_results, all_relevant_doc_ids, k):
    """
    Compute average Recall@K across multiple queries.
    """

    recalls = []

    for results, relevant_doc_ids in zip(
        all_results,
        all_relevant_doc_ids
    ):
        recall = recall_at_k(
            results,
            relevant_doc_ids,
            k
        )

        recalls.append(recall)

    if not recalls:
        return 0.0

    return sum(recalls) / len(recalls)