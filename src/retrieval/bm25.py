import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    """
    Convert text into lowercase word tokens.

    Example:
    "Electric School Buses!" -> ["electric", "school", "buses"]
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, documents):
        """
        documents should look like:

        [
            {"doc_id": "doc1", "text": "..."},
            {"doc_id": "doc2", "text": "..."},
        ]
        """

        self.documents = documents

        tokenized_documents = [
            tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(self, query, k=10):
        """
        Search the documents and return the top-k results.
        """

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results = []

        for i in ranked_indices[:k]:
            results.append({
                "doc_id": self.documents[i]["doc_id"],
                "text": self.documents[i]["text"],
                "score": float(scores[i])
            })

        return results