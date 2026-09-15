import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer


TRAIN_DATA = "data/processed/train.csv"
EMBEDDINGS = "data/processed/train_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


class AppleRetriever:

    def __init__(self):
        print("Loading retrieval data...")

        self.df = pd.read_csv(TRAIN_DATA)

        self.df["customer_message"] = (
            self.df["customer_message"].astype(str)
        )

        self.df["response"] = (
            self.df["response"].astype(str)
        )

        self.embeddings = np.load(EMBEDDINGS)

        print(f"Loaded {len(self.df):,} historical conversations.")

        self.model = SentenceTransformer(MODEL_NAME)

    def retrieve(self, customer_message, k=3):

        query_embedding = self.model.encode(
            [customer_message],
            normalize_embeddings=True,
        )[0]

        similarities = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []

        for idx in top_indices:

            results.append({
                "customer_message": self.df.iloc[idx]["customer_message"],
                "response": self.df.iloc[idx]["response"],
                "thread_id": self.df.iloc[idx]["thread_id"],
                "similarity": float(similarities[idx]),
            })

        return results


if __name__ == "__main__":

    retriever = AppleRetriever()

    query = (
        "My iPhone battery is draining very quickly "
        "after the latest update."
    )

    print("\nQUERY:")
    print(query)

    print("\nTOP HISTORICAL EXAMPLES:")

    results = retriever.retrieve(query, k=3)

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 70)
        print(f"RESULT {i}")
        print("=" * 70)

        print("Similarity:", round(result["similarity"], 3))

        print("\nCUSTOMER:")
        print(result["customer_message"])

        print("\nAPPLE RESPONSE:")
        print(result["response"])