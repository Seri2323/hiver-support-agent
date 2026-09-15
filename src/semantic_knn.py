import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report

TRAIN_LABELS = "data/gold/taxonomy_check_50.csv"
TEST_DATA = "data/gold/golden_set_200_frozen.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

train = pd.read_csv(TRAIN_LABELS)
test = pd.read_csv(TEST_DATA)

train_texts = train["customer_message"].astype(str).tolist()
train_labels = train["intent"].astype(str).tolist()

test_texts = test["customer_message"].astype(str).tolist()
test_labels = test["intent"].astype(str).tolist()

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")

train_embeddings = model.encode(
    train_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

test_embeddings = model.encode(
    test_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

def predict_knn(test_embedding, k):
    similarities = np.dot(train_embeddings, test_embedding)

    top_indices = np.argsort(similarities)[-k:][::-1]

    # Weighted voting: more similar examples get more weight
    scores = {}

    for idx in top_indices:
        label = train_labels[idx]
        similarity = max(similarities[idx], 0.0)

        scores[label] = scores.get(label, 0.0) + similarity

    return max(scores, key=scores.get)

for k in [3, 5]:

    predictions = [
        predict_knn(embedding, k)
        for embedding in test_embeddings
    ]

    print("\n" + "=" * 60)
    print(f"SEMANTIC KNN — TOP {k}")
    print("=" * 60)

    print(
        "Accuracy:",
        round(accuracy_score(test_labels, predictions), 3)
    )

    print(
        classification_report(
            test_labels,
            predictions,
            zero_division=0,
        )
    )