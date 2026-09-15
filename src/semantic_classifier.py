import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

TRAIN_LABELS = "data/gold/taxonomy_check_50.csv"
TEST_DATA = "data/gold/golden_set_200_frozen.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

# Load labeled examples
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

print("Running nearest-example classification...")

predictions = []

for test_embedding in test_embeddings:
    # Cosine similarity because embeddings are normalized
    scores = np.dot(train_embeddings, test_embedding)

    best_index = np.argmax(scores)

    predictions.append(train_labels[best_index])

print("\n" + "=" * 60)
print("SEMANTIC BASELINE — NEAREST LABELED EXAMPLE")
print("=" * 60)

print("Accuracy:", round(accuracy_score(test_labels, predictions), 3))

print("\nClassification report:")
print(
    classification_report(
        test_labels,
        predictions,
        zero_division=0,
    )
)