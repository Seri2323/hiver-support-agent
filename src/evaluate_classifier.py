import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

TRAIN_LABELS = "data/gold/taxonomy_check_50.csv"
TEST_DATA = "data/gold/golden_set_200_frozen.csv"

train = pd.read_csv(TRAIN_LABELS)
test = pd.read_csv(TEST_DATA)

X_train = train["customer_message"].astype(str)
y_train = train["intent"].astype(str)

X_test = test["customer_message"].astype(str)
y_test = test["intent"].astype(str)

results = {}

# ---------------------------------------------------------
# Baseline 1: majority class
# ---------------------------------------------------------

majority_class = y_train.value_counts().idxmax()
majority_predictions = [majority_class] * len(y_test)

results["majority"] = {
    "accuracy": accuracy_score(y_test, majority_predictions)
}

print("=" * 60)
print("BASELINE 1 — MAJORITY CLASS")
print("=" * 60)

print("Majority class:", majority_class)
print("Accuracy:", round(results["majority"]["accuracy"], 3))

# ---------------------------------------------------------
# Baseline 2: TF-IDF + Logistic Regression
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    ),
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

results["tfidf_logreg"] = {
    "accuracy": accuracy_score(y_test, predictions)
}

print("\n" + "=" * 60)
print("BASELINE 2 — TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print("Accuracy:", round(results["tfidf_logreg"]["accuracy"], 3))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)