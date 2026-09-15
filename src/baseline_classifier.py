import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

TRAIN_LABELS = "data/gold/taxonomy_check_50.csv"
TEST_DATA = "data/gold/golden_set_200_frozen.csv"

# ---------------------------------------------------------
# Load our 50 manually labelled TRAIN examples
# ---------------------------------------------------------

train = pd.read_csv(TRAIN_LABELS)

X_train = train["customer_message"].astype(str)
y_train = train["intent"].astype(str)

# ---------------------------------------------------------
# Load the frozen 200-example evaluation set
# ---------------------------------------------------------

test = pd.read_csv(TEST_DATA)

X_test = test["customer_message"].astype(str)
y_test = test["intent"].astype(str)

# ---------------------------------------------------------
# Simple baseline:
# TF-IDF + Logistic Regression
# ---------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
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

print("Training simple baseline...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n==============================")
print("SIMPLE BASELINE RESULTS")
print("==============================")

print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)