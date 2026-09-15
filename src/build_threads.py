import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/apple_threads.csv"

df = pd.read_csv(DATA_PATH)

# Convert all tweet IDs to strings immediately.
# This prevents 696 and "696" from being treated as different IDs.
df["tweet_id"] = df["tweet_id"].astype(str)

df["in_response_to_tweet_id"] = (
    df["in_response_to_tweet_id"]
    .fillna("")
    .astype(str)
)

# Map each tweet -> parent tweet
parent_map = dict(
    zip(
        df["tweet_id"],
        df["in_response_to_tweet_id"]
    )
)

def find_root(tweet_id):
    current = str(tweet_id)
    seen = set()

    while current in parent_map:
        parent = parent_map[current]

        if not parent or parent == "nan" or parent in seen:
            break

        seen.add(current)
        current = parent

    return current


# ---------------------------------------------------------
# Apple customer messages that received an Apple response
# ---------------------------------------------------------

apple_outbound = df[
    (df["author_id"] == "AppleSupport") &
    (df["inbound"] == False)
].copy()

apple_ids = set(
    apple_outbound["tweet_id"]
)

customer = df[
    (df["inbound"] == True) &
    (df["response_tweet_id"].notna())
].copy()

customer["response_tweet_id"] = (
    customer["response_tweet_id"].astype(str)
)

def has_apple_response(value):
    return any(
        x.strip() in apple_ids
        for x in str(value).split(",")
    )

customer = customer[
    customer["response_tweet_id"].apply(has_apple_response)
].copy()

# ---------------------------------------------------------
# Build stable conversation/thread IDs
# ---------------------------------------------------------

customer["thread_id"] = customer["tweet_id"].apply(find_root)

customer = customer[
    [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "response_tweet_id",
        "thread_id",
    ]
]

customer.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Apple thread dataset created!")
print(f"Customer messages: {len(customer):,}")
print(f"Unique threads: {customer['thread_id'].nunique():,}")
print(f"Thread ID type: {customer['thread_id'].map(type).value_counts().to_dict()}")