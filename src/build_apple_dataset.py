import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/apple_support.csv"

df = pd.read_csv(DATA_PATH)

# AppleSupport's tweets
apple = df[df["author_id"] == "AppleSupport"].copy()

# Customer tweets that received a response
customer = df[
    (df["inbound"] == True) &
    (df["response_tweet_id"].notna())
].copy()

apple_ids = set(apple["tweet_id"].astype(str))

def connected_to_apple(value):
    if pd.isna(value):
        return False

    response_ids = str(value).split(",")

    return any(
        x.strip() in apple_ids
        for x in response_ids
    )

customer = customer[
    customer["response_tweet_id"].apply(connected_to_apple)
].copy()

customer = customer[
    ["tweet_id", "author_id", "created_at", "text", "response_tweet_id"]
]

customer.to_csv(OUTPUT_PATH, index=False)

print("Apple dataset created!")
print(f"Rows: {len(customer):,}")
print(f"Saved to: {OUTPUT_PATH}")