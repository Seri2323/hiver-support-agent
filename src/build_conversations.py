import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/processed/apple_conversations.csv"

df = pd.read_csv(DATA_PATH)

# AppleSupport tweets
apple = df[df["author_id"] == "AppleSupport"].copy()

# Map each AppleSupport tweet ID to its response text
apple_responses = dict(
    zip(
        apple["tweet_id"].astype(str),
        apple["text"].astype(str)
    )
)

# Customer tweets that have a recorded response
customer = df[
    (df["inbound"] == True) &
    (df["response_tweet_id"].notna())
].copy()

customer["response_tweet_id"] = customer["response_tweet_id"].astype(str)

# Keep only conversations where the response is from AppleSupport
customer["response"] = customer["response_tweet_id"].map(apple_responses)

customer = customer[customer["response"].notna()].copy()

# Keep the useful columns
customer = customer[
    ["tweet_id", "created_at", "text", "response"]
]

customer = customer.rename(
    columns={
        "text": "customer_message"
    }
)

customer.to_csv(OUTPUT_PATH, index=False)

print("Conversation dataset created!")
print(f"Rows: {len(customer):,}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nExample conversation:")
print("\nCUSTOMER:")
print(customer.iloc[0]["customer_message"])
print("\nAPPLE SUPPORT:")
print(customer.iloc[0]["response"])