import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Keep tweet IDs consistent for matching
df["tweet_id"] = df["tweet_id"].astype(str)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print(f"Total tweets: {len(df):,}")
print(f"Unique authors: {df['author_id'].nunique():,}")

# ---------------------------------------------------------
# 2. Identify company/support tweets
# ---------------------------------------------------------

company_tweets = df[df["inbound"] == False]

brand_counts = company_tweets["author_id"].value_counts().head(20)

print("\nTOP 20 SUPPORT ACCOUNTS")
print("-" * 70)

for brand, count in brand_counts.items():
    print(f"{brand:25} {count:,}")

# ---------------------------------------------------------
# 3. Build a lookup table for tweets
# ---------------------------------------------------------

tweet_lookup = df.set_index("tweet_id")

# These are our current candidate brands
brands = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
]

# ---------------------------------------------------------
# 4. Inspect real customer -> company exchanges
# ---------------------------------------------------------

print("\n\n")
print("=" * 70)
print("REAL CUSTOMER -> COMPANY EXCHANGES")
print("=" * 70)

for brand in brands:

    # IDs of tweets written by this company
    company_ids = set(
        company_tweets.loc[
            company_tweets["author_id"] == brand,
            "tweet_id"
        ]
    )

    # Customer tweets whose response_tweet_id contains
    # one of this company's tweets
    customer_tweets = df[df["inbound"] == True].copy()

    def has_company_response(value):
        if pd.isna(value):
            return False

        response_ids = str(value).split(",")

        return any(
            response_id.strip() in company_ids
            for response_id in response_ids
        )

    candidates = customer_tweets[
        customer_tweets["response_tweet_id"].apply(has_company_response)
    ]

    # Pick 5 random examples
    sample = candidates.sample(
        n=min(5, len(candidates)),
        random_state=42
    )

    print("\n")
    print("#" * 70)
    print(f"{brand}")
    print("#" * 70)

    for i, (_, customer) in enumerate(sample.iterrows(), 1):

        print(f"\n--- Example {i} ---")

        print("\nCUSTOMER:")
        print(customer["text"])

        response_ids = str(customer["response_tweet_id"]).split(",")

        for response_id in response_ids:

            response_id = response_id.strip()

            if response_id in company_ids:

                response = tweet_lookup.loc[response_id]

                print("\nCOMPANY:")
                print(response["text"])

                break