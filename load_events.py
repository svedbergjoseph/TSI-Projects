"""
load_events.py
--------------
Reads auction, click, and impression Parquet files from S3 for today's
date only and loads them into DataFrames

Behavior:
- clicks      -> one combined dataframe
- impressions -> one combined dataframe
- auctions    -> split into ~100 MB dataframe chunks

Usage:
    python -3.14 load_events.py
"""

from dotenv import load_dotenv
load_dotenv()

import io
import os
import sys
from datetime import date
import boto3
import pandas as pd

# Configuration
BUCKET = "tsi-ads360-sftp"

EVENT_PREFIXES = {
    "auctions": (
        "staging/bidder-data/tsq-events/parquet/"
        "ad360_tabular/auctions/"
    ),
    "clicks": (
        "staging/bidder-data/tsq-events/parquet/"
        "ad360_tabular/clicks/"
    ),
    "impressions": (
        "staging/bidder-data/tsq-events/parquet/"
        "ad360_tabular/impressions/"
    ),
}

# Hard stop if total S3 size for a single event type exceeds this.
MAX_S3_BYTES_PER_EVENT = 5 * 1024 ** 3  # 5 GB

# S3 helpers
def build_s3_client() -> boto3.client:
    """
    Build an S3 client using standard boto3 credential resolution.
    """
    return boto3.client("s3")

def list_parquet_keys(
    s3_client,
    bucket: str,
    prefix: str,
    year: str,
    month: str,
    day: str,
) -> tuple[list[str], int]:
    """
    Return (keys, total_bytes) for all parquet files under:

        prefix/year=YYYY/month=MM/day=DD/

    while printing file information during discovery.
    """

    day_prefix = f"{prefix}year={year}/month={month}/day={day}/"

    paginator = s3_client.get_paginator("list_objects_v2")

    pages = paginator.paginate(
        Bucket=bucket,
        Prefix=day_prefix,
    )

    keys = []
    total_bytes = 0

    for page in pages:

        for obj in page.get("Contents", []):

            key = obj["Key"]
            size = obj["Size"]

            if key.endswith(".parquet"):

                keys.append(key)
                total_bytes += size

                print(
                    f"    {key}  "
                    f"({size / 1024 / 1024:.2f} MB)"
                )

    print(
        f"\n  Subtotal: {len(keys)} file(s), "
        f"{total_bytes / 1024 / 1024:.2f} MB on S3"
    )

    print(
        f"  Estimated in-memory size: "
        f"~{total_bytes * 4 / 1024 / 1024:.0f}-"
        f"{total_bytes * 5 / 1024 / 1024:.0f} MB "
        f"(3-5x decompression factor)"
    )

    return keys, total_bytes

def read_parquet_from_s3(
    s3_client,
    bucket: str,
    key: str,
) -> pd.DataFrame:
    """
    Download a single parquet file from S3
    and return it as a dataframe.
    """

    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
    )

    body_bytes = response["Body"].read()

    return pd.read_parquet(io.BytesIO(body_bytes))

# Main loader
def load_event_data():
    """
    Load parquet data for all event types for today's date only.

    Returns:
        (
            auction_dfs,
            df_clicks,
            df_impressions,
        )

    auction_dfs is a list of dataframes split into
    ~100 MB chunks.
    """
    today = date.today()
    year  = today.strftime("%Y")
    month = today.strftime("%m")
    day   = today.strftime("%d")

    print(f"Loading data for today: {today.isoformat()}")

    s3 = build_s3_client()

    dataframes = {}

    for event_name, prefix in EVENT_PREFIXES.items():

        print(
            f"\n[{event_name}] "
            f"Listing keys under year={year}/month={month}/day={day} ..."
        )

        keys, total_bytes = list_parquet_keys(
            s3,
            BUCKET,
            prefix,
            year,
            month,
            day,
        )

        if not keys:

            print(
                "  WARNING: no .parquet files found - "
                "returning empty DataFrame."
            )

            dataframes[event_name] = pd.DataFrame()

            continue

        if total_bytes > MAX_S3_BYTES_PER_EVENT:

            raise ValueError(
                f"[{event_name}] Total S3 size "
                f"{total_bytes / 1024 ** 3:.2f} GB "
                f"exceeds the "
                f"{MAX_S3_BYTES_PER_EVENT / 1024 ** 3:.0f} GB limit."
            )

        print(f"\n  Reading {len(keys)} file(s) into memory ...")

        parts = []

        for i, key in enumerate(keys, start=1):

            print(f"  [{i}/{len(keys)}] {key.split('/')[-1]}")

            df_part = read_parquet_from_s3(
                s3,
                BUCKET,
                key,
            )

            parts.append(df_part)

        combined_df = pd.concat(
            parts,
            ignore_index=True,
        )

        # Split auctions dataframe into ~100 MB chunks
        if event_name == "auctions":

            target_chunk_size_mb = 100

            target_chunk_size_bytes = (
                target_chunk_size_mb
                * 1024
                * 1024
            )

            total_memory = (
                combined_df.memory_usage(deep=True).sum()
            )

            bytes_per_row = total_memory / len(combined_df)

            rows_per_chunk = max(
                1,
                int(target_chunk_size_bytes / bytes_per_row)
            )

            auction_chunks = []

            for start in range(
                0,
                len(combined_df),
                rows_per_chunk,
            ):

                end = start + rows_per_chunk

                chunk_df = (
                    combined_df.iloc[start:end]
                    .reset_index(drop=True)
                )

                auction_chunks.append(chunk_df)

            dataframes[event_name] = auction_chunks

            print(
                f"  Done. Split into "
                f"{len(auction_chunks)} "
                f"auction dataframe chunk(s) "
                f"(~{target_chunk_size_mb} MB each)."
            )

            for idx, chunk in enumerate(
                auction_chunks,
                start=1,
            ):

                chunk_memory_mb = (
                    chunk.memory_usage(deep=True).sum()
                    / 1024
                    / 1024
                )

                print(
                    f"    auction_df{idx}.shape = "
                    f"{chunk.shape} "
                    f"(~{chunk_memory_mb:.2f} MB)"
                )

        else:

            dataframes[event_name] = combined_df

            print(
                f"  Done. Combined shape: "
                f"{combined_df.shape}"
            )

    return (
        dataframes["auctions"],
        dataframes["clicks"],
        dataframes["impressions"],
    )

# Entry point
def main() -> None:

    # Validate credentials are present
    if not (
        os.getenv("AWS_ACCESS_KEY_ID")
        and os.getenv("AWS_SECRET_ACCESS_KEY")
    ):

        sys.exit(
            "ERROR: AWS credentials not found.\n"
            "Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY before running."
        )

    auction_dfs, df_clicks, df_impressions = load_event_data()

    # Summary
    print(f"\n{'='*60}")
    print("  df_clicks")
    print(f"{'='*60}")
    print(f"  shape : {df_clicks.shape}")

    print(f"\n{'='*60}")
    print("  df_impressions")
    print(f"{'='*60}")
    print(f"  shape : {df_impressions.shape}")

    print(f"\n{'='*60}")
    print("  auction dataframe chunks")
    print(f"{'='*60}")

    for idx, df in enumerate(auction_dfs, start=1):

        print(
            f"  auction_df{idx}.shape = {df.shape}"
        )

if __name__ == "__main__":
    main()