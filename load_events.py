"""
load_events.py
--------------
Reads auction, click, and impression Parquet files from S3 for a given
year/month and loads them into three separate DataFrames.

Usage:
    python load_events.py                      # defaults to year=2026, month=05
    python load_events.py --year 2026 --month 04
"""
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present
import argparse
import io
import os
import sys
import boto3
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUCKET = "tsi-ads360-sftp"

EVENT_PREFIXES = {
    "auctions":    "staging/bidder-data/tsq-events/parquet/ad360_tabular/auctions/",
    "clicks":      "staging/bidder-data/tsq-events/parquet/ad360_tabular/clicks/",
    "impressions": "staging/bidder-data/tsq-events/parquet/ad360_tabular/impressions/",
}

# Hard stop if total S3 size for a single event type exceeds this.
# Note: in-memory DataFrames are typically 3-5x larger than the S3
# compressed size, so budget your RAM accordingly.
MAX_S3_BYTES_PER_EVENT = 5 * 1024 ** 3  # 5 GB

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def build_s3_client() -> boto3.client:
    """
    Build an S3 client.  Credentials are resolved from the standard boto3
    chain: env vars → ~/.aws/credentials → IAM role.  No secrets in code.
    """
    return boto3.client("s3")

def list_parquet_keys(
    s3_client,
    bucket: str,
    prefix: str,
    year: str,
    month: str,
) -> tuple[list[str], int]:
   """
   Return (keys, total_bytes) for every .parquet object under
   prefix/year=YYYY/month=MM/, handling S3 pagination transparently.
 
   Prints each file path and its compressed S3 size as it scans so you
   can see what exists before anything is loaded into memory.
   """
   month_prefix = f"{prefix}year={year}/month={month}/"
   paginator = s3_client.get_paginator("list_objects_v2")
   pages = paginator.paginate(Bucket=bucket, Prefix=month_prefix)
 
   keys = []
   total_bytes = 0
 
   for page in pages:
       for obj in page.get("Contents", []):
           key = obj["Key"]
           size = obj["Size"]
           if key.endswith(".parquet"):
               keys.append(key)
               total_bytes += size
               print(f"    {key}  ({size / 1024 / 1024:.2f} MB)")
 
   print(f"\n  Subtotal: {len(keys)} file(s), {total_bytes / 1024 / 1024:.2f} MB on S3")
   print(f"  Estimated in-memory size: ~{total_bytes * 4 / 1024 / 1024:.0f}-{total_bytes * 5 / 1024 / 1024:.0f} MB "
         f"(3-5x decompression factor)")
 
   return keys, total_bytes

def read_parquet_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """
    Download a single Parquet file from S3 into memory and return a DataFrame.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    body_bytes = response["Body"].read()
    return pd.read_parquet(io.BytesIO(body_bytes))

# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_event_data(year: str, month: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   """
   Pull all Parquet files for the given year/month for every event type
   and return (df_auctions, df_clicks, df_impressions).
   """
   s3 = build_s3_client()
   dataframes: dict[str, pd.DataFrame] = {}
 
   for event_name, prefix in EVENT_PREFIXES.items():
       print(f"\n[{event_name}] Listing keys under year={year}/month={month} ...")
       keys, total_bytes = list_parquet_keys(s3, BUCKET, prefix, year, month)
 
       if not keys:
           print(f"  WARNING: no .parquet files found - returning empty DataFrame.")
           dataframes[event_name] = pd.DataFrame()
           continue
 
       if total_bytes > MAX_S3_BYTES_PER_EVENT:
           raise ValueError(
               f"[{event_name}] Total S3 size {total_bytes / 1024 ** 3:.2f} GB "
               f"exceeds the {MAX_S3_BYTES_PER_EVENT / 1024 ** 3:.0f} GB limit. "
               f"Narrow the date range or raise MAX_S3_BYTES_PER_EVENT."
           )
 
       print(f"\n  Reading {len(keys)} file(s) into memory ...")
       parts: list[pd.DataFrame] = []
       for i, key in enumerate(keys, start=1):
           print(f"  [{i}/{len(keys)}] {key.split('/')[-1]}")
           df_part = read_parquet_from_s3(s3, BUCKET, key)
           parts.append(df_part)
 
       dataframes[event_name] = pd.concat(parts, ignore_index=True)
       print(f"  Done.  Combined shape: {dataframes[event_name].shape}")
 
   return dataframes["auctions"], dataframes["clicks"], dataframes["impressions"]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Load PPQ event Parquet data from S3.")
    parser.add_argument("--year",  default="2026", help="4-digit year  (default: 2026)")
    parser.add_argument("--month", default="05",   help="Zero-padded month (default: 05)")
    args = parser.parse_args()

    # Validate credentials are present (fail fast with a clear message)
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        sys.exit(
            "ERROR: AWS credentials not found.\n"
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables "
            "before running this script."
        )

    df_auctions, df_clicks, df_impressions = load_event_data(args.year, args.month)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

    for name, df in [
        ("df_auctions",    df_auctions),
        ("df_clicks",      df_clicks),
        ("df_impressions", df_impressions),
    ]:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  shape : {df.shape}")
        print(f"  dtypes:\n{df.dtypes.to_string()}")

if __name__ == "__main__":
    main()