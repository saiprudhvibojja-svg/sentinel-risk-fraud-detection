import os
import random
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/creditcard.db")
OUTPUT_DIR = Path("data/test_batches")

# SQL columns in transactions table
COLS = ["time_elapsed"] + [f"v{i}" for i in range(1, 29)] + ["amount"]
HEADER_RENAME = {"time_elapsed": "Time", "amount": "Amount"}
for i in range(1, 29):
    HEADER_RENAME[f"v{i}"] = f"V{i}"


def fetch_samples(conn: sqlite3.Connection, legit_count: int, fraud_count: int) -> pd.DataFrame:
    col_str = ", ".join(COLS)
    dfs = []
    
    if legit_count > 0:
        query_legit = f"""
            SELECT {col_str} FROM transactions 
            WHERE is_fraud = 0 
            ORDER BY RANDOM() 
            LIMIT {legit_count}
        """
        df_legit = pd.read_sql_query(query_legit, conn)
        dfs.append(df_legit)

    if fraud_count > 0:
        query_fraud = f"""
            SELECT {col_str} FROM transactions 
            WHERE is_fraud = 1 
            ORDER BY RANDOM() 
            LIMIT {fraud_count}
        """
        df_fraud = pd.read_sql_query(query_fraud, conn)
        dfs.append(df_fraud)

    if not dfs:
        return pd.DataFrame(columns=COLS)

    combined = pd.concat(dfs, ignore_index=True)
    # Shuffle combined dataset
    shuffled = combined.sample(frac=1.0, random_state=random.randint(1, 100000)).reset_index(drop=True)
    # Rename to standard CSV headers
    return shuffled.rename(columns=HEADER_RENAME)


def generate_synthetic_samples(legit_count: int, fraud_count: int) -> pd.DataFrame:
    rows = []
    for _ in range(legit_count):
        row = {"Time": round(random.uniform(100.0, 150000.0), 1), "Amount": round(random.uniform(5.0, 250.0), 2)}
        for i in range(1, 29):
            row[f"V{i}"] = round(random.gauss(0.0, 0.4), 4)
        rows.append(row)

    for _ in range(fraud_count):
        row = {"Time": round(random.uniform(100.0, 150000.0), 1), "Amount": round(random.uniform(0.0, 800.0), 2)}
        for i in range(1, 29):
            row[f"V{i}"] = round(random.gauss(0.0, 1.2), 4)
        row["V4"] = round(random.uniform(2.5, 4.5), 4)
        row["V14"] = round(random.uniform(-4.5, -2.5), 4)
        row["V17"] = round(random.uniform(-3.5, -1.5), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0).reset_index(drop=True)


def generate_all_batches():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    use_db = DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH) if use_db else None

    print("=" * 60)
    source_msg = "SQLite (data/creditcard.db)" if use_db else "Synthetic Zero-Config Generator"
    print(f"Generating Offline Test Batches from {source_msg}")
    print("=" * 60)

    generated_files = []
    sampler = (lambda l, f: fetch_samples(conn, legit_count=l, fraud_count=f)) if use_db else generate_synthetic_samples

    # 1. Normal Day Batches: 50 records each (0 to 1 fraud cases)
    print("\n--- Generating 5x Normal Day Batches (50 rows) ---")
    for i in range(1, 6):
        fraud_cnt = random.choice([0, 1])
        legit_cnt = 50 - fraud_cnt
        df = sampler(legit_cnt, fraud_cnt)
        file_path = OUTPUT_DIR / f"normal_day_batch_{i}.csv"
        df.to_csv(file_path, index=False)
        generated_files.append((file_path.name, len(df), fraud_cnt))
        print(f"Created {file_path.name}: {len(df)} records ({fraud_cnt} fraud, {legit_cnt} legit)")

    # 2. Attack Wave Batches: 60 records each (5 to 12 fraud cases)
    print("\n--- Generating 5x Attack Wave Batches (60 rows) ---")
    for i in range(1, 6):
        fraud_cnt = random.randint(5, 12)
        legit_cnt = 60 - fraud_cnt
        df = sampler(legit_cnt, fraud_cnt)
        file_path = OUTPUT_DIR / f"attack_wave_batch_{i}.csv"
        df.to_csv(file_path, index=False)
        generated_files.append((file_path.name, len(df), fraud_cnt))
        print(f"Created {file_path.name}: {len(df)} records ({fraud_cnt} fraud, {legit_cnt} legit)")

    # 3. Enterprise Audit Batches: 200 records each (5 to 20 fraud cases)
    print("\n--- Generating 5x Enterprise Audit Batches (200 rows) ---")
    for i in range(1, 6):
        fraud_cnt = random.randint(5, 20)
        legit_cnt = 200 - fraud_cnt
        df = sampler(legit_cnt, fraud_cnt)
        file_path = OUTPUT_DIR / f"enterprise_audit_batch_{i}.csv"
        df.to_csv(file_path, index=False)
        generated_files.append((file_path.name, len(df), fraud_cnt))
        print(f"Created {file_path.name}: {len(df)} records ({fraud_cnt} fraud, {legit_cnt} legit)")

    if conn:
        conn.close()

    print("\n" + "=" * 60)
    print(f"Successfully generated {len(generated_files)} batch files in {OUTPUT_DIR}")
    print("=" * 60)
    return generated_files


if __name__ == "__main__":
    generate_all_batches()
