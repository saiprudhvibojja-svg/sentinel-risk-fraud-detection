import os
import sqlite3
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, Index
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    time_elapsed = Column(Float, nullable=False, index=True)
    v1 = Column(Float, nullable=False)
    v2 = Column(Float, nullable=False)
    v3 = Column(Float, nullable=False)
    v4 = Column(Float, nullable=False)
    v5 = Column(Float, nullable=False)
    v6 = Column(Float, nullable=False)
    v7 = Column(Float, nullable=False)
    v8 = Column(Float, nullable=False)
    v9 = Column(Float, nullable=False)
    v10 = Column(Float, nullable=False)
    v11 = Column(Float, nullable=False)
    v12 = Column(Float, nullable=False)
    v13 = Column(Float, nullable=False)
    v14 = Column(Float, nullable=False)
    v15 = Column(Float, nullable=False)
    v16 = Column(Float, nullable=False)
    v17 = Column(Float, nullable=False)
    v18 = Column(Float, nullable=False)
    v19 = Column(Float, nullable=False)
    v20 = Column(Float, nullable=False)
    v21 = Column(Float, nullable=False)
    v22 = Column(Float, nullable=False)
    v23 = Column(Float, nullable=False)
    v24 = Column(Float, nullable=False)
    v25 = Column(Float, nullable=False)
    v26 = Column(Float, nullable=False)
    v27 = Column(Float, nullable=False)
    v28 = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    is_fraud = Column(Integer, nullable=False, index=True)

    __table_args__ = (
        Index("idx_transactions_is_fraud", "is_fraud"),
        Index("idx_transactions_time_elapsed", "time_elapsed"),
    )


def get_db_path():
    return os.getenv("DB_PATH", "data/creditcard.db")


def get_engine(db_path=None):
    if db_path is None:
        db_path = get_db_path()
    # Ensure parent dir exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db(engine):
    Base.metadata.create_all(engine)


def ingest_csv_to_db(csv_path="data/creditcard.csv", db_path="data/creditcard.db", chunk_size=50000):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Source CSV not found at: {csv_path}")

    engine = get_engine(db_path)
    # Drop and recreate table if refreshing
    Base.metadata.drop_all(engine)
    init_db(engine)

    col_rename = {
        "Time": "time_elapsed",
        "Amount": "amount",
        "Class": "is_fraud",
    }
    for i in range(1, 29):
        col_rename[f"V{i}"] = f"v{i}"

    print(f"Ingesting {csv_path} into {db_path} in chunks of {chunk_size}...")
    total_ingested = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk = chunk.rename(columns=col_rename)
        chunk.to_sql("transactions", con=engine, if_exists="append", index=False)
        total_ingested += len(chunk)
        print(f"Ingested batch: cumulative rows = {total_ingested}")

    print("Verifying ingested count...")
    with engine.connect() as conn:
        result = conn.exec_driver_sql("SELECT COUNT(*) FROM transactions").scalar()
        fraud_count = conn.exec_driver_sql("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1").scalar()
    print(f"Total rows in transactions: {result}")
    print(f"Total fraud rows: {fraud_count}")
    return result


if __name__ == "__main__":
    count = ingest_csv_to_db()
    assert count == 284807, f"Expected 284,807 rows but found {count}"
    print("Database pipeline completed successfully!")
