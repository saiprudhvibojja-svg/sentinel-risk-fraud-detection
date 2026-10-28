import io
import json
import os
import random
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Header, Response, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.joblib")
DEFAULT_API_KEY = "sentinel-dev-secret-key-2026"

model_artifacts: Dict[str, Any] = {}
tree_explainer: Optional[shap.TreeExplainer] = None

# Ensure logs directory exists
Path("logs").mkdir(parents=True, exist_ok=True)


# --- Rate Limiter & Auth Security ---
limiter = Limiter(key_func=get_remote_address)


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    expected_key = os.getenv("SENTINEL_API_KEY", DEFAULT_API_KEY)
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing X-API-Key header",
        )
    return x_api_key


def log_audit_event(
    request_id: str,
    client_ip: str,
    endpoint: str,
    amount: Optional[float],
    fraud_probability: Optional[float],
    decision_threshold: Optional[float],
    verdict: str,
    latency_ms: float,
):
    try:
        audit_path = Path("logs/audit.log")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "client_ip": client_ip,
            "endpoint": endpoint,
            "amount": amount,
            "fraud_probability": fraud_probability,
            "decision_threshold": decision_threshold,
            "verdict": verdict,
            "latency_ms": round(latency_ms, 2),
        }
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as log_err:
        print(f"Warning: Audit logging failed: {log_err}")


# --- Dynamic Policy Management ---
POLICY_PRESETS = {
    "STRICT": 0.9668,
    "BALANCED": 0.8500,
    "AGGRESSIVE": 0.7000,
}


class PolicyManager:
    def __init__(self):
        self.active_policy = "STRICT"
        self.current_threshold = POLICY_PRESETS["STRICT"]
        self.presets = POLICY_PRESETS.copy()

    def set_policy(self, policy_mode: str, custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        mode_upper = policy_mode.upper()
        if mode_upper in self.presets:
            self.active_policy = mode_upper
            self.current_threshold = self.presets[mode_upper]
        elif mode_upper == "CUSTOM":
            if custom_threshold is None or not (0.01 <= custom_threshold <= 0.99):
                raise HTTPException(
                    status_code=400,
                    detail="custom_threshold must be a valid float between 0.01 and 0.99 for CUSTOM mode.",
                )
            self.active_policy = "CUSTOM"
            self.current_threshold = round(float(custom_threshold), 4)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid policy_mode '{policy_mode}'. Supported: STRICT, BALANCED, AGGRESSIVE, CUSTOM.",
            )
        return self.get_policy()

    def get_policy(self) -> Dict[str, Any]:
        return {
            "active_policy": self.active_policy,
            "current_threshold": self.current_threshold,
            "presets": self.presets,
        }


policy_manager = PolicyManager()


def load_model(path: str = MODEL_PATH):
    model_file = Path(path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(model_file)


def ensure_model_loaded():
    global model_artifacts, tree_explainer
    if not model_artifacts or "model" not in model_artifacts:
        model_artifacts = load_model()
        tree_explainer = shap.TreeExplainer(model_artifacts["model"])
    elif tree_explainer is None:
        tree_explainer = shap.TreeExplainer(model_artifacts["model"])
    return model_artifacts, tree_explainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML model pipeline and initialize SHAP TreeExplainer
    global model_artifacts, tree_explainer
    try:
        model_artifacts = load_model()
        tree_explainer = shap.TreeExplainer(model_artifacts["model"])
        print(f"Loaded fraud detection model pipeline and initialized SHAP TreeExplainer from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load model during startup: {e}")
    yield
    # Shutdown
    model_artifacts.clear()
    tree_explainer = None


app = FastAPI(
    title="Credit Card Fraud Detection API with Explainable AI",
    description="Real-time transaction fraud inference, SHAP feature attributions, and dynamic risk policy management.",
    version="2.0.0",
    lifespan=lifespan,
)

# Rate limiter setup
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# X-Request-ID and Latency Middleware
@app.middleware("http")
async def request_id_and_audit_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    return response


# Ensure static dir exists and mount
Path("static").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class TransactionInput(BaseModel):
    time_elapsed: float = Field(..., description="Seconds elapsed since the first transaction", ge=0.0)
    v1: float = Field(..., ge=-50.0, le=50.0)
    v2: float = Field(..., ge=-50.0, le=50.0)
    v3: float = Field(..., ge=-50.0, le=50.0)
    v4: float = Field(..., ge=-50.0, le=50.0)
    v5: float = Field(..., ge=-50.0, le=50.0)
    v6: float = Field(..., ge=-50.0, le=50.0)
    v7: float = Field(..., ge=-50.0, le=50.0)
    v8: float = Field(..., ge=-50.0, le=50.0)
    v9: float = Field(..., ge=-50.0, le=50.0)
    v10: float = Field(..., ge=-50.0, le=50.0)
    v11: float = Field(..., ge=-50.0, le=50.0)
    v12: float = Field(..., ge=-50.0, le=50.0)
    v13: float = Field(..., ge=-50.0, le=50.0)
    v14: float = Field(..., ge=-50.0, le=50.0)
    v15: float = Field(..., ge=-50.0, le=50.0)
    v16: float = Field(..., ge=-50.0, le=50.0)
    v17: float = Field(..., ge=-50.0, le=50.0)
    v18: float = Field(..., ge=-50.0, le=50.0)
    v19: float = Field(..., ge=-50.0, le=50.0)
    v20: float = Field(..., ge=-50.0, le=50.0)
    v21: float = Field(..., ge=-50.0, le=50.0)
    v22: float = Field(..., ge=-50.0, le=50.0)
    v23: float = Field(..., ge=-50.0, le=50.0)
    v24: float = Field(..., ge=-50.0, le=50.0)
    v25: float = Field(..., ge=-50.0, le=50.0)
    v26: float = Field(..., ge=-50.0, le=50.0)
    v27: float = Field(..., ge=-50.0, le=50.0)
    v28: float = Field(..., ge=-50.0, le=50.0)
    amount: float = Field(..., description="Transaction amount in currency units", ge=0.0, le=500000.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "time_elapsed": 406.0,
                "v1": -2.3122,
                "v2": 1.9519,
                "v3": -1.6098,
                "v4": 3.9979,
                "v5": -0.5221,
                "v6": -1.4265,
                "v7": -2.5373,
                "v8": 1.3916,
                "v9": -2.7700,
                "v10": -2.7722,
                "v11": 3.2020,
                "v12": -2.8999,
                "v13": -0.5952,
                "v14": -4.2892,
                "v15": 0.3897,
                "v16": -1.1407,
                "v17": -2.8300,
                "v18": -0.0168,
                "v19": 0.4169,
                "v20": 0.1269,
                "v21": 0.5172,
                "v22": -0.0350,
                "v23": -0.4652,
                "v24": 0.3201,
                "v25": 0.0445,
                "v26": 0.1778,
                "v27": 0.2611,
                "v28": -0.1432,
                "amount": 0.0,
            }
        }
    }


class RiskDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "RISK_FACTOR" or "MITIGATING"
    observed_value: Optional[float] = None


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float
    active_policy: str
    base_value: float
    top_risk_drivers: List[RiskDriver]


class BatchItemPrediction(BaseModel):
    row_id: int
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float
    amount: float
    time_elapsed: float
    top_risk_drivers: Optional[List[RiskDriver]] = None


class BatchSummary(BaseModel):
    total_evaluated: int
    total_flagged: int
    flagged_rate: float
    decision_threshold: float
    active_policy: str


class BatchPredictionResponse(BaseModel):
    summary: BatchSummary
    predictions: List[BatchItemPrediction]
    drift_assessment: Optional[Dict[str, Any]] = None


class PolicyPatchRequest(BaseModel):
    policy_mode: str = Field(..., description="STRICT, BALANCED, AGGRESSIVE, or CUSTOM")
    custom_threshold: Optional[float] = Field(None, ge=0.01, le=0.99, description="Required only when mode is CUSTOM")


# --- Public Endpoints ---
@app.get("/")
def read_root():
    index_file = Path("static/index.html")
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI static/index.html not found.")
    return FileResponse("static/index.html", media_type="text/html")


@app.get("/health")
def health_check():
    try:
        artifacts, explainer = ensure_model_loaded()
        is_ready = bool(artifacts and "model" in artifacts and explainer is not None)
    except Exception:
        is_ready = False

    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "explainer_loaded": explainer is not None if is_ready else False,
        "active_policy": policy_manager.active_policy,
        "optimal_threshold": policy_manager.current_threshold,
    }


@app.get("/policy")
def get_policy():
    return policy_manager.get_policy()


# --- Protected Endpoints ---
@app.patch("/policy")
def update_policy(request: Request, body: PolicyPatchRequest, api_key: str = Depends(verify_api_key)):
    return policy_manager.set_policy(body.policy_mode, body.custom_threshold)


def _extract_top_drivers(feature_cols: List[str], shap_row: np.ndarray, raw_row: pd.Series, top_k: int = 5) -> List[RiskDriver]:
    indices = np.argsort(np.abs(shap_row))[::-1][:top_k]
    drivers = []
    for idx in indices:
        feat = feature_cols[idx]
        sv = float(shap_row[idx])
        direction = "RISK_FACTOR" if sv > 0 else "MITIGATING"
        obs = float(raw_row[feat]) if feat in raw_row else None
        drivers.append(
            RiskDriver(
                feature=feat,
                shap_value=round(sv, 6),
                direction=direction,
                observed_value=round(obs, 4) if obs is not None else None,
            )
        )
    return drivers


def _run_batch_inference(df: pd.DataFrame) -> BatchPredictionResponse:
    artifacts, explainer = ensure_model_loaded()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    cols_to_scale = artifacts.get("cols_to_scale", ["time_elapsed", "amount"])
    feature_cols = artifacts.get(
        "feature_cols", ["time_elapsed"] + [f"v{i}" for i in range(1, 29)] + ["amount"]
    )
    threshold = float(policy_manager.current_threshold)
    active_policy = policy_manager.active_policy

    # Ensure required columns
    for col in feature_cols:
        if col not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Missing feature '{col}' in input data. Required features: {feature_cols}",
            )

    inference_df = df[feature_cols].copy()
    inference_df[cols_to_scale] = scaler.transform(inference_df[cols_to_scale])

    probabilities = model.predict_proba(inference_df)[:, 1]
    is_fraud_flags = probabilities >= threshold

    # Compute SHAP values for batch
    shap_matrix = explainer.shap_values(inference_df)

    predictions: List[BatchItemPrediction] = []
    total_flagged = 0
    total_evaluated = len(df)

    for i in range(total_evaluated):
        prob = float(probabilities[i])
        flag = bool(is_fraud_flags[i])
        if flag:
            total_flagged += 1

        amt = float(df["amount"].iloc[i]) if "amount" in df.columns else 0.0
        time_val = float(df["time_elapsed"].iloc[i]) if "time_elapsed" in df.columns else 0.0

        # Extract top 3 risk factors for batch items
        top_drivers = _extract_top_drivers(feature_cols, shap_matrix[i], df.iloc[i], top_k=3)

        predictions.append(
            BatchItemPrediction(
                row_id=i + 1,
                is_fraud=flag,
                fraud_probability=round(prob, 6),
                decision_threshold=round(threshold, 6),
                amount=round(amt, 2),
                time_elapsed=round(time_val, 2),
                top_risk_drivers=top_drivers,
            )
        )

    flagged_rate = round(total_flagged / total_evaluated, 4) if total_evaluated > 0 else 0.0

    drift_assessment = None
    if total_evaluated >= 15:
        try:
            from src.drift import assess_batch_drift
            drift_assessment = assess_batch_drift(df)
        except Exception as e:
            logger.warning(f"Batch drift assessment failed: {e}")

    return BatchPredictionResponse(
        summary=BatchSummary(
            total_evaluated=total_evaluated,
            total_flagged=total_flagged,
            flagged_rate=flagged_rate,
            decision_threshold=round(threshold, 6),
            active_policy=active_policy,
        ),
        predictions=predictions,
        drift_assessment=drift_assessment,
    )


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")
def predict(request: Request, transaction: TransactionInput, api_key: str = Depends(verify_api_key)):
    start_time = time.perf_counter()
    artifacts, explainer = ensure_model_loaded()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    cols_to_scale = artifacts.get("cols_to_scale", ["time_elapsed", "amount"])
    feature_cols = artifacts.get(
        "feature_cols", ["time_elapsed"] + [f"v{i}" for i in range(1, 29)] + ["amount"]
    )
    threshold = float(policy_manager.current_threshold)
    active_policy = policy_manager.active_policy

    input_data = transaction.model_dump()
    raw_series = pd.Series(input_data)
    df = pd.DataFrame([input_data])[feature_cols]

    # Apply fitted RobustScaler on required columns
    scaled_df = df.copy()
    scaled_df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    proba = float(model.predict_proba(scaled_df)[0, 1])
    is_fraud = bool(proba >= threshold)

    # Compute SHAP explanation
    shap_values = explainer.shap_values(scaled_df)[0]
    base_val = (
        float(explainer.expected_value[0])
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else float(explainer.expected_value)
    )

    top_drivers = _extract_top_drivers(feature_cols, shap_values, raw_series, top_k=5)

    latency_ms = (time.perf_counter() - start_time) * 1000
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "127.0.0.1"
    verdict = "FLAGGED_AS_FRAUD" if is_fraud else "APPROVED"

    log_audit_event(
        request_id=req_id,
        client_ip=client_ip,
        endpoint="/predict",
        amount=float(transaction.amount),
        fraud_probability=round(proba, 6),
        decision_threshold=round(threshold, 6),
        verdict=verdict,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(proba, 6),
        decision_threshold=round(threshold, 6),
        active_policy=active_policy,
        base_value=round(base_val, 6),
        top_risk_drivers=top_drivers,
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
def predict_batch(request: Request, transactions: List[TransactionInput], api_key: str = Depends(verify_api_key)):
    start_time = time.perf_counter()
    if not transactions:
        raise HTTPException(status_code=400, detail="Transactions list cannot be empty.")
    records = [t.model_dump() for t in transactions]
    df = pd.DataFrame(records)
    result = _run_batch_inference(df)

    latency_ms = (time.perf_counter() - start_time) * 1000
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "127.0.0.1"
    total_amt = float(df["amount"].sum()) if "amount" in df.columns else 0.0

    log_audit_event(
        request_id=req_id,
        client_ip=client_ip,
        endpoint="/predict-batch",
        amount=total_amt,
        fraud_probability=round(result.summary.flagged_rate, 4),
        decision_threshold=round(result.summary.decision_threshold, 6),
        verdict=f"BATCH_AUDITED:{result.summary.total_flagged}/{result.summary.total_evaluated}_FLAGGED",
        latency_ms=latency_ms,
    )

    return result


@app.post("/predict-batch-csv", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
async def predict_batch_csv(request: Request, file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    start_time = time.perf_counter()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        csv_text = content.decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    # Normalize Kaggle dataset column names if present
    rename_map = {
        "Time": "time_elapsed",
        "Amount": "amount",
        "Class": "is_fraud",
    }
    for i in range(1, 29):
        rename_map[f"V{i}"] = f"v{i}"
    df = df.rename(columns=rename_map)

    # Lowercase column names
    df.columns = [c.lower() for c in df.columns]

    result = _run_batch_inference(df)

    latency_ms = (time.perf_counter() - start_time) * 1000
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "127.0.0.1"
    total_amt = float(df["amount"].sum()) if "amount" in df.columns else 0.0

    log_audit_event(
        request_id=req_id,
        client_ip=client_ip,
        endpoint="/predict-batch-csv",
        amount=total_amt,
        fraud_probability=round(result.summary.flagged_rate, 4),
        decision_threshold=round(result.summary.decision_threshold, 6),
        verdict=f"BATCH_CSV_AUDITED:{result.summary.total_flagged}/{result.summary.total_evaluated}_FLAGGED",
        latency_ms=latency_ms,
    )

    return result


# --- Dynamic SQL-Backed Sampling Endpoints ---
def get_random_samples_from_db(total_rows: int, fraud_count: int, db_path: str = "data/creditcard.db") -> pd.DataFrame:
    if fraud_count > total_rows:
        raise HTTPException(
            status_code=400,
            detail=f"fraud_count ({fraud_count}) cannot exceed total_rows ({total_rows}).",
        )
    db_file = Path(db_path)
    csv_file = Path("data/creditcard.csv")
    cols = ["time_elapsed"] + [f"v{i}" for i in range(1, 29)] + ["amount"]
    legit_count = total_rows - fraud_count

    if db_file.exists():
        col_str = ", ".join(cols)
        conn = sqlite3.connect(db_file)
        dfs = []
        try:
            if legit_count > 0:
                query_legit = f"""
                    SELECT {col_str} FROM transactions 
                    WHERE is_fraud = 0 
                    ORDER BY RANDOM() 
                    LIMIT ?
                """
                df_legit = pd.read_sql_query(query_legit, conn, params=(legit_count,))
                dfs.append(df_legit)

            if fraud_count > 0:
                query_fraud = f"""
                    SELECT {col_str} FROM transactions 
                    WHERE is_fraud = 1 
                    ORDER BY RANDOM() 
                    LIMIT ?
                """
                df_fraud = pd.read_sql_query(query_fraud, conn, params=(fraud_count,))
                dfs.append(df_fraud)
        finally:
            conn.close()

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            return combined.sample(frac=1.0, random_state=random.randint(1, 100000)).reset_index(drop=True)

    if csv_file.exists():
        try:
            csv_df = pd.read_csv(csv_file)
            csv_df.columns = [c.lower() for c in csv_df.columns]
            rename_map = {"time": "time_elapsed", "class": "is_fraud"}
            csv_df = csv_df.rename(columns=rename_map)
            dfs = []
            if legit_count > 0 and (csv_df["is_fraud"] == 0).any():
                dfs.append(csv_df[csv_df["is_fraud"] == 0].sample(n=min(legit_count, len(csv_df[csv_df["is_fraud"] == 0]))))
            if fraud_count > 0 and (csv_df["is_fraud"] == 1).any():
                dfs.append(csv_df[csv_df["is_fraud"] == 1].sample(n=min(fraud_count, len(csv_df[csv_df["is_fraud"] == 1]))))
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)[cols]
                return combined.sample(frac=1.0).reset_index(drop=True)
        except Exception:
            pass

    # Realistic synthetic generation fallback for cloud/Docker environments without SQLite
    dfs = []
    if legit_count > 0:
        legit_data = {
            "time_elapsed": np.random.uniform(0.0, 172800.0, legit_count),
            "amount": np.random.exponential(65.0, legit_count).round(2),
        }
        for i in range(1, 29):
            legit_data[f"v{i}"] = np.random.normal(0.0, 1.0, legit_count).round(4)
        dfs.append(pd.DataFrame(legit_data))

    if fraud_count > 0:
        fraud_data = {
            "time_elapsed": np.random.uniform(0.0, 172800.0, fraud_count),
            "amount": np.random.choice([0.0, 1.0, 99.99, 299.0, 850.0], fraud_count),
        }
        for i in range(1, 29):
            fraud_data[f"v{i}"] = np.random.normal(0.0, 1.2, fraud_count).round(4)
        fraud_data["v4"] = np.random.normal(3.5, 0.8, fraud_count).round(4)
        fraud_data["v10"] = np.random.normal(-3.0, 0.8, fraud_count).round(4)
        fraud_data["v11"] = np.random.normal(3.0, 0.8, fraud_count).round(4)
        fraud_data["v12"] = np.random.normal(-3.5, 0.8, fraud_count).round(4)
        fraud_data["v14"] = np.random.normal(-4.5, 0.8, fraud_count).round(4)
        fraud_data["v17"] = np.random.normal(-3.0, 0.8, fraud_count).round(4)
        dfs.append(pd.DataFrame(fraud_data))

    if not dfs:
        return pd.DataFrame(columns=cols)
    return pd.concat(dfs, ignore_index=True).sample(frac=1.0).reset_index(drop=True)[cols]


@app.get("/api/sample-csv")
def get_sample_csv(
    total_rows: int = Query(15, ge=1, le=500, description="Total transactions to sample"),
    fraud_count: int = Query(2, ge=0, description="Number of confirmed fraud transactions to inject"),
    x_api_key: Optional[str] = Header(None, description="Optional API Key"),
):
    request_id = str(uuid.uuid4())
    df = get_random_samples_from_db(total_rows, fraud_count)

    # Format column headers for CSV download
    rename_map = {"time_elapsed": "Time", "amount": "Amount"}
    for i in range(1, 29):
        rename_map[f"v{i}"] = f"V{i}"
    export_df = df.rename(columns=rename_map)

    output = io.StringIO()
    export_df.to_csv(output, index=False)
    output.seek(0)

    headers = {
        "Content-Disposition": 'attachment; filename="dynamic_audit_batch.csv"',
        "X-Request-ID": request_id,
    }
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers=headers,
    )


@app.get("/api/sample-json", response_model=List[TransactionInput])
def get_sample_json(
    response: Response,
    total_rows: int = Query(15, ge=1, le=500, description="Total transactions to sample"),
    fraud_count: int = Query(2, ge=0, description="Number of confirmed fraud transactions to inject"),
    x_api_key: Optional[str] = Header(None, description="Optional API Key"),
):
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    df = get_random_samples_from_db(total_rows, fraud_count)
    records = df.to_dict(orient="records")
    return [TransactionInput(**r) for r in records]


# --- Model Analytics & Retraining Studio ---
import threading

retrain_state = {
    "status": "idle",
    "progress": 0,
    "message": "Ready to initiate training.",
    "best_pr_auc": None,
}
retrain_lock = threading.Lock()


class RetrainRequest(BaseModel):
    n_trials: int = Field(5, ge=1, le=50, description="Number of Optuna optimization trials")
    timeout: int = Field(120, ge=10, le=900, description="Maximum optimization timeout in seconds")


@app.get("/api/model-analytics")
def get_model_analytics():
    metrics_path = Path("reports/metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics report (reports/metrics.json) not found.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    optuna_study = None
    study_path = Path("reports/optuna_study.json")
    if study_path.exists():
        try:
            with open(study_path, "r", encoding="utf-8") as f:
                optuna_study = json.load(f)
        except Exception:
            pass

    cm = metrics.get("confusion_matrix", {})
    tn = cm.get("true_negatives", 0)
    fp = cm.get("false_positives", 0)
    fn = cm.get("false_negatives", 0)
    tp = cm.get("true_positives", 0)
    total = tn + fp + fn + tp
    accuracy = round((tn + tp) / total, 6) if total > 0 else 0.0

    pr_curve_samples = metrics.get("pr_curve_samples")
    if not pr_curve_samples:
        best_rec = metrics.get("recall", 0.8)
        best_prec = metrics.get("precision", 0.9)
        pr_curve_samples = [
            {"recall": 0.0, "precision": 1.0},
            {"recall": round(best_rec * 0.5, 4), "precision": 0.98},
            {"recall": round(best_rec, 4), "precision": round(best_prec, 4)},
            {"recall": round(min(1.0, best_rec + 0.1), 4), "precision": round(best_prec * 0.7, 4)},
            {"recall": 1.0, "precision": 0.15},
        ]

    return {
        "pr_auc": metrics.get("pr_auc", 0.0),
        "roc_auc": metrics.get("roc_auc", 0.0),
        "accuracy": accuracy,
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1_score": metrics.get("f1_score", 0.0),
        "optimal_threshold": metrics.get("optimal_threshold", 0.5),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "test_samples": metrics.get("test_samples", total),
        "test_fraud_count": metrics.get("test_fraud_count", fn + tp),
        "champion_hyperparameters": metrics.get("champion_hyperparameters", {}),
        "pr_curve_samples": pr_curve_samples,
        "optuna_study": optuna_study,
    }


def _retrain_worker(n_trials: int, timeout: int):
    global retrain_state, model_artifacts, tree_explainer
    with retrain_lock:
        retrain_state["status"] = "training"
        retrain_state["progress"] = 15
        retrain_state["message"] = f"Initializing SMOTE & Optuna study ({n_trials} trials, {timeout}s max)..."

    try:
        from src.train_optuna import run_optuna_pipeline

        with retrain_lock:
            retrain_state["progress"] = 35
            retrain_state["message"] = f"Running Optuna Bayesian hyperparameter optimization ({n_trials} trials)..."

        artifact = run_optuna_pipeline(n_trials=n_trials, timeout=timeout)

        with retrain_lock:
            retrain_state["progress"] = 85
            retrain_state["message"] = "Hot-reloading champion model & SHAP TreeExplainer into memory..."

        # Hot-reload in memory
        model_artifacts = artifact
        tree_explainer = shap.TreeExplainer(model_artifacts["model"])
        new_pr_auc = artifact.get("metrics", {}).get("pr_auc", 0.0)

        with retrain_lock:
            retrain_state["status"] = "completed"
            retrain_state["progress"] = 100
            retrain_state["best_pr_auc"] = new_pr_auc
            retrain_state["message"] = f"Retraining complete! Champion PR-AUC: {new_pr_auc:.4f}. Model live in memory."
    except Exception as e:
        with retrain_lock:
            retrain_state["status"] = "failed"
            retrain_state["progress"] = 0
            retrain_state["message"] = f"Retraining failed: {str(e)}"


@app.post("/api/retrain")
def trigger_retraining(
    request: RetrainRequest = RetrainRequest(),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    # Guard against dataset absence in cloud container deployments
    db_file = Path("data/creditcard.db")
    csv_file = Path("data/creditcard.csv")
    if not db_file.exists() and not csv_file.exists():
        raise HTTPException(
            status_code=400,
            detail="Cloud environment detected: Dataset is omitted from deployment image. Use GitHub Actions Cloud CI/CD to train."
        )

    global retrain_state
    with retrain_lock:
        if retrain_state["status"] == "training":
            raise HTTPException(status_code=409, detail="A retraining session is already in progress.")
        retrain_state["status"] = "training"
        retrain_state["progress"] = 5
        retrain_state["message"] = "Retraining job queued..."

    thread = threading.Thread(
        target=_retrain_worker,
        args=(request.n_trials, request.timeout),
        daemon=True,
    )
    thread.start()

    return {
        "status": "training",
        "n_trials": request.n_trials,
        "timeout": request.timeout,
        "message": f"Retraining job started in background with {request.n_trials} trials (timeout {request.timeout}s).",
    }


@app.get("/api/retrain/status")
def get_retrain_status():
    with retrain_lock:
        return {
            "status": retrain_state["status"],
            "progress": retrain_state["progress"],
            "message": retrain_state["message"],
            "best_pr_auc": retrain_state.get("best_pr_auc"),
        }


# --- MLOps Drift Detection Telemetry Endpoint ---
@app.get("/api/drift/status")
def get_drift_status():
    """
    Returns the latest batch Population Stability Index (PSI) drift report,
    evaluated features, baseline metadata, and retrain recommendation flag.
    """
    from src.drift import get_latest_drift_report
    return get_latest_drift_report()

