import os
import sqlite3
import pytest
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.preprocessing import RobustScaler
import joblib

from src.api import app, load_model, TransactionInput
from src.db import Base, Transaction, get_engine


@pytest.fixture(scope="module")
def auth_headers():
    return {"X-API-Key": "sentinel-dev-secret-key-2026"}


@pytest.fixture(scope="module")
def client(auth_headers):
    with TestClient(app, headers=auth_headers) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def unauthed_client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_payload():
    # Transaction pattern with high fraud probability
    return {
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


@pytest.fixture(scope="module")
def normal_payload():
    # Typical legitimate transaction (zero or small centered PCA values)
    payload = {f"v{i}": 0.0 for i in range(1, 29)}
    payload["time_elapsed"] = 50000.0
    payload["amount"] = 25.50
    return payload


# --- Test Group 1: Database Schema & Integrity ---
def test_db_schema_and_record_count():
    db_path = "data/creditcard.db"
    assert Path(db_path).exists(), f"Database {db_path} does not exist"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT count(*) FROM transactions")
    count = cursor.fetchone()[0]
    assert count == 284807, f"Expected 284,807 records, found {count}"

    cursor.execute("PRAGMA table_info(transactions)")
    columns = [row[1] for row in cursor.fetchall()]

    expected_cols = ["id", "time_elapsed"] + [f"v{i}" for i in range(1, 29)] + ["amount", "is_fraud"]
    for col in expected_cols:
        assert col in columns, f"Missing column {col} in transactions table"

    # Verify indices
    cursor.execute("PRAGMA index_list(transactions)")
    indices = [row[1] for row in cursor.fetchall()]
    assert any("is_fraud" in idx for idx in indices), "Index on is_fraud is missing"
    assert any("time_elapsed" in idx for idx in indices), "Index on time_elapsed is missing"

    conn.close()


# --- Test Group 2: Data Scaling & Pipeline Preprocessing ---
def test_data_scaling_behavior():
    raw_data = pd.DataFrame({
        "time_elapsed": [0.0, 50000.0, 100000.0],
        "amount": [10.0, 100.0, 1000.0]
    })
    scaler = RobustScaler()
    scaled = scaler.fit_transform(raw_data)

    assert scaled.shape == (3, 2)
    # Median should be mapped to 0 in RobustScaler
    assert scaled[1, 0] == pytest.approx(0.0)
    assert scaled[1, 1] == pytest.approx(0.0)


def test_model_artifact_integrity():
    model_path = "models/fraud_model.joblib"
    assert Path(model_path).exists(), f"Model artifact {model_path} missing"

    artifact = joblib.load(model_path)
    assert "model" in artifact
    assert "scaler" in artifact
    assert "optimal_threshold" in artifact
    assert 0.0 < artifact["optimal_threshold"] < 1.0
    assert "metrics" in artifact
    assert artifact["metrics"]["pr_auc"] > 0.80


# --- Test Group 3: API Endpoints & Inference ---
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["optimal_threshold"] is not None


def test_predict_normal_transaction(client, normal_payload):
    response = client.post("/predict", json=normal_payload)
    assert response.status_code == 200
    result = response.json()

    assert "is_fraud" in result
    assert "fraud_probability" in result
    assert "decision_threshold" in result
    assert result["is_fraud"] is False
    assert result["fraud_probability"] < result["decision_threshold"]


def test_predict_suspicious_transaction(client, sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    result = response.json()

    assert "is_fraud" in result
    assert "fraud_probability" in result
    assert isinstance(result["is_fraud"], bool)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_invalid_payload_validation(client, normal_payload):
    # Test missing field
    invalid_payload = normal_payload.copy()
    del invalid_payload["amount"]
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

    # Test negative amount
    invalid_negative = normal_payload.copy()
    invalid_negative["amount"] = -50.0
    response = client.post("/predict", json=invalid_negative)
    assert response.status_code == 422


# --- Test Group 4: Integrated Dashboard & Batch Processing ---
def test_root_dashboard_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "SentinelRisk Console" in response.text
    assert "Transaction Simulator" in response.text
    assert "Batch CSV Auditor" in response.text


def test_predict_batch_json(client, normal_payload, sample_payload):
    batch = [normal_payload, sample_payload]
    response = client.post("/predict-batch", json=batch)
    assert response.status_code == 200

    data = response.json()
    assert "summary" in data
    assert "predictions" in data

    summary = data["summary"]
    assert summary["total_evaluated"] == 2
    assert summary["total_flagged"] >= 1
    assert 0.0 <= summary["flagged_rate"] <= 1.0

    preds = data["predictions"]
    assert len(preds) == 2
    assert preds[0]["is_fraud"] is False
    assert preds[1]["is_fraud"] is True


def test_predict_batch_empty(client):
    response = client.post("/predict-batch", json=[])
    assert response.status_code == 400


def test_predict_batch_csv(client, normal_payload, sample_payload):
    headers = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # 1 Normal row
    r1 = [str(normal_payload["time_elapsed"])] + [str(normal_payload[f"v{i}"]) for i in range(1, 29)] + [str(normal_payload["amount"])]
    # 1 Fraud row
    r2 = [str(sample_payload["time_elapsed"])] + [str(sample_payload[f"v{i}"]) for i in range(1, 29)] + [str(sample_payload["amount"])]

    csv_content = ",".join(headers) + "\n" + ",".join(r1) + "\n" + ",".join(r2) + "\n"
    files = {"file": ("test_transactions.csv", csv_content, "text/csv")}

    response = client.post("/predict-batch-csv", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["summary"]["total_evaluated"] == 2
    assert data["summary"]["total_flagged"] >= 1
    assert len(data["predictions"]) == 2


# --- Test Group 5: Phase 2 Explainable AI (SHAP) & Dynamic Policy Controls ---
@pytest.fixture(scope="module")
def card_testing_payload():
    payload = {f"v{i}": 0.0 for i in range(1, 29)}
    payload.update({
        "time_elapsed": 24310.0,
        "amount": 1.0,
        "v1": -1.85, "v2": 1.42, "v3": -0.95, "v4": 2.15,
        "v7": -1.4, "v10": -1.35, "v11": 1.8, "v12": -1.75,
        "v14": -2.1, "v17": -1.5
    })
    return payload


def test_shap_explainer_attribution_and_latency(client, sample_payload):
    import time
    t0 = time.perf_counter()
    response = client.post("/predict", json=sample_payload)
    elapsed = time.perf_counter() - t0

    assert response.status_code == 200
    data = response.json()

    # Latency SLA check (< 100ms)
    assert elapsed < 0.100, f"SHAP inference latency {elapsed*1000:.2f}ms exceeded SLA target of 100ms"

    # Schema integrity for XAI
    assert "top_risk_drivers" in data
    assert "base_value" in data
    assert "active_policy" in data

    drivers = data["top_risk_drivers"]
    assert len(drivers) == 5, f"Expected 5 top risk drivers, got {len(drivers)}"

    for d in drivers:
        assert "feature" in d
        assert "shap_value" in d
        assert "direction" in d
        assert d["direction"] in ["RISK_FACTOR", "MITIGATING"]
        if d["direction"] == "RISK_FACTOR":
            assert d["shap_value"] > 0
        else:
            assert d["shap_value"] <= 0


def test_policy_management_endpoints(client):
    # Check initial policy
    res = client.get("/policy")
    assert res.status_code == 200
    pol = res.json()
    assert "active_policy" in pol
    assert "current_threshold" in pol
    assert "STRICT" in pol["presets"]
    assert "BALANCED" in pol["presets"]
    assert "AGGRESSIVE" in pol["presets"]

    # Switch to BALANCED
    res_b = client.patch("/policy", json={"policy_mode": "BALANCED"})
    assert res_b.status_code == 200
    assert res_b.json()["active_policy"] == "BALANCED"
    assert res_b.json()["current_threshold"] == 0.8500

    # Switch to CUSTOM
    res_c = client.patch("/policy", json={"policy_mode": "CUSTOM", "custom_threshold": 0.55})
    assert res_c.status_code == 200
    assert res_c.json()["active_policy"] == "CUSTOM"
    assert res_c.json()["current_threshold"] == 0.55

    # Test invalid custom threshold (caught by Pydantic schema validation or PolicyManager)
    res_inv = client.patch("/policy", json={"policy_mode": "CUSTOM", "custom_threshold": 1.5})
    assert res_inv.status_code in [400, 422]

    # Test invalid mode
    res_inv_mode = client.patch("/policy", json={"policy_mode": "UNKNOWN_MODE"})
    assert res_inv_mode.status_code in [400, 422]

    # Reset back to STRICT
    res_s = client.patch("/policy", json={"policy_mode": "STRICT"})
    assert res_s.status_code == 200
    assert res_s.json()["active_policy"] == "STRICT"
    assert res_s.json()["current_threshold"] == 0.9668


def test_policy_switch_flips_borderline_fraud(client, card_testing_payload):
    # Under STRICT (threshold = 0.9668), card-testing (~0.92 prob) is SAFE
    client.patch("/policy", json={"policy_mode": "STRICT"})
    res_strict = client.post("/predict", json=card_testing_payload)
    assert res_strict.status_code == 200
    data_strict = res_strict.json()
    assert data_strict["active_policy"] == "STRICT"
    assert data_strict["is_fraud"] is False, "Card-testing should be SAFE under STRICT policy"

    # Switch to AGGRESSIVE (threshold = 0.7000)
    client.patch("/policy", json={"policy_mode": "AGGRESSIVE"})
    res_agg = client.post("/predict", json=card_testing_payload)
    assert res_agg.status_code == 200
    data_agg = res_agg.json()
    assert data_agg["active_policy"] == "AGGRESSIVE"
    assert data_agg["is_fraud"] is True, "Card-testing should flip to FLAGGED AS FRAUD under AGGRESSIVE policy"

    # Reset to STRICT
    client.patch("/policy", json={"policy_mode": "STRICT"})


# --- Test Group 5: Dynamic SQL-Backed Sampling ---
def test_dynamic_sample_csv_endpoint(client):
    response = client.get("/api/sample-csv?total_rows=20&fraud_count=3")
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    assert "X-Request-ID" in response.headers
    assert "attachment; filename=\"dynamic_audit_batch.csv\"" in response.headers.get("content-disposition", "")

    # Parse CSV content
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    assert len(lines) == 21  # 1 header + 20 data rows

    headers = lines[0].split(",")
    expected_headers = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    assert headers == expected_headers
    assert "is_fraud" not in headers
    assert "Class" not in headers


def test_dynamic_sample_json_endpoint(client):
    response = client.get("/api/sample-json?total_rows=15&fraud_count=2")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 15

    # Schema validation against TransactionInput
    for item in data:
        t = TransactionInput(**item)
        assert hasattr(t, "time_elapsed")
        assert hasattr(t, "v1")
        assert hasattr(t, "v28")
        assert hasattr(t, "amount")


def test_dynamic_sample_validation_error(client):
    # fraud_count > total_rows should return HTTP 400
    response = client.get("/api/sample-json?total_rows=5&fraud_count=10")
    assert response.status_code == 400
    assert "cannot exceed total_rows" in response.json()["detail"]

    response_csv = client.get("/api/sample-csv?total_rows=5&fraud_count=10")
    assert response_csv.status_code == 400


# --- Test Group 6: Phase 3 Security, Auth, Rate Limiting & Audit Trail ---
def test_security_unauthenticated_request_rejected(unauthed_client, normal_payload):
    # Unauthenticated /predict should return 401
    res = unauthed_client.post("/predict", json=normal_payload)
    assert res.status_code == 401
    assert "Unauthorized" in res.json()["detail"]

    # Unauthenticated /predict-batch should return 401
    res_batch = unauthed_client.post("/predict-batch", json=[normal_payload])
    assert res_batch.status_code == 401

    # Unauthenticated PATCH /policy should return 401
    res_pol = unauthed_client.patch("/policy", json={"policy_mode": "BALANCED"})
    assert res_pol.status_code == 401


def test_security_invalid_api_key_rejected(unauthed_client, normal_payload):
    res = unauthed_client.post(
        "/predict",
        json=normal_payload,
        headers={"X-API-Key": "invalid-bogus-key"},
    )
    assert res.status_code == 401
    assert "Unauthorized" in res.json()["detail"]


def test_security_valid_api_key_accepted(unauthed_client, normal_payload):
    res = unauthed_client.post(
        "/predict",
        json=normal_payload,
        headers={"X-API-Key": "sentinel-dev-secret-key-2026"},
    )
    assert res.status_code == 200
    assert "is_fraud" in res.json()


def test_security_public_endpoints_accessible_without_auth(unauthed_client):
    # GET / is public
    res_root = unauthed_client.get("/")
    assert res_root.status_code == 200

    # GET /health is public
    res_health = unauthed_client.get("/health")
    assert res_health.status_code == 200

    # GET /policy is public
    res_policy = unauthed_client.get("/policy")
    assert res_policy.status_code == 200

    # GET /api/sample-* is public
    res_sample = unauthed_client.get("/api/sample-csv?total_rows=5&fraud_count=1")
    assert res_sample.status_code == 200


def test_validation_boundary_limits(client, normal_payload):
    # Negative amount -> 422
    neg_payload = normal_payload.copy()
    neg_payload["amount"] = -1.0
    assert client.post("/predict", json=neg_payload).status_code == 422

    # Amount > 500,000 -> 422
    huge_payload = normal_payload.copy()
    huge_payload["amount"] = 500001.0
    assert client.post("/predict", json=huge_payload).status_code == 422

    # Negative time_elapsed -> 422
    neg_time = normal_payload.copy()
    neg_time["time_elapsed"] = -5.0
    assert client.post("/predict", json=neg_time).status_code == 422

    # PCA feature out of boundary (e.g. v1 > 50.0) -> 422
    out_v = normal_payload.copy()
    out_v["v1"] = 55.0
    assert client.post("/predict", json=out_v).status_code == 422


def test_request_id_middleware_propagation(client, normal_payload):
    # Auto-generated UUID
    res = client.post("/predict", json=normal_payload)
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    assert len(res.headers["X-Request-ID"]) > 10

    # Custom caller-supplied request ID propagated
    custom_id = "audit-req-xyz-987"
    res2 = client.post("/predict", json=normal_payload, headers={"X-Request-ID": custom_id})
    assert res2.status_code == 200
    assert res2.headers["X-Request-ID"] == custom_id


def test_structured_audit_log_generation(client, sample_payload):
    import json
    log_file = Path("logs/audit.log")

    # Make a prediction
    res = client.post("/predict", json=sample_payload)
    assert res.status_code == 200

    assert log_file.exists(), "logs/audit.log should exist after prediction"
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0
    # Parse last record
    last_record = json.loads(lines[-1])
    required_keys = [
        "timestamp_utc",
        "request_id",
        "client_ip",
        "endpoint",
        "amount",
        "fraud_probability",
        "decision_threshold",
        "verdict",
        "latency_ms",
    ]
    for key in required_keys:
        assert key in last_record, f"Missing {key} in audit log record"
    assert last_record["endpoint"] == "/predict"
    assert last_record["verdict"] in ["APPROVED", "FLAGGED_AS_FRAUD"]


# --- Test Group 7: Tab 3 Model Analytics & Retraining Studio ---
def test_model_analytics_endpoint(client):
    res = client.get("/api/model-analytics")
    assert res.status_code == 200
    data = res.json()

    # Core performance metrics
    assert "pr_auc" in data
    assert "roc_auc" in data
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "optimal_threshold" in data
    assert 0.0 < data["pr_auc"] <= 1.0
    assert 0.0 < data["roc_auc"] <= 1.0

    # Confusion matrix
    assert "confusion_matrix" in data
    cm = data["confusion_matrix"]
    assert "true_negatives" in cm
    assert "false_positives" in cm
    assert "false_negatives" in cm
    assert "true_positives" in cm
    assert cm["true_negatives"] > 0
    assert cm["true_positives"] > 0

    # Sampled Precision-Recall coordinates for plotting
    assert "pr_curve_samples" in data
    assert isinstance(data["pr_curve_samples"], list)
    assert len(data["pr_curve_samples"]) > 0
    sample_pt = data["pr_curve_samples"][0]
    assert "recall" in sample_pt
    assert "precision" in sample_pt


def test_retrain_status_endpoint(client):
    res = client.get("/api/retrain/status")
    assert res.status_code == 200
    data = res.json()

    assert "status" in data
    assert data["status"] in ["idle", "training", "completed", "failed"]
    assert "progress" in data
    assert 0 <= data["progress"] <= 100
    assert "message" in data


def test_retrain_endpoint_validation(client):
    # Test boundary limits on n_trials
    res_invalid_trials = client.post("/api/retrain", json={"n_trials": 0})
    assert res_invalid_trials.status_code in [400, 422]

    # Test boundary limits on timeout
    res_invalid_timeout = client.post("/api/retrain", json={"timeout": 5})
    assert res_invalid_timeout.status_code in [400, 422]





