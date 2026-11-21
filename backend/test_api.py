"""
Automated Backend API Test Suite for Hinglish Robustness Suite
Tests:
- Application startup & model loading
- GET /metrics (Hold-out Accuracy & Macro F1 for baseline & upgraded models)
- POST /analyze with diverse Hinglish test sentences across both models
- Asserts HTTP 200 status and validates payload schemas
"""

import os
import sys

# Ensure backend directory is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from fastapi.testclient import TestClient
from main import app


def test_api_suite():
    print("=" * 70)
    print("STARTING FASTAPI TEST SUITE (FastAPI TestClient)")
    print("=" * 70)

    # TestClient context manager triggers startup and shutdown lifespan events
    with TestClient(app) as client:
        # 1. Test GET /metrics
        print("\n[1] Testing GET /metrics ...")
        metrics_resp = client.get("/metrics")
        assert metrics_resp.status_code == 200, f"Expected 200, got {metrics_resp.status_code}"
        metrics = metrics_resp.json()
        print("[+] GET /metrics returned HTTP 200 OK")
        print(f"    - Baseline Model:  Accuracy = {metrics.get('baseline', {}).get('accuracy')}% | Macro F1 = {metrics.get('baseline', {}).get('macro_f1')}%")
        print(f"    - Classical Upgrade: Accuracy = {metrics.get('upgraded', {}).get('accuracy')}% | Macro F1 = {metrics.get('upgraded', {}).get('macro_f1')}%")
        print(f"    - Hold-out Test Partition: {metrics.get('upgraded', {}).get('test_samples')} samples")

        assert "baseline" in metrics, "Missing 'baseline' in /metrics response"
        assert "upgraded" in metrics, "Missing 'upgraded' in /metrics response"

        # 2. Test POST /analyze across test cases and models
        test_sentences = [
            "m bahut khush hu, movie ekdum mast thi",
            "Khana bilkul thik nhi laga mujhe",
            "bhai wo banda ekdum badtamiz hai, bilkul bakwas behaviour",
            "sb log kb aur kha milenge, average plan h",
        ]

        models_to_test = ["upgraded", "baseline"]

        print("\n" + "=" * 70)
        print("[2] Testing POST /analyze on Hinglish Test Cases")
        print("=" * 70)

        for sentence_idx, sentence in enumerate(test_sentences, 1):
            print(f"\n>>> TEST CASE {sentence_idx}: \"{sentence}\"")
            print("-" * 70)

            for model_type in models_to_test:
                payload = {
                    "sentence": sentence,
                    "model_type": model_type,
                }
                response = client.post("/analyze", json=payload)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"

                data = response.json()

                # Assert essential payload fields
                assert "original" in data
                assert "baseline_pred" in data
                assert "consistency_score" in data
                assert "variations" in data
                assert "model_used" in data
                assert "negation_tagged" in data

                print(f"  Model:           {data['model_used']}")
                print(f"  Predicted Class: {data['baseline_pred']}")
                print(f"  Negation Tagged: {data['negation_tagged']}")
                print(f"  Consistency:     {data['consistency_score']}% agreement")
                print("  Variations Generated:")
                for v in data["variations"]:
                    status_symbol = "[OK]" if v["pred"] == data["baseline_pred"] else "[FLIP]"
                    print(f"    {status_symbol} \"{v['text']}\" -> {v['pred']}")
                print()

    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! (HTTP 200 on all endpoints)")
    print("=" * 70)


if __name__ == "__main__":
    test_api_suite()
