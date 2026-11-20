import unittest

from fastapi.testclient import TestClient

from src.transaction_labeling.api import app, bundle, manifest


class PredictionApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_reports_loaded_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "ok",
            "model": manifest["model_type"],
        })

    def test_predict_returns_two_coherent_categories(self):
        response = self.client.post("/predict", json={
            "title": "UNKNOWN TEST MERCHANT 8472",
            "amount_value": -18.4,
            "created_at": "2026-06-11T12:30:00Z",
            "card_name": "unknown test card",
            "digitalization_type": "unknown wallet",
            "merchant_pan_entry_method": "unknown entry method",
        })

        self.assertEqual(response.status_code, 200, response.text)
        prediction = response.json()
        self.assertEqual(set(prediction), {"minor_category", "major_category"})
        self.assertIn(prediction["minor_category"], bundle["classes"])
        self.assertEqual(
            prediction["major_category"],
            bundle["major_category_mapping"][prediction["minor_category"]],
        )

    def test_predict_rejects_invalid_payload(self):
        response = self.client.post("/predict", json={
            "title": "STORE",
            "amount_value": -18.4,
            "created_at": "2026-06-11T12:30:00",
        })

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
