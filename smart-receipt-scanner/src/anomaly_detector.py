"""
anomaly_detector.py — Spending anomaly detection using Isolation Forest.

Isolation Forest is an unsupervised ML algorithm that isolates anomalies
by randomly selecting features and split values. Anomalous data points
(unusually high spending) require fewer splits to isolate, making them
easy to detect.

This learns YOUR spending patterns and flags purchases that deviate
significantly from your norm.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Detects anomalous spending using Isolation Forest."""

    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: Expected proportion of anomalies (0.05 = 5%)
        """
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            max_samples="auto",
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.stats = {"mean": 0, "std": 0, "threshold": 0}

    def train(self, historical_totals: list[float]):
        """
        Train the anomaly detector on historical spending data.

        Args:
            historical_totals: List of past receipt totals
        """
        if len(historical_totals) < 10:
            # Not enough data for meaningful anomaly detection
            # Fall back to simple statistical threshold
            self.is_trained = False
            if historical_totals:
                self.stats["mean"] = np.mean(historical_totals)
                self.stats["std"] = np.std(historical_totals) if len(historical_totals) > 1 else 50
                self.stats["threshold"] = self.stats["mean"] + (3 * self.stats["std"])
            print(f"[AnomalyDetector] Too few samples ({len(historical_totals)}), "
                  f"using statistical fallback (threshold: ${self.stats['threshold']:.2f})")
            return

        # Reshape for sklearn (needs 2D array)
        X = np.array(historical_totals).reshape(-1, 1)

        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Train Isolation Forest
        self.model.fit(X_scaled)
        self.is_trained = True

        # Store stats for reporting
        self.stats["mean"] = np.mean(historical_totals)
        self.stats["std"] = np.std(historical_totals)
        self.stats["threshold"] = self._find_threshold(historical_totals)

        print(f"[AnomalyDetector] Trained on {len(historical_totals)} receipts")
        print(f"[AnomalyDetector] Average spending: ${self.stats['mean']:.2f}")
        print(f"[AnomalyDetector] Anomaly threshold: ~${self.stats['threshold']:.2f}")

    def _find_threshold(self, totals: list[float]) -> float:
        """Find the approximate spending threshold that triggers anomaly."""
        # Test a range of values to find the boundary
        test_range = np.linspace(0, max(totals) * 3, 100).reshape(-1, 1)
        test_scaled = self.scaler.transform(test_range)
        predictions = self.model.predict(test_scaled)

        # Find where predictions switch from normal (1) to anomaly (-1)
        for i, (val, pred) in enumerate(zip(test_range.flatten(), predictions)):
            if pred == -1:
                return float(val)
        return float(max(totals) * 2)

    def is_anomaly(self, amount: float) -> tuple[bool, float, str]:
        """
        Check if a spending amount is anomalous.

        Args:
            amount: The receipt total to check

        Returns:
            Tuple of (is_anomaly, anomaly_score, reason)
        """
        if self.is_trained:
            # Use Isolation Forest
            X = np.array([[amount]])
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            score = self.model.decision_function(X_scaled)[0]

            is_anom = prediction == -1
            reason = ""
            if is_anom:
                how_much = amount / self.stats["mean"] if self.stats["mean"] > 0 else 0
                reason = (f"${amount:.2f} is {how_much:.1f}x your average "
                         f"(${self.stats['mean']:.2f})")

            return is_anom, float(score), reason
        else:
            # Statistical fallback
            if self.stats["threshold"] > 0 and amount > self.stats["threshold"]:
                how_much = amount / self.stats["mean"] if self.stats["mean"] > 0 else 0
                reason = (f"${amount:.2f} exceeds threshold "
                         f"(${self.stats['threshold']:.2f})")
                return True, -1.0, reason
            return False, 1.0, ""

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "is_trained": self.is_trained,
            "average_spending": round(self.stats["mean"], 2),
            "std_deviation": round(self.stats["std"], 2),
            "anomaly_threshold": round(self.stats["threshold"], 2),
            "contamination_rate": self.contamination,
        }


if __name__ == "__main__":
    # Demo with synthetic data
    import random
    random.seed(42)

    # Simulate 100 normal purchases + a few anomalies
    normal = [random.gauss(45, 20) for _ in range(100)]
    normal = [max(5, x) for x in normal]  # No negative amounts

    detector = AnomalyDetector()
    detector.train(normal)

    print("\nTesting anomaly detection:")
    print("-" * 60)
    test_amounts = [25.0, 45.0, 80.0, 150.0, 300.0, 500.0]
    for amount in test_amounts:
        is_anom, score, reason = detector.is_anomaly(amount)
        flag = "⚠️  ANOMALY" if is_anom else "   Normal"
        print(f"  ${amount:8.2f}  {flag}  (score: {score:.3f})")
        if reason:
            print(f"              → {reason}")
