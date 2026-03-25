"""
agent.py — The Agent orchestrator.

This is the brain that ties everything together:
1. Takes a receipt image
2. Runs the Donut DL model to extract data
3. Categorizes the purchase with the ML classifier
4. Checks for anomalies with Isolation Forest
5. Stores everything in the database
6. Returns a complete analysis

Think of this as the "agentic" layer — it decides what to do,
in what order, and handles edge cases.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from receipt_parser import ReceiptParser
from categorizer import SpendingCategorizer
from anomaly_detector import AnomalyDetector
from database import (
    save_receipt, get_all_receipts, get_receipt_with_items,
    get_spending_by_category, get_daily_spending, get_spending_totals,
    get_all_totals_for_anomaly, get_budgets, set_budget,
)


class ReceiptAgent:
    """
    The main agent that orchestrates receipt scanning and expense tracking.

    Pipeline:
        Image → Donut Model → Categorizer → Anomaly Detector → Database → Report
    """

    def __init__(self, device: str = None, lazy_load: bool = True):
        """
        Initialize the agent.

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            lazy_load: If True, don't load the heavy Donut model until first use
        """
        self.device = device
        self._parser = None
        self._categorizer = None
        self._anomaly_detector = None

        if not lazy_load:
            self._init_all()

    def _init_all(self):
        """Initialize all components."""
        self._get_parser()
        self._get_categorizer()
        self._get_anomaly_detector()

    def _get_parser(self) -> ReceiptParser:
        """Lazy-load the Donut model."""
        if self._parser is None:
            print("\n[Agent] Initializing Donut receipt parser...")
            self._parser = ReceiptParser(device=self.device)
        return self._parser

    def _get_categorizer(self) -> SpendingCategorizer:
        """Lazy-load the categorizer."""
        if self._categorizer is None:
            print("[Agent] Initializing spending categorizer...")
            self._categorizer = SpendingCategorizer()
        return self._categorizer

    def _get_anomaly_detector(self) -> AnomalyDetector:
        """Lazy-load and train the anomaly detector."""
        if self._anomaly_detector is None:
            print("[Agent] Initializing anomaly detector...")
            self._anomaly_detector = AnomalyDetector()

            # Train on historical data
            historical = get_all_totals_for_anomaly()
            if historical:
                self._anomaly_detector.train(historical)
            else:
                print("[Agent] No historical data yet — anomaly detection will "
                      "improve as you scan more receipts.")
        return self._anomaly_detector

    def process_receipt(self, image_path: str) -> dict:
        """
        Main agent pipeline: process a receipt image end-to-end.

        Steps:
            1. Validate the image file
            2. Run Donut DL model to extract receipt data
            3. Categorize the purchase
            4. Check for spending anomalies
            5. Save to database
            6. Check budget status
            7. Return complete analysis

        Args:
            image_path: Path to receipt image (jpg, png, etc.)

        Returns:
            Complete analysis dict with all extracted data, category,
            anomaly info, and budget status.
        """
        print(f"\n{'='*60}")
        print(f"[Agent] Processing receipt: {image_path}")
        print(f"{'='*60}")

        # Step 1: Validate
        path = Path(image_path)
        if not path.exists():
            return {"error": f"File not found: {image_path}"}
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            return {"error": f"Unsupported image format: {path.suffix}"}

        # Step 2: Parse receipt with Donut DL model
        print("\n[Agent] Step 1/5: Running Donut model inference...")
        parser = self._get_parser()
        parsed = parser.parse_receipt(image_path)

        print(f"[Agent] Extracted: {parsed['store_name']} | "
              f"${parsed['total']:.2f} | {len(parsed['items'])} items")

        # Step 3: Categorize
        print("[Agent] Step 2/5: Categorizing purchase...")
        categorizer = self._get_categorizer()
        category, confidence = categorizer.categorize(
            parsed["store_name"],
            parsed["items"]
        )
        print(f"[Agent] Category: {category} (confidence: {confidence:.1%})")

        # Step 4: Anomaly check
        print("[Agent] Step 3/5: Checking for anomalies...")
        detector = self._get_anomaly_detector()
        is_anomaly, anomaly_score, anomaly_reason = detector.is_anomaly(parsed["total"])
        if is_anomaly:
            print(f"[Agent] ⚠️  ANOMALY DETECTED: {anomaly_reason}")
        else:
            print(f"[Agent] Spending looks normal.")

        # Step 5: Save to database
        print("[Agent] Step 4/5: Saving to database...")
        receipt_id = save_receipt(parsed, category=category, is_anomaly=is_anomaly)
        print(f"[Agent] Saved as receipt #{receipt_id}")

        # Step 6: Check budgets
        print("[Agent] Step 5/5: Checking budget status...")
        budget_status = self._check_budget(category)

        # Build complete analysis
        analysis = {
            "receipt_id": receipt_id,
            "store_name": parsed["store_name"],
            "date": parsed["date"],
            "items": parsed["items"],
            "subtotal": parsed["subtotal"],
            "tax": parsed["tax"],
            "total": parsed["total"],
            "category": category,
            "category_confidence": confidence,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "anomaly_reason": anomaly_reason,
            "budget_status": budget_status,
            "processing_time": parsed.get("processing_time_seconds", 0),
        }

        # Print summary
        self._print_summary(analysis)

        return analysis

    def _check_budget(self, category: str) -> Optional[dict]:
        """Check if spending in this category is within budget."""
        budgets = get_budgets()
        for budget in budgets:
            if budget["category"] == category:
                remaining = budget["monthly_limit"] - budget["spent"]
                pct_used = (budget["spent"] / budget["monthly_limit"] * 100
                           if budget["monthly_limit"] > 0 else 0)
                status = {
                    "category": category,
                    "monthly_limit": budget["monthly_limit"],
                    "spent_this_month": budget["spent"],
                    "remaining": remaining,
                    "percent_used": round(pct_used, 1),
                    "over_budget": remaining < 0,
                }
                if pct_used >= 90:
                    status["warning"] = f"⚠️ {pct_used:.0f}% of {category} budget used!"
                return status
        return None

    def _print_summary(self, analysis: dict):
        """Print a formatted summary of the analysis."""
        print(f"\n{'='*60}")
        print(f"  RECEIPT ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"  Store:      {analysis['store_name']}")
        print(f"  Date:       {analysis['date']}")
        print(f"  Total:      ${analysis['total']:.2f}")
        print(f"  Category:   {analysis['category']} "
              f"({analysis['category_confidence']:.0%} confidence)")
        print(f"  Items:      {len(analysis['items'])}")

        if analysis["is_anomaly"]:
            print(f"  ⚠️  ANOMALY: {analysis['anomaly_reason']}")

        if analysis.get("budget_status"):
            bs = analysis["budget_status"]
            print(f"  Budget:     ${bs['spent_this_month']:.2f} / "
                  f"${bs['monthly_limit']:.2f} "
                  f"({bs['percent_used']:.0f}% used)")
            if bs.get("warning"):
                print(f"  {bs['warning']}")

        print(f"  Time:       {analysis['processing_time']:.2f}s")
        print(f"{'='*60}\n")

    def get_dashboard_data(self) -> dict:
        """Get all data needed for the dashboard."""
        return {
            "totals": get_spending_totals(),
            "by_category": get_spending_by_category(days=30),
            "daily": get_daily_spending(days=30),
            "recent_receipts": get_all_receipts(limit=20),
            "budgets": get_budgets(),
            "anomaly_stats": self._get_anomaly_detector().get_stats(),
        }


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Receipt Scanner Agent")
    ap.add_argument("--image", help="Path to receipt image")
    ap.add_argument("--demo", action="store_true", help="Run with demo data")
    ap.add_argument("--device", default=None, help="Device: cuda or cpu")
    args = ap.parse_args()

    if args.demo:
        from database import seed_demo_data
        seed_demo_data()
        agent = ReceiptAgent(device=args.device)
        data = agent.get_dashboard_data()
        print("\nDashboard Data:")
        print(json.dumps(data, indent=2, default=str))
    elif args.image:
        agent = ReceiptAgent(device=args.device)
        result = agent.process_receipt(args.image)
    else:
        print("Usage:")
        print("  python agent.py --image receipt.jpg   # Scan a receipt")
        print("  python agent.py --demo                # Load demo data")
