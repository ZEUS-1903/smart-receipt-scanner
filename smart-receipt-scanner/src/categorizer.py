"""
categorizer.py — Spending categorization using scikit-learn.

Uses TF-IDF vectorization + Logistic Regression to classify purchases
into categories based on store name and item descriptions.

This is a lightweight ML model that trains on your spending history
and improves as you add more receipts.
"""

import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).parent.parent / "data" / "categorizer.pkl"

# Default training data — maps keywords to categories
# This bootstraps the model before you have your own data
DEFAULT_TRAINING_DATA = {
    "groceries": [
        "whole foods market", "trader joes", "stop and shop", "market basket",
        "aldi", "costco", "walmart grocery", "kroger", "safeway", "publix",
        "wegmans", "h-e-b", "food lion", "giant", "sprouts", "fresh market",
        "milk eggs bread cheese fruit vegetables meat chicken rice pasta",
        "bananas apples oranges lettuce tomatoes onions potatoes cereal",
    ],
    "dining": [
        "starbucks coffee", "mcdonalds", "chipotle", "panera bread", "subway",
        "dunkin donuts", "pizza hut", "dominos", "burger king", "wendys",
        "restaurant cafe bistro grill diner bar pub tavern",
        "food delivery doordash uber eats grubhub postmates",
    ],
    "transport": [
        "shell gas station", "exxon mobil", "chevron bp sunoco speedway",
        "uber lyft taxi cab ride", "mbta transit metro bus subway train",
        "parking garage toll turnpike highway", "oil change auto repair",
    ],
    "healthcare": [
        "cvs pharmacy walgreens rite aid", "hospital clinic doctor physician",
        "dentist dental orthodontist", "optometrist eye glasses contacts",
        "prescription medicine vitamins supplements", "urgent care medical",
    ],
    "entertainment": [
        "amc regal cinemark movie theater cinema", "netflix hulu disney spotify",
        "concert tickets event show performance", "bowling arcade mini golf",
        "museum zoo aquarium park recreation", "books barnes noble bookstore",
    ],
    "shopping": [
        "amazon target walmart bestbuy best buy", "home depot lowes ace hardware",
        "macys nordstrom gap old navy zara h&m", "nike adidas foot locker",
        "apple store electronics computer phone", "ikea furniture home goods",
    ],
    "utilities": [
        "electric power energy eversource national grid", "gas heating",
        "water sewer", "internet cable comcast verizon att spectrum",
        "phone mobile cell wireless", "insurance premium",
    ],
    "subscriptions": [
        "netflix spotify apple music hulu disney plus", "amazon prime membership",
        "gym fitness planet fitness equinox", "adobe microsoft office 365",
        "cloud storage dropbox google icloud", "newspaper magazine subscription",
    ],
}


class SpendingCategorizer:
    """Categorizes purchases using TF-IDF + Logistic Regression."""

    def __init__(self):
        self.categories = list(DEFAULT_TRAINING_DATA.keys())
        self.pipeline = None
        self._load_or_train()

    def _load_or_train(self):
        """Load saved model or train from default data."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.pipeline = pickle.load(f)
                print("[Categorizer] Loaded saved model")
                return
            except Exception as e:
                print(f"[Categorizer] Error loading model: {e}")

        print("[Categorizer] Training new model from default data...")
        self._train_default()

    def _train_default(self):
        """Train on default keyword data."""
        texts = []
        labels = []

        for category, examples in DEFAULT_TRAINING_DATA.items():
            for example in examples:
                texts.append(example.lower())
                labels.append(category)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                lowercase=True,
                stop_words="english",
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                C=1.0,
            )),
        ])

        self.pipeline.fit(texts, labels)
        self._save()
        print(f"[Categorizer] Model trained on {len(texts)} examples")

    def _save(self):
        """Save model to disk."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)

    def categorize(self, store_name: str, items: list[dict] = None) -> tuple[str, float]:
        """
        Categorize a purchase based on store name and items.

        Args:
            store_name: Name of the store/merchant
            items: List of line items [{"name": "...", "price": ...}, ...]

        Returns:
            Tuple of (category, confidence_score)
        """
        # Build text from store name + item names
        text_parts = [store_name.lower()]
        if items:
            for item in items:
                text_parts.append(item.get("name", "").lower())
        combined_text = " ".join(text_parts)

        # Predict
        category = self.pipeline.predict([combined_text])[0]
        probabilities = self.pipeline.predict_proba([combined_text])[0]
        confidence = max(probabilities)

        return category, round(confidence, 3)

    def retrain(self, receipts: list[dict]):
        """
        Retrain the model using actual categorized receipts from the database.

        Args:
            receipts: List of dicts with 'store_name', 'items', 'category'
        """
        # Start with default data
        texts = []
        labels = []

        for category, examples in DEFAULT_TRAINING_DATA.items():
            for example in examples:
                texts.append(example.lower())
                labels.append(category)

        # Add real receipt data (weighted more heavily by duplicating)
        for receipt in receipts:
            text = receipt.get("store_name", "").lower()
            category = receipt.get("category", "uncategorized")
            if category != "uncategorized":
                # Add twice to give real data more weight
                texts.extend([text, text])
                labels.extend([category, category])

        if len(set(labels)) < 2:
            print("[Categorizer] Not enough categories to retrain")
            return

        self.pipeline.fit(texts, labels)
        self._save()
        print(f"[Categorizer] Retrained on {len(texts)} examples "
              f"({len(receipts)} real receipts)")


if __name__ == "__main__":
    cat = SpendingCategorizer()

    test_cases = [
        ("Whole Foods Market", [{"name": "organic milk"}, {"name": "avocados"}]),
        ("Starbucks", [{"name": "grande latte"}, {"name": "blueberry muffin"}]),
        ("Shell", [{"name": "unleaded gas"}]),
        ("CVS Pharmacy", [{"name": "ibuprofen"}, {"name": "bandages"}]),
        ("AMC Theaters", [{"name": "movie ticket"}, {"name": "popcorn"}]),
        ("Amazon", [{"name": "USB cable"}, {"name": "phone case"}]),
    ]

    print("\nCategorization Test:")
    print("-" * 60)
    for store, items in test_cases:
        category, confidence = cat.categorize(store, items)
        print(f"  {store:25s} → {category:15s} (confidence: {confidence:.1%})")
