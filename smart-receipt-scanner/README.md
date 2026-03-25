# smart-receipt-scanner

I got tired of manually entering receipts into spreadsheets. So I built a thing.

This is a Python agent that scans receipt photos, pulls out the store name, items, prices, and total using a deep learning model (Donut — a vision transformer), then categorizes the spending and tracks it over time. No cloud APIs. Everything runs on your machine.

![Python](https://img.shields.io/badge/python-3.9+-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## what it does

You give it a receipt photo. It gives you back structured data:

```
$ python src/agent.py --image receipt.jpg

══════════════════════════════════════════
  RECEIPT ANALYSIS COMPLETE
══════════════════════════════════════════
  Store:      Trader Joe's
  Date:       2026-03-15
  Total:      $47.83
  Category:   groceries (91% confidence)
  Items:      7
  Budget:     $312 / $600 (52% used)
══════════════════════════════════════════
```

Then it stores everything in a local SQLite database and gives you a dashboard to see where your money's going.

## how it works

There are three ML models working together:

**1. Donut** (the heavy lifter)
- A vision transformer — Swin encoder + BART decoder
- Looks at the receipt image and outputs structured JSON
- No OCR involved. The model reads the image end-to-end
- I'm using `AdamCodd/donut-receipts-extract`, fine-tuned on ~1,100 receipts
- Runs via PyTorch, works on CPU (slower) or GPU (faster)

**2. TF-IDF + Logistic Regression** (the categorizer)
- Takes the store name + item names
- Classifies into: groceries, dining, transport, healthcare, entertainment, shopping, utilities
- Trained on keyword data, gets smarter as you use it
- scikit-learn, nothing fancy

**3. Isolation Forest** (the anomaly detector)
- Learns your normal spending patterns
- Flags anything unusually high
- "You spent $340 at Target — that's 4.2x your average"

```
receipt image
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Donut     │ ──▶ │  Categorizer │ ──▶ │  Anomaly    │
│  (PyTorch)  │     │  (sklearn)   │     │  Detector   │
└─────────────┘     └──────────────┘     └─────────────┘
     │                    │                     │
     └────────────────────┴─────────────────────┘
                          │
                    ┌─────▼─────┐
                    │  SQLite   │
                    │  Database │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ Streamlit │
                    │ Dashboard │
                    └───────────┘
```

## setup

```bash
git clone https://github.com/YOUR_USERNAME/smart-receipt-scanner.git
cd smart-receipt-scanner

python -m venv venv
source venv/bin/activate   # windows: venv\Scripts\activate

pip install -r requirements.txt
```

First run downloads the Donut model (~800MB). After that it's cached.

## usage

**Scan a receipt:**
```bash
python src/agent.py --image path/to/receipt.jpg
```

**Launch the dashboard:**
```bash
# seed some demo data first (optional)
python src/database.py

# start the dashboard
streamlit run src/dashboard.py
```

**Just the parser (no agent logic):**
```bash
python src/receipt_parser.py --image receipt.jpg
```

## project structure

```
smart-receipt-scanner/
├── src/
│   ├── receipt_parser.py    # donut model inference
│   ├── categorizer.py       # spending classification
│   ├── anomaly_detector.py  # unusual spending detection
│   ├── database.py          # sqlite storage + queries
│   ├── agent.py             # orchestrator
│   └── dashboard.py         # streamlit ui
├── data/                    # sqlite db lives here
├── tests/
│   └── test_agent.py
├── requirements.txt
└── README.md
```

## hardware

- **minimum:** 8GB RAM, any modern CPU. inference takes ~10-15 seconds per receipt
- **recommended:** 16GB RAM + NVIDIA GPU with CUDA. inference drops to ~2 seconds
- **storage:** ~1GB for model weights on first download

## limitations (being honest)

- The Donut model was trained on ~1,100 receipts. Crumpled, faded, or unusual layouts will trip it up
- Handwritten receipts don't work well
- The categorizer is keyword-based — if you shop at a store it's never seen, it guesses
- Anomaly detection needs at least 10 receipts before it's useful

## what I learned

- Donut's end-to-end approach is way more robust than the Tesseract → regex → pray pipeline I tried first
- The agent pattern (orchestrating multiple models) is how real ML systems work — it's never just one model
- Feature engineering in the categorizer matters more than the model choice
- SQLite is underrated for small projects. Zero setup, just works

## tech

python · pytorch · huggingface transformers · scikit-learn · streamlit · sqlite · pillow · opencv
