"""
receipt_parser.py — Core DL engine using Donut (Document Understanding Transformer)

The Donut model uses a Swin Transformer encoder to process receipt images
and a BART decoder to generate structured JSON output — no OCR needed.

Model: AdamCodd/donut-receipts-extract (fine-tuned on ~1100 receipts)
Fallback: naver-clova-ix/donut-base-finetuned-cord-v2 (CORD dataset)
"""

import re
import json
import torch
from PIL import Image
from pathlib import Path
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datetime import datetime


class ReceiptParser:
    """Parses receipt images using the Donut deep learning model."""

    def __init__(self, model_name: str = "AdamCodd/donut-receipts-extract", device: str = None):
        """
        Initialize the Donut model for receipt parsing.

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[ReceiptParser] Loading model: {model_name}")
        print(f"[ReceiptParser] Device: {self.device}")

        # Load the processor (handles image preprocessing + token decoding)
        self.processor = DonutProcessor.from_pretrained(model_name)

        # Load the Vision Encoder-Decoder model
        # Encoder: Swin Transformer (processes image patches)
        # Decoder: BART (generates structured text output)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to inference mode

        print(f"[ReceiptParser] Model loaded successfully!")
        print(f"[ReceiptParser] Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        Load and preprocess a receipt image.

        Args:
            image_path: Path to the receipt image file

        Returns:
            PIL Image in RGB format
        """
        image = Image.open(image_path).convert("RGB")

        # Basic preprocessing: ensure reasonable size
        # Donut handles its own resizing, but very large images slow things down
        max_dim = 2048
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        return image

    @torch.no_grad()  # Disable gradient computation for inference
    def parse_receipt(self, image_path: str) -> dict:
        """
        Parse a receipt image and extract structured data.

        This is where the deep learning magic happens:
        1. Image → Swin Transformer encoder → image embeddings
        2. Image embeddings + task prompt → BART decoder → structured tokens
        3. Tokens → decoded text → parsed JSON

        Args:
            image_path: Path to receipt image

        Returns:
            Dictionary with extracted receipt data:
            {
                "store_name": str,
                "date": str,
                "items": [{"name": str, "price": float, "quantity": int}, ...],
                "subtotal": float,
                "tax": float,
                "total": float,
                "raw_output": str  # Raw model output for debugging
            }
        """
        print(f"\n[ReceiptParser] Processing: {image_path}")
        start_time = datetime.now()

        # Step 1: Load and preprocess image
        image = self.preprocess_image(image_path)

        # Step 2: Prepare inputs for the model
        # The task prompt tells Donut what to extract
        task_prompt = "<s_receipt>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Process image into pixel values (tensor)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Step 3: Run inference
        # The model autoregressively generates tokens conditioned on the image
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Step 4: Decode the generated tokens back to text
        raw_output = self.processor.batch_decode(outputs.sequences)[0]

        # Step 5: Clean up and parse the output
        # Remove special tokens and task prompt
        raw_output = raw_output.replace(self.processor.tokenizer.eos_token, "")
        raw_output = raw_output.replace(self.processor.tokenizer.pad_token, "")

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[ReceiptParser] Inference completed in {elapsed:.2f}s")
        print(f"[ReceiptParser] Raw output: {raw_output[:200]}...")

        # Parse the structured output into a clean dictionary
        parsed = self._parse_model_output(raw_output)
        parsed["raw_output"] = raw_output
        parsed["processing_time_seconds"] = elapsed
        parsed["image_path"] = str(image_path)

        return parsed

    def _parse_model_output(self, raw_output: str) -> dict:
        """
        Parse the Donut model's raw output into a structured dictionary.

        The model outputs XML-like tags:
        <s_store_name>STORE NAME</s_store_name>
        <s_date>2024-01-15</s_date>
        <s_line_items><s_item>...</s_item></s_line_items>
        etc.

        We extract these into a clean Python dict.
        """
        result = {
            "store_name": self._extract_tag(raw_output, "store_name") or "Unknown Store",
            "date": self._extract_tag(raw_output, "date") or datetime.now().strftime("%Y-%m-%d"),
            "address": self._extract_tag(raw_output, "address") or "",
            "phone": self._extract_tag(raw_output, "phone") or "",
            "items": [],
            "subtotal": 0.0,
            "tax": 0.0,
            "total": 0.0,
            "tips": 0.0,
            "discount": 0.0,
        }

        # Extract monetary values
        for field in ["subtotal", "tax", "total", "tips", "discount", "svc"]:
            value = self._extract_tag(raw_output, field)
            if value:
                try:
                    # Clean and parse the numeric value
                    cleaned = re.sub(r'[^\d.]', '', value)
                    if cleaned:
                        result[field] = float(cleaned)
                except (ValueError, TypeError):
                    pass

        # Extract line items
        items_raw = self._extract_all_items(raw_output)
        for item in items_raw:
            parsed_item = {
                "name": self._extract_tag(item, "nm") or "Unknown Item",
                "quantity": 1,
                "price": 0.0,
            }

            qty = self._extract_tag(item, "cnt") or self._extract_tag(item, "qty")
            if qty:
                try:
                    parsed_item["quantity"] = int(re.sub(r'[^\d]', '', qty))
                except (ValueError, TypeError):
                    pass

            price = self._extract_tag(item, "price") or self._extract_tag(item, "unitprice")
            if price:
                try:
                    parsed_item["price"] = float(re.sub(r'[^\d.]', '', price))
                except (ValueError, TypeError):
                    pass

            result["items"].append(parsed_item)

        # If total is 0 but we have items, calculate it
        if result["total"] == 0 and result["items"]:
            result["total"] = sum(
                item["price"] * item["quantity"] for item in result["items"]
            )

        return result

    def _extract_tag(self, text: str, tag_name: str) -> str | None:
        """Extract content from a Donut-style tag."""
        pattern = rf'<s_{tag_name}>(.*?)</s_{tag_name}>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_all_items(self, text: str) -> list[str]:
        """Extract all item blocks from the model output."""
        # Try different item tag patterns the model might use
        items = re.findall(r'<s_item>(.*?)</s_item>', text, re.DOTALL)
        if not items:
            items = re.findall(r'<s_menu_item>(.*?)</s_menu_item>', text, re.DOTALL)
        return items


def parse_receipt_from_image(image_path: str) -> dict:
    """
    Convenience function: parse a single receipt image.

    Usage:
        result = parse_receipt_from_image("receipt.jpg")
        print(result["store_name"])
        print(result["total"])
        for item in result["items"]:
            print(f"  {item['name']}: ${item['price']}")
    """
    parser = ReceiptParser()
    return parser.parse_receipt(image_path)


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse a receipt image using Donut DL model")
    ap.add_argument("--image", required=True, help="Path to receipt image (jpg/png)")
    ap.add_argument("--model", default="AdamCodd/donut-receipts-extract",
                    help="HuggingFace model name")
    ap.add_argument("--device", default=None, help="Device: cuda or cpu")
    args = ap.parse_args()

    parser = ReceiptParser(model_name=args.model, device=args.device)
    result = parser.parse_receipt(args.image)

    print("\n" + "=" * 50)
    print("PARSED RECEIPT")
    print("=" * 50)
    print(f"Store:    {result['store_name']}")
    print(f"Date:     {result['date']}")
    print(f"Items:")
    for item in result["items"]:
        print(f"  - {item['name']} x{item['quantity']}  ${item['price']:.2f}")
    print(f"Subtotal: ${result['subtotal']:.2f}")
    print(f"Tax:      ${result['tax']:.2f}")
    print(f"Total:    ${result['total']:.2f}")
    print(f"\nProcessing time: {result['processing_time_seconds']:.2f}s")
    print(f"\nFull JSON:")
    print(json.dumps(result, indent=2, default=str))
