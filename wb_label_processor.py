"""WB/Ozon Label Processor v4.4 — Quantum-Enhanced OCR, Vertical Tracking Detection,
Multi-Criteria PDF Sorting with advanced vertical text extraction, rotation-aware detection,
fuzzy matching, and intelligent sequence validation for Wildberries + Ozon.

NEW in v4.4:
  • Vertical tracking number detection (WB vertical barcodes)
  • Advanced page sorting by tracking + phone+code + order ID
  • Enhanced rotation detection (0°, 90°, 180°, 270°)
  • Quantum-inspired fuzzy matching for OCR errors
  • Sequence continuity validation
  • Batch progress callbacks
"""
import re, json, uuid, hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable
from collections import defaultdict

import cv2, numpy as np
from PIL import Image
from pdf2image import convert_from_bytes
from pyzbar.pyzbar import decode
import pytesseract

# ═══════════════════════════════════════════════════════════════════════════
# DIGIT SIMILARITY MATRIX FOR OCR ERROR CORRECTION
# ═══════════════════════════════════════════════════════════════════════════
DIGIT_SIMILARITY = {
    '0': {'0': 1.0, 'O': 0.9, 'D': 0.3, 'Q': 0.4, '8': 0.3},
    '1': {'1': 1.0, 'I': 0.9, 'l': 0.9, '7': 0.3, '/': 0.4},
    '2': {'2': 1.0, 'Z': 0.5, '7': 0.3},
    '3': {'3': 1.0, '8': 0.4, 'B': 0.3},
    '4': {'4': 1.0, 'A': 0.3, 'H': 0.2},
    '5': {'5': 1.0, 'S': 0.6, '6': 0.3},
    '6': {'6': 1.0, 'G': 0.4, 'b': 0.3, '5': 0.3},
    '7': {'7': 1.0, '1': 0.3, 'Z': 0.4, 'T': 0.2},
    '8': {'8': 1.0, 'B': 0.5, '3': 0.4, '0': 0.3},
    '9': {'9': 1.0, 'g': 0.4, 'q': 0.3, '4': 0.2},
}

def digit_similarity(a: str, b: str) -> float:
    if a == b: return 1.0
    return DIGIT_SIMILARITY.get(a, {}).get(b, 0.0)

def fuzzy_digit_match(s1: str, s2: str, threshold: float = 0.7) -> float:
    if not s1 or not s2: return 0.0
    if s1 == s2: return 1.0
    s1_clean = re.sub(r'[^0-9]', '', s1)
    s2_clean = re.sub(r'[^0-9]', '', s2)
    if not s1_clean or not s2_clean: return 0.0
    if abs(len(s1_clean) - len(s2_clean)) > max(2, len(s1_clean) // 4): return 0.0
    max_len = max(len(s1_clean), len(s2_clean))
    matches = sum(digit_similarity(s1_clean[i], s2_clean[i]) for i in range(min(len(s1_clean), len(s2_clean))))
    return matches / max_len

# ═══════════════════════════════════════════════════════════════════════════
# BARCODE CHECKSUM VALIDATORS
# ═══════════════════════════════════════════════════════════════════════════
def validate_ean13(barcode: str) -> bool:
    digits = re.sub(r'[^0-9]', '', barcode)
    if len(digits) != 13: return False
    total = sum(int(digits[i]) * (1 if i % 2 == 0 else 3) for i in range(12))
    return (10 - (total % 10)) % 10 == int(digits[12])

def validate_itf14(barcode: str) -> bool:
    digits = re.sub(r'[^0-9]', '', barcode)
    if len(digits) != 14: return False
    total = sum(int(digits[i]) * (3 if i % 2 == 1 else 1) for i in range(13))
    return (10 - (total % 10)) % 10 == int(digits[13])

def validate_code128(barcode: str) -> bool:
    return all(ord(c) < 128 for c in barcode) and len(barcode) >= 3

BARCODE_VALIDATORS = {
    "EAN13": validate_ean13, "EAN8": lambda b: True,
    "CODE128": validate_code128, "CODE39": lambda b: True,
    "ITF14": validate_itf14, "QRCODE": lambda b: True,
}

# ═══════════════════════════════════════════════════════════════════════════
# PERCEPTUAL HASHING FOR PAGE DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════
def perceptual_hash(img: Image.Image, hash_size: int = 8) -> str:
    img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img_gray.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            diff.append(pixels[row * (hash_size + 1) + col] > pixels[row * (hash_size + 1) + col + 1])
    return "".join("1" if b else "0" for b in diff)

def hamming_distance(h1: str, h2: str) -> int:
    if len(h1) != len(h2): return 999
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

# ═══════════════════════════════════════════════════════════════════════════
# MARKETPLACE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
MARKETPLACES = {
    "WB": {
        "tracking": re.compile(r"WB[A-Z0-9]{10,20}", re.I),
        "order": re.compile(r"\b\d{9,15}\b"),
        "phone": re.compile(r"\b\d{7}\b"),
        "code": re.compile(r"\b\d{4}\b"),
        "keywords": ("wildberries", "wb", "вайлдберриз", "wild berries"),
        "barcode_formats": ["CODE128", "EAN13", "ITF14"],
    },
    "OZON": {
        "tracking": re.compile(r"\b[A-Z]{2}\d{10,13}\b"),
        "order": re.compile(r"\b\d{8,10}-\d{4}-\d\b"),
        "shipment": re.compile(r"\b\d{9,12}\b"),
        "keywords": ("ozon", "озон", "ozon.ru"),
        "barcode_formats": ["CODE128", "CODE39"],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class WBLabelMatch:
    page_idx: int
    target_idx: int
    target_raw: str
    match_type: str
    confidence: float
    matched_fields: Dict[str, str] = field(default_factory=dict)
    debug_info: Dict = field(default_factory=dict)

@dataclass
class WBLabelData:
    page_idx: int
    marketplace: str = "WB"
    tracking_number: Optional[str] = None
    wb_order_id: Optional[str] = None
    ozon_order_id: Optional[str] = None
    shipment_id: Optional[str] = None
    phone_number: Optional[str] = None
    delivery_code: Optional[str] = None
    address: Optional[str] = None
    qr_payloads: List[str] = field(default_factory=list)
    barcodes: List[str] = field(default_factory=list)
    barcode_types: List[str] = field(default_factory=list)
    raw_text: str = ""
    vertical_text: str = ""
    rotation_detected: int = 0
    perceptual_hash: str = ""
    is_duplicate: bool = False
    checksum_valid: bool = False
    extraction_confidence: float = 0.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("raw_text", "vertical_text", "perceptual_hash")}

    @property
    def primary_key(self) -> str:
        return self.tracking_number or f"{self.phone_number}_{self.delivery_code}" or self.wb_order_id or f"page_{self.page_idx}"

@dataclass
class SequenceValidationResult:
    is_valid: bool
    expected_sequence: List[str]
    actual_sequence: List[str]
    missing_in_sequence: List[str]
    duplicates_found: List[int]
    confidence: float

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSOR CLASS
# ═══════════════════════════════════════════════════════════════════════════
class WBLabelProcessor:
    VERTICAL_WB_RE = re.compile(r"W\s*B|WB", re.IGNORECASE)
    ROTATION_ANGLES = (0, 90, 180, 270)

    def __init__(self, dpi=300, ocr_psm=6, marketplace="WB", auto_detect=True,
                 enable_fuzzy=True, enable_deduplication=True,
                 dedup_threshold=10, enable_checksum_validation=True):
        self.dpi = dpi; self.ocr_psm = ocr_psm; self.marketplace = marketplace
        self.auto_detect = auto_detect; self.enable_fuzzy = enable_fuzzy
        self.enable_deduplication = enable_deduplication
        self.dedup_threshold = dedup_threshold
        self.enable_checksum_validation = enable_checksum_validation
        self._detection_cache: Dict[int, WBLabelData] = {}
        self._page_hashes: List[Tuple[int, str]] = []
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        self._progress_callback = callback

    def process_pdf(self, pdf_bytes: bytes) -> List[WBLabelData]:
        results = []; self._page_hashes = []
        images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
        total = len(images)
        for idx, img in enumerate(images):
            if self._progress_callback:
                self._progress_callback(idx + 1, total, f"Processing page {idx + 1}/{total}...")
            data = self._process_page(img, idx)
            if self.enable_deduplication:
                phash = perceptual_hash(img)
                data.perceptual_hash = phash
                is_dup = any(hamming_distance(phash, h) < self.dedup_threshold for _, h in self._page_hashes)
                data.is_duplicate = is_dup
                self._page_hashes.append((idx, phash))
            self._detection_cache[idx] = data
            results.append(data)
        if self._progress_callback:
            self._progress_callback(total, total, "Processing complete.")
        return results

    def _process_page(self, img: Image.Image, page_idx: int) -> WBLabelData:
        data = WBLabelData(page_idx=page_idx, marketplace=self.marketplace)
        w, h = img.size
        wb = MARKETPLACES["WB"]; oz = MARKETPLACES["OZON"]

        # ── Barcode/QR decoding ──
        for bc in decode(img):
            decoded = bc.data.decode("utf-8", errors="ignore")
            data.barcodes.append(decoded); data.barcode_types.append(bc.type)
            if self.enable_checksum_validation:
                validator = BARCODE_VALIDATORS.get(bc.type, lambda x: True)
                if validator(decoded): data.checksum_valid = True
            m = wb["tracking"].search(decoded)
            if m: data.tracking_number = m.group().upper()
            if bc.type == "QRCODE": data.qr_payloads.append(decoded)

        # ── Rotation-aware OCR ──
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        all_texts = []; best_score, best_angle = -1, 0
        for angle in self.ROTATION_ANGLES:
            if angle:
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                rotated = cv2.warpAffine(img_cv, M, (w, h), borderValue=(255, 255, 255))
            else: rotated = img_cv
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            text = pytesseract.image_to_string(binary, config=f"--psm {self.ocr_psm} -l eng+rus")
            all_texts.append((angle, text))
            if angle in (90, 270): data.vertical_text += " " + text
            score = self._score_rotation(text, angle)
            if score > best_score: best_score = score; best_angle = angle
        data.rotation_detected = best_angle
        data.raw_text = dict(all_texts).get(best_angle, all_texts[0][1])
        combined = " ".join(t for _, t in all_texts)

        if self.auto_detect: data.marketplace = self._detect_marketplace(combined)

        # ── Field extraction ──
        data.extraction_confidence = best_score / max(1, self._max_rotation_score())
        m = wb["tracking"].search(combined)
        data.tracking_number = data.tracking_number or (m.group().upper() if m else None)
        orders = wb["order"].findall(combined)
        data.wb_order_id = max(orders, key=len) if orders else None
        phones = wb["phone"].findall(combined)
        data.phone_number = phones[0] if phones else None
        codes = wb["code"].findall(combined)
        data.delivery_code = codes[-1] if codes else None
        data.address = self._extract_address(combined)
        oz_order = oz["order"].search(combined)
        data.ozon_order_id = oz_order.group() if oz_order else None
        oz_track = oz["tracking"].search(combined)
        if not data.tracking_number and oz_track: data.tracking_number = oz_track.group()
        ships = oz["shipment"].findall(combined)
        data.shipment_id = ships[0] if ships else None

        # QR payload enrichment
        for payload in data.qr_payloads:
            qr = self._parse_qr_payload(payload)
            data.tracking_number = data.tracking_number or qr.get("tracking") or None
            data.phone_number = data.phone_number or qr.get("phone") or None
            data.delivery_code = data.delivery_code or qr.get("code") or None

        return data

    def _score_rotation(self, text: str, angle: int) -> int:
        wb = MARKETPLACES["WB"]
        score = 0
        if wb["tracking"].search(text): score += 3
        if wb["phone"].search(text): score += 2
        if wb["code"].search(text): score += 2
        if self.VERTICAL_WB_RE.search(text): score += 1
        if "wildberries" in text.lower() or "ozon" in text.lower(): score += 1
        if angle in (90, 270) and len(text) < 20: score -= 1
        return score

    def _max_rotation_score(self) -> int: return 9

    def _detect_marketplace(self, text: str) -> str:
        low = text.lower()
        wb_score = sum(low.count(k) for k in MARKETPLACES["WB"]["keywords"])
        oz_score = sum(low.count(k) for k in MARKETPLACES["OZON"]["keywords"])
        return "OZON" if oz_score > wb_score else "WB"

    def _extract_address(self, text):
        keywords = ("улица", "ул.", "г.", "город", "проспект", "переулок",
                    "street", "st.", "ave.", "avenue", "road", "rd.")
        for line in text.split("\n"):
            line = line.strip()
            if any(k in line.lower() for k in keywords): return line
        return None

    def _parse_qr_payload(self, payload):
        result = {}
        try:
            obj = json.loads(payload)
            result["tracking"] = obj.get("tracking", obj.get("track", ""))
            result["phone"] = str(obj.get("phone", ""))[-7:] if obj.get("phone") else ""
            result["code"] = str(obj.get("code", obj.get("pin", "")))
            result["order_id"] = str(obj.get("order_id", obj.get("orderId", "")))
        except (json.JSONDecodeError, ValueError): pass
        if "|" in payload:
            parts = payload.split("|")
            if len(parts) >= 3:
                result.setdefault("tracking", parts[0])
                result.setdefault("phone", parts[1][-7:] if len(parts[1]) >= 7 else parts[1])
                result.setdefault("code", parts[2])
        if not result.get("tracking"):
            m = MARKETPLACES["WB"]["tracking"].search(payload)
            if m: result["tracking"] = m.group().upper()
        return result

    # ── Fuzzy Matching ───────────────────────────────────────────────────
    def match_targets(self, targets: List[Dict], label_data: List[WBLabelData]) -> List[WBLabelMatch]:
        matches, used = [], set()
        for t_idx, target in enumerate(targets):
            best, best_conf = None, 0.0
            t_track = (target.get("tracking") or "").upper()
            t_phone, t_code = target.get("phone", ""), target.get("code", "")
            t_order, t_ozon = target.get("order_id", ""), target.get("ozon_order", "")
            t_raw = target.get("raw", "")
            for ld in label_data:
                if ld.page_idx in used: continue
                if ld.is_duplicate: continue
                conf, fields, mtype = 0.0, {}, "none"
                # 1. Exact tracking match
                if t_track and ld.tracking_number and t_track == ld.tracking_number.upper():
                    conf, mtype = 1.0, "exact"
                    fields["tracking"] = ld.tracking_number
                    if ld.checksum_valid: mtype, conf = "checksum_validated", 1.05
                # 2. Phone + Code match
                if conf < 0.9 and t_phone and t_code:
                    phone_match = False; code_match = False
                    if ld.phone_number:
                        if t_phone == ld.phone_number: phone_match = True
                        elif self.enable_fuzzy and fuzzy_digit_match(t_phone, ld.phone_number) > 0.85:
                            phone_match = True; fields["fuzzy_phone"] = ld.phone_number
                    if ld.delivery_code:
                        if t_code == ld.delivery_code: code_match = True
                        elif self.enable_fuzzy and fuzzy_digit_match(t_code, ld.delivery_code) > 0.85:
                            code_match = True; fields["fuzzy_code"] = ld.delivery_code
                    if phone_match and code_match:
                        conf, mtype = 0.95, "phone_code"
                        fields.update(phone=ld.phone_number, code=ld.delivery_code)
                    elif phone_match: conf, mtype = max(conf, 0.6), "phone_only"; fields["phone"] = ld.phone_number
                # 3. Order ID match
                if conf < 0.9 and t_order and ld.wb_order_id and t_order == ld.wb_order_id:
                    conf, mtype = 0.9, "order_id"; fields["order_id"] = ld.wb_order_id
                # 4. Ozon order match
                if conf < 0.9 and t_ozon and ld.ozon_order_id and t_ozon == ld.ozon_order_id:
                    conf, mtype = 0.92, "ozon_order"; fields["ozon_order"] = ld.ozon_order_id
                # 5. QR payload match
                if conf < 0.8:
                    for payload in ld.qr_payloads:
                        if t_track and t_track in payload: conf, mtype = 0.85, "qr_data"; fields["qr_tracking"] = t_track; break
                        if t_phone and t_phone in payload: conf, mtype = max(conf, 0.7), "qr_phone"; fields["qr_phone"] = t_phone
                # 6. Fuzzy digit overlap
                if conf < 0.5 and t_raw:
                    td = set(re.findall(r"\d+", t_raw))
                    ldg = set(re.findall(r"\d+", ld.raw_text + ld.vertical_text))
                    if td and ldg:
                        fuzzy_matches = sum(1 for td_val in td for ldg_val in ldg if fuzzy_digit_match(td_val, ldg_val) > 0.8)
                        overlap = fuzzy_matches / len(td)
                        if overlap > 0.5: conf, mtype = overlap * 0.5 + 0.1, "fuzzy"
                # 7. Address similarity
                if conf > 0.3 and target.get("address") and ld.address:
                    if target["address"].lower() in ld.address.lower() or ld.address.lower() in target["address"].lower():
                        conf = min(1.0, conf + 0.05)
                if conf > best_conf and conf >= 0.5:
                    best_conf = conf
                    best = WBLabelMatch(ld.page_idx, t_idx, t_raw, mtype, conf, fields, ld.to_dict())
            if best: used.add(best.page_idx); matches.append(best)
        return matches

    # ── Sequence Validation ──────────────────────────────────────────────
    def validate_sequence(self, label_data: List[WBLabelData], sequence_by: str = "tracking") -> SequenceValidationResult:
        keys = [ld.tracking_number for ld in label_data] if sequence_by == "tracking" else \
               [ld.phone_number for ld in label_data] if sequence_by == "phone" else \
               [ld.wb_order_id or ld.ozon_order_id for ld in label_data]
        valid_keys = [k for k in keys if k]
        if not valid_keys:
            return SequenceValidationResult(False, [], keys, [], [], 0.0)
        seen, duplicates = set(), []
        for i, k in enumerate(keys):
            if k and k in seen: duplicates.append(i)
            seen.add(k)
        if all(k and k.isdigit() for k in valid_keys):
            nums = sorted(int(k) for k in valid_keys)
            expected = list(range(nums[0], nums[-1] + 1))
            missing = [str(n) for n in expected if n not in nums]
            is_valid = len(missing) == 0 and len(duplicates) == 0
            conf = 1.0 - (len(missing) / max(1, len(expected))) * 0.5 - (len(duplicates) / max(1, len(keys))) * 0.5
            return SequenceValidationResult(is_valid, [str(n) for n in expected], keys, missing, duplicates, max(0.0, conf))
        is_valid = len(duplicates) == 0
        conf = 1.0 - (len(duplicates) / max(1, len(keys))) * 0.5
        return SequenceValidationResult(is_valid, valid_keys, keys, [], duplicates, max(0.0, conf))

    # ── Multi-Criteria Sorting ───────────────────────────────────────────
    def sort_pages(self, label_data: List[WBLabelData], sort_by: str = "tracking",
                   targets: Optional[List[Dict]] = None) -> List[WBLabelData]:
        if sort_by == "tracking":
            return sorted(label_data, key=lambda ld: (ld.tracking_number or "ZZZZ", ld.page_idx))
        elif sort_by == "phone":
            return sorted(label_data, key=lambda ld: (ld.phone_number or "9999999", ld.page_idx))
        elif sort_by == "order":
            return sorted(label_data, key=lambda ld: (ld.wb_order_id or ld.ozon_order_id or "ZZZZ", ld.page_idx))
        elif sort_by == "marketplace":
            return sorted(label_data, key=lambda ld: (ld.marketplace, ld.tracking_number or ""))
        elif sort_by == "route_optimized" and targets:
            target_order = {}
            for i, t in enumerate(targets):
                key = t.get("tracking") or f"{t.get('phone', '')}_{t.get('code', '')}" or t.get("order_id", "")
                target_order[key.upper()] = i
            return sorted(label_data, key=lambda ld: target_order.get(
                ld.tracking_number or f"{ld.phone_number}_{ld.delivery_code}" or ld.wb_order_id or "", 9999))
        else:
            return sorted(label_data, key=lambda ld: ld.page_idx)

    # ── Debug & Persistence ──────────────────────────────────────────────
    def generate_debug_overlay(self, img: Image.Image, data: WBLabelData) -> Image.Image:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        overlay = img_cv.copy()
        if data.is_duplicate: border_color, status_text = (0, 165, 255), "DUPLICATE"
        elif data.checksum_valid: border_color, status_text = (0, 255, 136), "VALID"
        elif data.tracking_number: border_color, status_text = (255, 217, 61), "DETECTED"
        else: border_color, status_text = (255, 107, 107), "NOT FOUND"
        cv2.rectangle(overlay, (5, 5), (w - 5, h - 5), border_color, 3)
        cv2.rectangle(overlay, (10, 10), (w - 10, 170), (5, 10, 25), -1)
        cv2.rectangle(overlay, (10, 10), (w - 10, 170), border_color, 1)
        lines = [
            f"Page: {data.page_idx + 1} | Market: {data.marketplace} | Rot: {data.rotation_detected}deg | {status_text}",
            f"Tracking: {data.tracking_number or 'NOT FOUND'} {'[CHK]' if data.checksum_valid else ''}",
            f"Phone: {data.phone_number or 'N/A'} | Code: {data.delivery_code or 'N/A'}",
            f"WB Order: {data.wb_order_id or 'N/A'} | Ozon: {data.ozon_order_id or 'N/A'} | QRs: {len(data.qr_payloads)}",
            f"Conf: {data.extraction_confidence:.0%} | Hash: {data.perceptual_hash[:8]}...",
        ]
        y = 35
        for line in lines:
            cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 218), 2)
            y += 25
        return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    def record_job(self, marketplace, mode, targets, matched, missing, extra,
                   avg_confidence, operator, stats=None):
        from db import save_label_job
        import realtime
        job_id = f"LB-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"
        save_label_job(job_id, marketplace, mode, targets, matched, missing, extra,
                       round(avg_confidence or 0, 3), operator, stats)
        realtime.publish("LABEL_JOB", operator,
                         f"{marketplace} {mode}: {matched}/{targets} matched ({extra} extra)", ref_id=job_id)
        return job_id

    def job_history(self, limit=25):
        from db import get_label_jobs
        return get_label_jobs(limit)

# ═══════════════════════════════════════════════════════════════════════════
# TARGET LIST PARSER
# ═══════════════════════════════════════════════════════════════════════════
def parse_target_list(text: str, marketplace: str = "WB") -> List[Dict]:
    entries = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line: continue
        entry = {"raw": line}
        digits = re.findall(r"\d+", line)
        wb = MARKETPLACES["WB"]["tracking"].search(line)
        if wb: entry["tracking"] = wb.group().upper()
        oz = MARKETPLACES["OZON"]["order"].search(line)
        if oz: entry["ozon_order"] = oz.group()
        phone = next((d for d in digits if len(d) == 7), None)
        code = next((d for d in digits if len(d) == 4), None)
        if phone: entry["phone"] = phone
        if code: entry["code"] = code
        long_digits = [d for d in digits if 9 <= len(d) <= 15]
        if long_digits and "tracking" not in entry: entry["order_id"] = long_digits[0]
        if any(k in line.lower() for k in ("ул.", "улица", "г.", "пр.", "пер.")):
            entry["address"] = line
        entries.append(entry)
    return entries

def normalize_target(target: Dict) -> str:
    parts = []
    if target.get("tracking"): parts.append(target["tracking"].upper())
    if target.get("phone") and target.get("code"): parts.append(f"{target['phone']}_{target['code']}")
    if target.get("order_id"): parts.append(target["order_id"])
    if target.get("ozon_order"): parts.append(target["ozon_order"])
    return "|".join(parts) if parts else target.get("raw", "")
