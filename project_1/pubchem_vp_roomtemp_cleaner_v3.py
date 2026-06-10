#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pubchem_vp_roomtemp_cleaner_v3.py

Retrieve and clean vapor pressure values from PubChem.

Version v3 adds CID fallback: if the primary CAS/name-resolved CID has no accepted
room-temperature VP, the script can try the existing CID from the input file and
name/IUPAC-name resolved alternatives.

Main goals:
- Get vapor pressure in mmHg.
- Accept common PubChem units: mmHg / mm Hg / torr / Pa / kPa / bar / mbar / atm / psi.
- Convert all values to mmHg.
- Prefer values at room / standard temperature: default 15-30 C.
- Reject values with no explicit temperature by default, because VP is highly temperature-dependent.
- Parse common patterns like:
    "0.094 mm Hg at 25 °C"
    "0.1 mmHg at 68 °F"
    "Vapor pressure, Pa at 20 °C: 13.2"
    "1 mmHg at 79.2 °F ; 5 mmHg at 122.2 °F"
    "13.2 Pa at 20 °C"
    "0.09 [mmHg]"

Run:
    python pubchem_vp_roomtemp_cleaner_v3.py input.csv output_vp_roomtemp.csv

Test one CID:
    python pubchem_vp_roomtemp_cleaner_v3.py --test-cid 244
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
DEFAULT_TIMEOUT = 25
DEFAULT_SLEEP = 0.20


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    t = str(text)
    t = t.replace("\u00b0", " °")
    t = t.replace("º", " °")
    t = t.replace("℃", " °C")
    t = t.replace("℉", " °F")
    t = t.replace("˚", " °")
    t = t.replace("@", " @ ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    norm_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def row_key(row: pd.Series, idx: int, cas_col: Optional[str], name_col: Optional[str], cid_col: Optional[str]) -> str:
    cas = str(row.get(cas_col, "")).strip() if cas_col else ""
    name = str(row.get(name_col, "")).strip() if name_col else ""
    cid = str(row.get(cid_col, "")).strip() if cid_col else ""
    return f"{idx}|CAS={cas}|CID={cid}|NAME={name}"


def f_to_c(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def k_to_c(k: float) -> float:
    return k - 273.15


def temp_to_c(value: float, unit: str) -> float:
    u = unit.lower().strip()
    if "f" in u:
        return f_to_c(value)
    if u == "k" or "kelvin" in u:
        return k_to_c(value)
    return value


def vp_to_mmhg(value: float, unit: str) -> float:
    u = re.sub(r"[\[\]\s]+", "", unit.lower())
    if u in {"mmhg", "mmofhg", "torr"}:
        return value
    if u == "pa":
        return value / 133.322368
    if u == "kpa":
        return value * 1000.0 / 133.322368
    if u == "atm":
        return value * 760.0
    if u == "bar":
        return value * 750.061683
    if u == "mbar":
        return value * 0.750061683
    if u == "psi":
        return value * 51.7149326
    raise ValueError(f"Unsupported vapor pressure unit: {unit}")


TEMP_RE = re.compile(
    r"(?P<t>-?\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?)?\s*(?P<tunit>C|F|K|c|f|k|kelvin)\b",
    re.IGNORECASE,
)

# Value before unit, common case: 0.094 mm Hg at 25 C
VP_VALUE_UNIT_RE = re.compile(
    r"(?P<v>\d+(?:\.\d+)?(?:\s*(?:x|×)\s*10\s*[+\-−]?\s*\d+)?)\s*\[?\s*(?P<unit>mm\s*hg|mmhg|mm\s*of\s*hg|torr|pa|kpa|atm|bar|mbar|psi)\s*\]?\b",
    re.IGNORECASE,
)

# Unit before value, common case: Vapor pressure, Pa at 20 C: 13.2
VP_UNIT_VALUE_RE = re.compile(
    r"(?P<unit>mm\s*hg|mmhg|mm\s*of\s*hg|torr|pa|kpa|atm|bar|mbar|psi)\s*(?:at\s*-?\d+(?:\.\d+)?\s*°?\s*(?:C|F|K))?\s*[:=]\s*(?P<v>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

BAD_CONTEXT_RE = re.compile(
    r"\b(?:boiling\s*point|melting\s*point|flash\s*point|density|autoignition|ignition|decomposition)\b",
    re.IGNORECASE,
)

ROOM_TEMP_WORD_RE = re.compile(r"\b(?:room\s*temperature|ambient\s*temperature|rt\b|25\s*°?\s*c|20\s*°?\s*c)\b", re.IGNORECASE)


def parse_numeric_maybe_scientific(s: str) -> float:
    t = s.strip().lower().replace("×", "x").replace("−", "-")
    m = re.match(r"^(\d+(?:\.\d+)?)\s*x\s*10\s*([+\-]?\d+)$", t)
    if m:
        return float(m.group(1)) * (10 ** int(m.group(2)))
    return float(t)


def nearby_temperature_c(text: str, span: Tuple[int, int], window: int = 80) -> Tuple[Optional[float], str]:
    """Find the closest explicit temperature to a VP value.

    PubChem often stores several pairs in one string, e.g.
    "1 mmHg at 79.2 F ; 5 mmHg at 122.2 F".
    Choosing the first temperature in the window is wrong for such strings,
    so we choose the temperature whose regex span is closest to the VP span.
    """
    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    nearby = text[start:end]
    value_center = (span[0] + span[1]) / 2.0

    best = None
    best_dist = None
    for m in TEMP_RE.finditer(nearby):
        absolute_span = (start + m.start(), start + m.end())
        temp_center = (absolute_span[0] + absolute_span[1]) / 2.0
        dist = abs(temp_center - value_center)
        tc = temp_to_c(float(m.group("t")), m.group("tunit"))
        if best is None or dist < best_dist:
            best = tc
            best_dist = dist

    if best is not None:
        return best, "explicit_temperature_near_value"
    if ROOM_TEMP_WORD_RE.search(nearby):
        return 25.0, "room_temperature_text_assume_25C"
    return None, "no_temperature_near_value"


def candidate_score(raw: str, temp_c: Optional[float], temp_reason: str, temp_min: float, temp_max: float) -> int:
    t = normalize_text(raw).lower()
    score = 0
    if "vapor pressure" in t or "vapour pressure" in t:
        score += 30
    if temp_c is not None:
        score += 25
        # prefer 25 C, then 20 C, then other room temperature values
        score += max(0, int(20 - abs(temp_c - 25)))
    if temp_reason.startswith("explicit"):
        score += 10
    if temp_c is not None and temp_min <= temp_c <= temp_max:
        score += 20
    return score


def plausible_vp_mmhg(vp: float) -> bool:
    return 0.0 < vp < 76000.0



def split_pubchem_vp_text(raw: str) -> List[Tuple[str, int]]:
    """Split multi-pair PubChem strings into clauses while keeping offsets.

    Returns (clause_text, offset_in_raw). Offsets allow temperature-distance
    logic to still work if needed.
    """
    parts = []
    pos = 0
    for piece in re.split(r"\s*;\s*", raw):
        idx = raw.find(piece, pos)
        if idx < 0:
            idx = pos
        if piece.strip():
            parts.append((piece.strip(), idx))
        pos = idx + len(piece) + 1
    return parts or [(raw, 0)]

def parse_vp_candidates(text: str, temp_min_c: float = 15.0, temp_max_c: float = 30.0, accept_no_temp: bool = False) -> List[Dict[str, Any]]:
    raw = normalize_text(text)
    if not raw:
        return []
    if BAD_CONTEXT_RE.search(raw) and not re.search(r"vapou?r\s*pressure", raw, re.IGNORECASE):
        return []

    candidates: List[Dict[str, Any]] = []

    # Important: split strings containing several VP/T pairs.
    # Example: "1 mmHg at 79.2 F ; 5 mmHg at 122.2 F".
    # Without this, the first room-temperature value can be assigned to every VP value.
    for clause, _offset in split_pubchem_vp_text(raw):
        matches = list(VP_VALUE_UNIT_RE.finditer(clause)) + list(VP_UNIT_VALUE_RE.finditer(clause))

        for m in matches:
            try:
                value = parse_numeric_maybe_scientific(m.group("v"))
                unit = m.group("unit")
                vp_mmhg = vp_to_mmhg(value, unit)
            except Exception:
                continue

            if not plausible_vp_mmhg(vp_mmhg):
                continue

            temp_c, temp_reason = nearby_temperature_c(clause, m.span())

            if temp_c is None and not accept_no_temp:
                continue
            if temp_c is not None and not (temp_min_c <= temp_c <= temp_max_c):
                continue

            is_estimated = bool(re.search(r"\b(?:estimated|est\.?|extrapolated|calculated|predicted)\b", clause, re.IGNORECASE))
            score = candidate_score(clause, temp_c, temp_reason, temp_min_c, temp_max_c)
            if is_estimated:
                score -= 8

            candidates.append({
                "vp_mmhg_pubchem": round(vp_mmhg, 8),
                "vp_value_original": value,
                "vp_unit_original": re.sub(r"\s+", " ", unit.strip()),
                "vp_temperature_c": round(temp_c, 3) if temp_c is not None else None,
                "vp_temperature_reason": temp_reason,
                "vp_is_estimated_or_extrapolated": is_estimated,
                "vp_raw_pubchem": clause,
                "vp_parse_score": score,
            })

    # remove duplicates from same string
    unique = []
    seen = set()
    for c in candidates:
        key = (c["vp_mmhg_pubchem"], c["vp_temperature_c"], c["vp_unit_original"], c["vp_raw_pubchem"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

def choose_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda c: (
            c.get("vp_parse_score", 0),
            -abs((c.get("vp_temperature_c") or 25.0) - 25.0),
        ),
        reverse=True,
    )[0]


def safe_get_json(url: str, timeout: int = DEFAULT_TIMEOUT, max_retries: int = 3, sleep: float = DEFAULT_SLEEP) -> Optional[Dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in {400, 404}:
                return None
        except Exception:
            pass
        time.sleep(sleep * (attempt + 1) * 2)
    return None


def resolve_cid(identifier: str, namespace: str = "name", sleep: float = DEFAULT_SLEEP) -> Optional[int]:
    identifier = str(identifier or "").strip()
    if not identifier or identifier.lower() in {"nan", "none"}:
        return None
    from urllib.parse import quote
    url = f"{PUBCHEM_PUG_REST}/compound/{namespace}/{quote(identifier)}/cids/JSON"
    data = safe_get_json(url, sleep=sleep)
    time.sleep(sleep)
    try:
        cids = data["IdentifierList"]["CID"]
        if cids:
            return int(cids[0])
    except Exception:
        return None
    return None


def resolve_cid_from_row(row: pd.Series, cas_col: Optional[str], name_col: Optional[str], cid_col: Optional[str], sleep: float) -> Tuple[Optional[int], str]:
    cas = str(row.get(cas_col, "")).strip() if cas_col else ""
    name = str(row.get(name_col, "")).strip() if name_col else ""
    existing_cid = str(row.get(cid_col, "")).strip() if cid_col else ""
    if cas and cas.lower() not in {"nan", "none"}:
        cid = resolve_cid(cas, namespace="name", sleep=sleep)
        if cid:
            return cid, "resolved_from_cas"
    if name and name.lower() not in {"nan", "none"}:
        cid = resolve_cid(name, namespace="name", sleep=sleep)
        if cid:
            return cid, "resolved_from_name"
    if existing_cid and existing_cid.lower() not in {"nan", "none"}:
        try:
            return int(float(existing_cid)), "existing_cid_fallback"
        except Exception:
            pass
    return None, "not_resolved"


def clean_cell_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_existing_cid(row: pd.Series, cid_col: Optional[str]) -> Optional[int]:
    if not cid_col:
        return None
    existing_cid = clean_cell_text(row.get(cid_col, ""))
    if not existing_cid:
        return None
    try:
        return int(float(existing_cid))
    except Exception:
        return None


def build_cid_attempts_for_row(
    row: pd.Series,
    cas_col: Optional[str],
    name_col: Optional[str],
    cid_col: Optional[str],
    iupac_col: Optional[str],
    sleep: float,
    use_fallback: bool,
) -> List[Tuple[int, str]]:
    """Build ordered CID attempts.

    Priority:
    1. Primary resolver, which prefers CAS. This preserves strict identity.
    2. Existing CID from the input table. This is useful when CAS resolves to a
       stereoisomer/salt record without a VP section while the table CID is the
       general parent compound.
    3. Name and IUPAC-name resolved CIDs, useful as additional PubChem aliases.
    """
    attempts: List[Tuple[int, str]] = []

    def add(cid: Optional[int], method: str) -> None:
        if cid is None:
            return
        if cid <= 0:
            return
        if cid not in [c for c, _ in attempts]:
            attempts.append((cid, method))

    primary_cid, primary_method = resolve_cid_from_row(row, cas_col, name_col, cid_col, sleep=sleep)
    add(primary_cid, primary_method)

    if not use_fallback:
        return attempts

    existing_cid = parse_existing_cid(row, cid_col)
    add(existing_cid, "fallback_existing_input_cid")

    name = clean_cell_text(row.get(name_col, "")) if name_col else ""
    if name:
        add(resolve_cid(name, namespace="name", sleep=sleep), "fallback_resolved_from_name")

    iupac = clean_cell_text(row.get(iupac_col, "")) if iupac_col else ""
    if iupac and iupac.lower() != name.lower():
        add(resolve_cid(iupac, namespace="name", sleep=sleep), "fallback_resolved_from_iupac")

    return attempts


def get_compound_view(cid: int, cache_dir: Optional[Path], sleep: float = DEFAULT_SLEEP) -> Optional[Dict[str, Any]]:
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"cid_{cid}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    url = f"{PUBCHEM_PUG_VIEW}/data/compound/{cid}/JSON"
    data = safe_get_json(url, sleep=sleep)
    time.sleep(sleep)
    if data and cache_dir:
        try:
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
    return data


def extract_strings_from_value(value: Any) -> List[str]:
    strings: List[str] = []
    if isinstance(value, dict):
        if "StringWithMarkup" in value:
            swm = value.get("StringWithMarkup") or []
            if isinstance(swm, list):
                for item in swm:
                    if isinstance(item, dict) and item.get("String"):
                        strings.append(str(item["String"]))
                    elif isinstance(item, str):
                        strings.append(item)
        for key in ["String", "Number"]:
            if key in value and value[key] is not None:
                strings.append(str(value[key]))
        for v in value.values():
            strings.extend(extract_strings_from_value(v))
    elif isinstance(value, list):
        for item in value:
            strings.extend(extract_strings_from_value(item))
    elif isinstance(value, str):
        strings.append(value)
    return strings


def walk_sections(section: Dict[str, Any], path: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
    heading = section.get("TOCHeading", "")
    new_path = f"{path} > {heading}" if heading else path
    yield new_path, section
    for sub in section.get("Section", []) or []:
        yield from walk_sections(sub, new_path)


def extract_vapor_pressure_texts(compound_view: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    try:
        root_sections = compound_view["Record"]["Section"]
    except Exception:
        return texts
    for root in root_sections:
        for path, sec in walk_sections(root):
            path_l = path.lower()
            heading_l = str(sec.get("TOCHeading", "")).lower()
            is_vp_section = ("vapor pressure" in path_l) or ("vapour pressure" in path_l) or (heading_l.strip() in {"vapor pressure", "vapour pressure"})
            if not is_vp_section:
                continue
            for info in sec.get("Information", []) or []:
                value = info.get("Value")
                for s in extract_strings_from_value(value):
                    ns = normalize_text(s)
                    if ns:
                        texts.append(ns)
    seen = set()
    unique = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def get_pubchem_title(compound_view: Dict[str, Any]) -> Optional[str]:
    try:
        return compound_view["Record"].get("RecordTitle")
    except Exception:
        return None


def get_best_vp_for_cid(cid: int, cache_dir: Optional[Path], sleep: float, temp_min_c: float, temp_max_c: float, accept_no_temp: bool) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[str], Optional[str]]:
    view = get_compound_view(cid, cache_dir=cache_dir, sleep=sleep)
    if not view:
        return None, [], [], None
    title = get_pubchem_title(view)
    vp_texts = extract_vapor_pressure_texts(view)
    all_candidates: List[Dict[str, Any]] = []
    for text in vp_texts:
        all_candidates.extend(parse_vp_candidates(text, temp_min_c=temp_min_c, temp_max_c=temp_max_c, accept_no_temp=accept_no_temp))
    best = choose_best_candidate(all_candidates)
    return best, all_candidates, vp_texts, title


OUTPUT_COLUMNS = [
    "vp_mmhg_pubchem_clean",
    "vp_value_original_clean",
    "vp_unit_original_clean",
    "vp_temperature_c",
    "vp_temperature_reason",
    "vp_is_estimated_or_extrapolated",
    "vp_raw_pubchem_clean",
    "vp_parse_score",
    "resolved_cid_clean",
    "cid_resolution_method",
    "vp_primary_cid",
    "vp_primary_method",
    "vp_attempted_cids",
    "vp_fallback_used",
    "vp_fallback_reason",
    "pubchem_title_clean",
    "vp_status",
    "vp_error",
]


def read_input_table(input_file: Path) -> pd.DataFrame:
    suffix = input_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_file)
    return pd.read_csv(input_file)


def process_file(input_file: Path, output_file: Path, autosave_every: int, cache_dir: Optional[Path], sleep: float, no_resume: bool, temp_min_c: float, temp_max_c: float, accept_no_temp: bool, use_fallback: bool) -> None:
    df = read_input_table(input_file)
    cas_col = find_column(df, ["CAS", "CAS Number", "CAS No", "cas_number", "cas_no"])
    name_col = find_column(df, ["Name of Material", "Name", "Material", "material_name", "Chemical Name", "Synonym"])
    cid_col = find_column(df, ["CID", "PubChem CID", "cid", "pubchem_cid"])
    iupac_col = find_column(df, ["IUPACName", "IUPAC Name", "iupac_name", "IUPAC"])

    print("Input file:", input_file)
    print("Output file:", output_file)
    print("Rows:", len(df))
    print("Detected columns:")
    print("  CAS column :", cas_col)
    print("  Name column:", name_col)
    print("  CID column :", cid_col)
    print("  IUPAC column:", iupac_col)
    print(f"Accepted temperature window: {temp_min_c:g}-{temp_max_c:g} C")
    print("Accept no temperature:", accept_no_temp)
    print("CID fallback enabled:", use_fallback)
    print()

    processed_by_key: Dict[str, Dict[str, Any]] = {}
    if output_file.exists() and not no_resume:
        try:
            prev = pd.read_csv(output_file)
            if "_processing_key" in prev.columns:
                for _, r in prev.iterrows():
                    k = str(r.get("_processing_key", ""))
                    if k:
                        processed_by_key[k] = r.to_dict()
                print(f"Resume enabled: loaded {len(processed_by_key)} already processed rows.")
        except Exception as e:
            print(f"Could not resume from existing output: {e}")

    results: List[Dict[str, Any]] = []
    found_count = 0
    error_count = 0
    resumed_count = 0

    pbar = tqdm(df.iterrows(), total=len(df), desc="Retrieving PubChem VP", unit="mol")
    for idx, row in pbar:
        key = row_key(row, idx, cas_col, name_col, cid_col)
        if key in processed_by_key and not no_resume:
            existing = processed_by_key[key]
            results.append(existing)
            resumed_count += 1
            if pd.notna(existing.get("vp_mmhg_pubchem_clean")):
                found_count += 1
            pbar.set_postfix({"found VP": found_count, "errors": error_count, "resumed": resumed_count})
            continue

        out = row.to_dict()
        out["_processing_key"] = key
        for col in OUTPUT_COLUMNS:
            out[col] = None

        try:
            attempts = build_cid_attempts_for_row(
                row, cas_col, name_col, cid_col, iupac_col, sleep=sleep, use_fallback=use_fallback
            )
            out["vp_attempted_cids"] = "; ".join([f"{cid}:{method}" for cid, method in attempts])

            if attempts:
                out["vp_primary_cid"] = attempts[0][0]
                out["vp_primary_method"] = attempts[0][1]
            else:
                out["vp_status"] = "no_cid"
                results.append(out)
                pbar.set_postfix({"found VP": found_count, "errors": error_count, "resumed": resumed_count})
                continue

            best = None
            title = None
            chosen_cid = None
            chosen_method = None
            best_no_candidate_texts: List[str] = []
            saw_vp_section = False

            for attempt_index, (cid, method) in enumerate(attempts):
                candidate_best, candidates, vp_texts, candidate_title = get_best_vp_for_cid(
                    cid, cache_dir, sleep, temp_min_c, temp_max_c, accept_no_temp
                )
                if candidate_title and title is None:
                    title = candidate_title
                if vp_texts:
                    saw_vp_section = True
                    if not best_no_candidate_texts:
                        best_no_candidate_texts = vp_texts

                if candidate_best:
                    best = candidate_best
                    title = candidate_title
                    chosen_cid = cid
                    chosen_method = method
                    out["vp_fallback_used"] = attempt_index > 0
                    out["vp_fallback_reason"] = "primary_cid_had_no_accepted_roomtemp_vp" if attempt_index > 0 else "not_needed"
                    break

            out["resolved_cid_clean"] = chosen_cid if chosen_cid is not None else attempts[0][0]
            out["cid_resolution_method"] = chosen_method if chosen_method is not None else attempts[0][1]
            out["pubchem_title_clean"] = title

            if best:
                out["vp_mmhg_pubchem_clean"] = best.get("vp_mmhg_pubchem")
                out["vp_value_original_clean"] = best.get("vp_value_original")
                out["vp_unit_original_clean"] = best.get("vp_unit_original")
                out["vp_temperature_c"] = best.get("vp_temperature_c")
                out["vp_temperature_reason"] = best.get("vp_temperature_reason")
                out["vp_is_estimated_or_extrapolated"] = best.get("vp_is_estimated_or_extrapolated")
                out["vp_raw_pubchem_clean"] = best.get("vp_raw_pubchem")
                out["vp_parse_score"] = best.get("vp_parse_score")
                out["vp_status"] = "found"
                found_count += 1
            else:
                out["vp_fallback_used"] = False
                out["vp_fallback_reason"] = "tried_all_cids_no_accepted_roomtemp_vp" if use_fallback else "fallback_disabled"
                if saw_vp_section:
                    out["vp_status"] = "vp_section_found_but_no_accepted_roomtemp_value"
                    out["vp_raw_pubchem_clean"] = " | ".join(best_no_candidate_texts[:5])
                else:
                    out["vp_status"] = "no_vapor_pressure_section"

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial output...")
            results.append(out)
            pd.DataFrame(results).to_csv(output_file, index=False)
            raise
        except Exception as e:
            out["vp_status"] = "error"
            out["vp_error"] = repr(e)
            error_count += 1

        results.append(out)
        pbar.set_postfix({"found VP": found_count, "errors": error_count, "resumed": resumed_count})
        if autosave_every and len(results) % autosave_every == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)

    final = pd.DataFrame(results)
    final.to_csv(output_file, index=False)
    print("\nDONE")
    print("Rows processed:", len(final))
    print("Found VP:", final["vp_mmhg_pubchem_clean"].notna().sum())
    print("Saved to:", output_file)


def test_cid(cid: int, cache_dir: Optional[Path], sleep: float, temp_min_c: float, temp_max_c: float, accept_no_temp: bool) -> None:
    print(f"Testing CID {cid}")
    best, candidates, vp_texts, title = get_best_vp_for_cid(cid, cache_dir, sleep, temp_min_c, temp_max_c, accept_no_temp)
    print("PubChem title:", title)
    print("\nVapor pressure texts found:")
    if not vp_texts:
        print("  No VP texts found.")
    for i, t in enumerate(vp_texts, start=1):
        print(f"  {i}. {t}")
        parsed = parse_vp_candidates(t, temp_min_c=temp_min_c, temp_max_c=temp_max_c, accept_no_temp=accept_no_temp)
        if parsed:
            for p in parsed:
                print(f"     ACCEPT: {p['vp_mmhg_pubchem']} mmHg at {p['vp_temperature_c']} C; unit={p['vp_unit_original']}; score={p['vp_parse_score']}")
        else:
            print("     REJECT: no accepted room-temperature candidate")
    print("\nBEST:")
    print(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve PubChem vapor pressure near room temperature and save mmHg values.")
    parser.add_argument("input_file", nargs="?", help="Input CSV file")
    parser.add_argument("output_file", nargs="?", help="Output CSV file")
    parser.add_argument("--autosave-every", type=int, default=25)
    parser.add_argument("--cache-dir", default=".pubchem_cache_vp")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--test-cid", type=int)
    parser.add_argument("--temp-min-c", type=float, default=15.0)
    parser.add_argument("--temp-max-c", type=float, default=30.0)
    parser.add_argument("--accept-no-temp", action="store_true", help="Accept VP values without explicit temperature. Not recommended for final data.")
    parser.add_argument("--no-fallback", action="store_true", help="Disable CID fallback attempts after the primary CAS/name CID.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if args.test_cid:
        test_cid(args.test_cid, cache_dir, args.sleep, args.temp_min_c, args.temp_max_c, args.accept_no_temp)
        return
    if not args.input_file or not args.output_file:
        parser.error("input_file and output_file are required unless --test-cid is used")
    process_file(
        Path(args.input_file),
        Path(args.output_file),
        args.autosave_every,
        cache_dir,
        args.sleep,
        args.no_resume,
        args.temp_min_c,
        args.temp_max_c,
        args.accept_no_temp,
        not args.no_fallback,
    )


if __name__ == "__main__":
    main()
