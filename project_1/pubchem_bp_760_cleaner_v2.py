#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pubchem_bp_760_cleaner_v2.py

Retrieve and clean boiling point values from PubChem.

Main goal:
- Get boiling point in Celsius.
- Prefer values at standard pressure:
    760 mmHg / 760 mm Hg / 760 torr / 1 atm / 101.3 kPa / 101.325 kPa.
- Accept values with no pressure mentioned, assuming they are normal boiling points.
- Reject reduced-pressure values such as:
    @ 10 mmHg, at 5 torr, 2 kPa, 0.1 mmHg, etc.
- Correctly parse PubChem patterns like:
    "194.00 to 197.00 °C. @ 760.00 mm Hg"

Run:
    python pubchem_bp_760_cleaner_v2.py data\moodify_inventory_pubchem_bp_vp.csv data\output_bp760.csv

Test linalool:
    python pubchem_bp_760_cleaner_v2.py --test-cid 6549
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
    columns = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in columns}

    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    norm_map = {norm(c): c for c in columns}
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


def convert_to_c(value: float, unit: str) -> float:
    unit_l = unit.lower()
    if "f" in unit_l:
        return f_to_c(value)
    if unit_l.strip() == "k" or "kelvin" in unit_l:
        return k_to_c(value)
    return value


STANDARD_PRESSURE_PATTERNS = [
    r"\b760(?:\.0+)?\s*mm\s*hg\b",
    r"\b760(?:\.0+)?\s*mmhg\b",
    r"\b760(?:\.0+)?\s*torr\b",
    r"\b1(?:\.0+)?\s*atm\b",
    r"\bone\s+atmosphere\b",
    r"\b101\.3(?:0+)?\s*kpa\b",
    r"\b101\.325(?:0+)?\s*kpa\b",
    r"\bstandard\s+pressure\b",
    r"\bnormal\s+pressure\b",
]

PRESSURE_VALUE_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm\s*hg|mmhg|torr|atm|kpa|pa|bar|mbar)\b",
    re.IGNORECASE,
)


def has_standard_pressure(text: str) -> bool:
    t = normalize_text(text).lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in STANDARD_PRESSURE_PATTERNS)


def pressure_value_is_standard(value: float, unit: str) -> bool:
    unit_l = re.sub(r"\s+", "", unit.lower())

    if unit_l in {"mmhg", "torr"}:
        return 740.0 <= value <= 780.0

    if unit_l == "atm":
        return 0.97 <= value <= 1.03

    if unit_l == "kpa":
        return 98.0 <= value <= 104.0

    if unit_l == "pa":
        return 98000.0 <= value <= 104000.0

    if unit_l == "bar":
        return 0.98 <= value <= 1.04

    if unit_l == "mbar":
        return 980.0 <= value <= 1040.0

    return False


def pressure_status(text: str) -> Tuple[bool, str]:
    """
    Accepted:
    - explicit standard pressure;
    - pressure equivalent to standard pressure;
    - no pressure mentioned.

    Rejected:
    - pressure is mentioned but not standard.
    """
    t = normalize_text(text)

    if has_standard_pressure(t):
        return True, "explicit_standard_pressure"

    matches = list(PRESSURE_VALUE_PATTERN.finditer(t))
    if not matches:
        return True, "no_pressure_mentioned_assume_normal_bp"

    for m in matches:
        value = float(m.group("value"))
        unit = m.group("unit")
        if pressure_value_is_standard(value, unit):
            return True, f"standard_pressure_numeric_{value}_{unit}"

    values = [f"{m.group('value')} {m.group('unit')}" for m in matches]
    return False, "nonstandard_pressure_" + "; ".join(values)


BP_RANGE_RE = re.compile(
    r"(?P<v1>-?\d+(?:\.\d+)?)\s*(?:to|[-–—])\s*(?P<v2>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>C|F|K|c|f|k|degrees?\s*C|degrees?\s*F|kelvin)\b",
    re.IGNORECASE,
)

BP_SINGLE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<v>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>C|F|K|c|f|k|degrees?\s*C|degrees?\s*F|kelvin)\b",
    re.IGNORECASE,
)

BAD_CONTEXT_RE = re.compile(
    r"\b(?:flash\s*point|melting\s*point|mp\b|density|vapor\s*pressure|vapou?r\s*pressure|decomposition|decomposes|ignition|autoignition)\b",
    re.IGNORECASE,
)


def clean_unit(unit: str) -> str:
    u = unit.strip().lower()
    if "f" in u:
        return "F"
    if u == "k" or "kelvin" in u:
        return "K"
    return "C"


def plausible_bp_c(value_c: float) -> bool:
    return -200.0 <= value_c <= 700.0


def candidate_score(raw: str, pressure_reason: str, is_range: bool) -> int:
    t = normalize_text(raw).lower()
    score = 0

    if "boiling" in t or "bp" in t:
        score += 30
    if "760" in t or "1 atm" in t or "101.3" in t or "101.325" in t:
        score += 25
    if pressure_reason.startswith("explicit") or pressure_reason.startswith("standard"):
        score += 20
    if is_range:
        score += 5
    if pressure_reason.startswith("no_pressure"):
        score += 5

    return score


def parse_bp_candidates(text: str) -> List[Dict[str, Any]]:
    raw_text = normalize_text(text)
    if not raw_text:
        return []

    # Avoid extracting flash point / vapor pressure etc. from incorrectly mixed strings.
    if BAD_CONTEXT_RE.search(raw_text) and "boiling" not in raw_text.lower() and " bp" not in raw_text.lower():
        return []

    accepted_pressure, pressure_reason = pressure_status(raw_text)
    if not accepted_pressure:
        return []

    candidates: List[Dict[str, Any]] = []
    consumed_spans = []

    for m in BP_RANGE_RE.finditer(raw_text):
        v1 = float(m.group("v1"))
        v2 = float(m.group("v2"))
        unit = clean_unit(m.group("unit"))

        c1 = convert_to_c(v1, unit)
        c2 = convert_to_c(v2, unit)

        bp_min = min(c1, c2)
        bp_max = max(c1, c2)
        bp_mid = (bp_min + bp_max) / 2.0

        if plausible_bp_c(bp_mid):
            candidates.append({
                "bp_c_pubchem": round(bp_mid, 3),
                "bp_c_min_pubchem": round(bp_min, 3),
                "bp_c_max_pubchem": round(bp_max, 3),
                "bp_unit_original": unit,
                "bp_raw_pubchem": raw_text,
                "bp_pressure_reason": pressure_reason,
                "bp_is_range": True,
                "bp_parse_score": candidate_score(raw_text, pressure_reason, True),
            })
            consumed_spans.append(m.span())

    def inside_consumed(span: Tuple[int, int]) -> bool:
        return any(span[0] >= s[0] and span[1] <= s[1] for s in consumed_spans)

    for m in BP_SINGLE_RE.finditer(raw_text):
        if inside_consumed(m.span()):
            continue

        v = float(m.group("v"))
        unit = clean_unit(m.group("unit"))
        c = convert_to_c(v, unit)

        if plausible_bp_c(c):
            candidates.append({
                "bp_c_pubchem": round(c, 3),
                "bp_c_min_pubchem": round(c, 3),
                "bp_c_max_pubchem": round(c, 3),
                "bp_unit_original": unit,
                "bp_raw_pubchem": raw_text,
                "bp_pressure_reason": pressure_reason,
                "bp_is_range": False,
                "bp_parse_score": candidate_score(raw_text, pressure_reason, False),
            })

    return candidates


def choose_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda c: (
            c.get("bp_parse_score", 0),
            1 if c.get("bp_is_range") else 0,
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


def resolve_cid_from_row(
    row: pd.Series,
    cas_col: Optional[str],
    name_col: Optional[str],
    cid_col: Optional[str],
    sleep: float,
) -> Tuple[Optional[int], str]:

    cas = str(row.get(cas_col, "")).strip() if cas_col else ""
    name = str(row.get(name_col, "")).strip() if name_col else ""
    existing_cid = str(row.get(cid_col, "")).strip() if cid_col else ""

    # Prefer CAS because your previous file had wrong CID assignments.
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


def extract_boiling_point_texts(compound_view: Dict[str, Any]) -> List[str]:
    texts: List[str] = []

    try:
        root_sections = compound_view["Record"]["Section"]
    except Exception:
        return texts

    for root in root_sections:
        for path, sec in walk_sections(root):
            path_l = path.lower()
            heading_l = str(sec.get("TOCHeading", "")).lower()

            is_bp_section = ("boiling point" in path_l) or (heading_l.strip() == "boiling point")
            if not is_bp_section:
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


def get_best_bp_for_cid(
    cid: int,
    cache_dir: Optional[Path],
    sleep: float = DEFAULT_SLEEP,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[str], Optional[str]]:

    view = get_compound_view(cid, cache_dir=cache_dir, sleep=sleep)
    if not view:
        return None, [], [], None

    title = get_pubchem_title(view)
    bp_texts = extract_boiling_point_texts(view)

    all_candidates: List[Dict[str, Any]] = []
    for text in bp_texts:
        all_candidates.extend(parse_bp_candidates(text))

    best = choose_best_candidate(all_candidates)
    return best, all_candidates, bp_texts, title


OUTPUT_COLUMNS = [
    "bp_c_pubchem_clean",
    "bp_c_min_pubchem_clean",
    "bp_c_max_pubchem_clean",
    "bp_raw_pubchem_clean",
    "bp_unit_original_clean",
    "bp_pressure_reason",
    "bp_is_range",
    "bp_parse_score",
    "resolved_cid_clean",
    "cid_resolution_method",
    "pubchem_title_clean",
    "bp_status",
    "bp_error",
]


def process_file(
    input_file: Path,
    output_file: Path,
    autosave_every: int = 25,
    cache_dir: Optional[Path] = None,
    sleep: float = DEFAULT_SLEEP,
    no_resume: bool = False,
) -> None:

    df = pd.read_csv(input_file)

    cas_col = find_column(df, ["CAS", "CAS Number", "CAS No", "cas_number", "cas_no"])
    name_col = find_column(df, ["Name of Material", "Name", "Material", "material_name", "Chemical Name", "Synonym"])
    cid_col = find_column(df, ["CID", "PubChem CID", "cid", "pubchem_cid"])

    print("Input file:", input_file)
    print("Output file:", output_file)
    print("Rows:", len(df))
    print("Detected columns:")
    print("  CAS column :", cas_col)
    print("  Name column:", name_col)
    print("  CID column :", cid_col)
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

    pbar = tqdm(df.iterrows(), total=len(df), desc="Retrieving PubChem BP", unit="mol")

    for idx, row in pbar:
        key = row_key(row, idx, cas_col, name_col, cid_col)

        if key in processed_by_key and not no_resume:
            existing = processed_by_key[key]
            results.append(existing)
            resumed_count += 1
            if pd.notna(existing.get("bp_c_pubchem_clean")):
                found_count += 1
            pbar.set_postfix({"found BP": found_count, "errors": error_count, "resumed": resumed_count})
            continue

        out = row.to_dict()
        out["_processing_key"] = key

        for col in OUTPUT_COLUMNS:
            out[col] = None

        try:
            cid, method = resolve_cid_from_row(row, cas_col, name_col, cid_col, sleep=sleep)

            out["resolved_cid_clean"] = cid
            out["cid_resolution_method"] = method

            if not cid:
                out["bp_status"] = "no_cid"
                results.append(out)
                pbar.set_postfix({"found BP": found_count, "errors": error_count, "resumed": resumed_count})
                continue

            best, candidates, bp_texts, title = get_best_bp_for_cid(cid, cache_dir=cache_dir, sleep=sleep)
            out["pubchem_title_clean"] = title

            if best:
                out["bp_c_pubchem_clean"] = best.get("bp_c_pubchem")
                out["bp_c_min_pubchem_clean"] = best.get("bp_c_min_pubchem")
                out["bp_c_max_pubchem_clean"] = best.get("bp_c_max_pubchem")
                out["bp_raw_pubchem_clean"] = best.get("bp_raw_pubchem")
                out["bp_unit_original_clean"] = best.get("bp_unit_original")
                out["bp_pressure_reason"] = best.get("bp_pressure_reason")
                out["bp_is_range"] = best.get("bp_is_range")
                out["bp_parse_score"] = best.get("bp_parse_score")
                out["bp_status"] = "found"
                found_count += 1
            else:
                if bp_texts:
                    out["bp_status"] = "bp_section_found_but_no_accepted_760_value"
                    out["bp_raw_pubchem_clean"] = " | ".join(bp_texts[:5])
                else:
                    out["bp_status"] = "no_boiling_point_section"

        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving partial output...")
            results.append(out)
            pd.DataFrame(results).to_csv(output_file, index=False)
            raise

        except Exception as e:
            out["bp_status"] = "error"
            out["bp_error"] = repr(e)
            error_count += 1

        results.append(out)

        pbar.set_postfix({"found BP": found_count, "errors": error_count, "resumed": resumed_count})

        if autosave_every and len(results) % autosave_every == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)

    final = pd.DataFrame(results)
    final.to_csv(output_file, index=False)

    print()
    print("DONE")
    print("Rows processed:", len(final))
    print("Found BP:", final["bp_c_pubchem_clean"].notna().sum())
    print("Saved to:", output_file)


def test_cid(cid: int, cache_dir: Optional[Path], sleep: float) -> None:
    print(f"Testing CID {cid}")
    best, candidates, bp_texts, title = get_best_bp_for_cid(cid, cache_dir=cache_dir, sleep=sleep)

    print("PubChem title:", title)
    print()
    print("Boiling point texts found:")

    if not bp_texts:
        print("  No BP texts found.")

    for i, t in enumerate(bp_texts, start=1):
        accepted, reason = pressure_status(t)
        print(f"  {i}. {t}")
        print(f"     pressure: {'ACCEPT' if accepted else 'REJECT'} / {reason}")

    print()
    print("Parsed accepted candidates:")

    if not candidates:
        print("  No accepted candidates.")

    for c in candidates:
        print(
            f"  BP={c['bp_c_pubchem']} °C "
            f"(min={c['bp_c_min_pubchem']}, max={c['bp_c_max_pubchem']}), "
            f"unit={c['bp_unit_original']}, range={c['bp_is_range']}, "
            f"score={c['bp_parse_score']}, reason={c['bp_pressure_reason']}"
        )
        print(f"     raw: {c['bp_raw_pubchem']}")

    print()
    print("BEST:")
    print(best)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve PubChem boiling points at standard pressure and save Celsius values."
    )

    parser.add_argument("input_file", nargs="?", help="Input CSV file")
    parser.add_argument("output_file", nargs="?", help="Output CSV file")
    parser.add_argument("--autosave-every", type=int, default=25, help="Autosave every N rows. Default: 25")
    parser.add_argument("--cache-dir", default=".pubchem_cache_bp", help="Cache directory for PubChem JSON")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="Delay between PubChem requests")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing output file")
    parser.add_argument("--test-cid", type=int, help="Test parser for one PubChem CID, e.g. --test-cid 6549")

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.test_cid:
        test_cid(args.test_cid, cache_dir=cache_dir, sleep=args.sleep)
        return

    if not args.input_file or not args.output_file:
        parser.error("input_file and output_file are required unless --test-cid is used")

    process_file(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        autosave_every=args.autosave_every,
        cache_dir=cache_dir,
        sleep=args.sleep,
        no_resume=args.no_resume,
    )


if __name__ == "__main__":
    main()
