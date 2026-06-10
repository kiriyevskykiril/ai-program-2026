#!/usr/bin/env python3
"""
pubchem_vp_roomtemp_cleaner_v4.py

Clean PubChem vapor-pressure text values into numeric room-temperature features.

Main improvements in v4:
- Supports temperatures in Celsius, Fahrenheit, and Kelvin.
- Treats Kelvin values such as 294 K as room temperature after conversion.
- Converts VP units to Pa and mmHg.
- Creates log10(VP_Pa), useful for ML.
- Prefers explicitly room-temperature values, then falls back to usable VP values.

Example:
    python pubchem_vp_roomtemp_cleaner_v4.py input.csv output.csv --raw-col vp_raw_pubchem_clean
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


ROOM_TEMP_MIN_C = 18.0
ROOM_TEMP_MAX_C = 30.0
MMHG_TO_PA = 133.322368


@dataclass
class VPCandidate:
    vp_pa: float
    vp_mmhg: float
    vp_log10_pa: float
    vp_temperature_c: Optional[float]
    vp_raw_selected: str
    has_explicit_roomtemp: bool


def kelvin_to_celsius(k: float) -> float:
    return k - 273.15


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def is_room_temperature(temp_c: Optional[float]) -> bool:
    return temp_c is not None and ROOM_TEMP_MIN_C <= temp_c <= ROOM_TEMP_MAX_C


def extract_temperature_c(text: str) -> Optional[float]:
    """
    Extract the first plausible temperature from text and convert it to Celsius.

    Supported examples:
    - 25 °C
    - 25 deg C
    - 77 °F
    - 294 K
    - @ 294 deg K
    """
    text = str(text)

    temp_patterns = [
        (r"([-+]?\d+(?:\.\d+)?)\s*(?:°\s*C|deg\s*C|\bC\b)", "C"),
        (r"([-+]?\d+(?:\.\d+)?)\s*(?:°\s*F|deg\s*F|\bF\b)", "F"),
        (r"([-+]?\d+(?:\.\d+)?)\s*(?:°\s*K|deg\s*K|\bK\b)", "K"),
    ]

    for pattern, unit in temp_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = float(match.group(1))

            if unit == "C":
                temp_c = value
            elif unit == "F":
                temp_c = fahrenheit_to_celsius(value)
            else:
                temp_c = kelvin_to_celsius(value)

            # Basic sanity check to avoid accidentally reading unrelated numbers.
            if -100.0 <= temp_c <= 250.0:
                return temp_c

    return None


def extract_vp_values_pa(text: str) -> list[float]:
    """
    Extract vapor pressure values and convert all to Pa.

    Supported units:
    - mmHg / mm Hg
    - torr
    - Pa
    - kPa

    Notes:
    The Pa regex uses a negative lookbehind to avoid matching kPa twice.
    """
    text = str(text)
    values_pa: list[float] = []

    patterns = [
        (r"([-+]?\d+(?:\.\d+)?)\s*\[?\s*mm\s*Hg\s*\]?", "mmHg"),
        (r"([-+]?\d+(?:\.\d+)?)\s*\[?\s*mmHg\s*\]?", "mmHg"),
        (r"([-+]?\d+(?:\.\d+)?)\s*torr\b", "mmHg"),
        (r"([-+]?\d+(?:\.\d+)?)\s*kPa\b", "kPa"),
        (r"(?<!k)\b([-+]?\d+(?:\.\d+)?)\s*\[?\s*Pa\s*\]?\b", "Pa"),
    ]

    for pattern, unit in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = float(match.group(1))

            if value <= 0:
                continue

            if unit == "mmHg":
                values_pa.append(value * MMHG_TO_PA)
            elif unit == "kPa":
                values_pa.append(value * 1000.0)
            elif unit == "Pa":
                values_pa.append(value)

    return values_pa


def make_candidate(part: str) -> Optional[VPCandidate]:
    temp_c = extract_temperature_c(part)
    vp_values_pa = extract_vp_values_pa(part)

    if not vp_values_pa:
        return None

    # If several equivalent values are reported in the same part, use median.
    vp_pa = float(np.median(vp_values_pa))

    return VPCandidate(
        vp_pa=vp_pa,
        vp_mmhg=vp_pa / MMHG_TO_PA,
        vp_log10_pa=float(np.log10(vp_pa)),
        vp_temperature_c=temp_c,
        vp_raw_selected=part.strip(),
        has_explicit_roomtemp=is_room_temperature(temp_c),
    )


def split_raw_entry(raw_text: str) -> list[str]:
    """
    Split raw PubChem text into smaller candidate fragments.

    PubChem values often look like:
    '0.05 [mmHg] | 5.45 Pa @ 294 deg K'
    or
    '5 mmHg at 194 °F ; 1 mmHg at 143 °F'
    """
    parts = re.split(r"\s*[;|]\s*", str(raw_text))
    return [p.strip() for p in parts if p.strip()]


def empty_result(status: str, raw_selected=np.nan, fallback_used=False) -> dict:
    return {
        "vp_pa": np.nan,
        "vp_mmhg": np.nan,
        "vp_log10_pa": np.nan,
        "vp_temperature_c": np.nan,
        "vp_status": status,
        "vp_fallback_used": fallback_used,
        "vp_raw_selected": raw_selected,
    }


def clean_vp_entry(raw_text) -> dict:
    """
    Clean one raw PubChem vapor pressure entry.

    Priority:
    1. Explicit room-temperature value: 18-30 C after unit conversion.
    2. Value without explicit temperature, as fallback.
    3. Non-room-temperature value, as fallback.

    If multiple room-temperature values exist, the one closest to 25 C is selected.
    """
    if pd.isna(raw_text):
        return empty_result("missing_raw_text")

    raw_text = str(raw_text).strip()
    if not raw_text:
        return empty_result("missing_raw_text")

    candidates: list[VPCandidate] = []
    for part in split_raw_entry(raw_text):
        candidate = make_candidate(part)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return empty_result("no_accepted_vp_value", raw_selected=raw_text)

    roomtemp_candidates = [c for c in candidates if c.has_explicit_roomtemp]
    no_temp_candidates = [c for c in candidates if c.vp_temperature_c is None]
    non_roomtemp_candidates = [
        c for c in candidates
        if c.vp_temperature_c is not None and not c.has_explicit_roomtemp
    ]

    if roomtemp_candidates:
        selected = min(
            roomtemp_candidates,
            key=lambda c: abs(float(c.vp_temperature_c) - 25.0),
        )
        status = "found_roomtemp_explicit"
        fallback = False
    elif no_temp_candidates:
        selected = no_temp_candidates[0]
        status = "found_no_temperature_fallback"
        fallback = True
    else:
        # If forced to use a non-room-temperature value, select the one closest to 25 C.
        selected = min(
            non_roomtemp_candidates,
            key=lambda c: abs(float(c.vp_temperature_c) - 25.0),
        )
        status = "found_non_roomtemp_fallback"
        fallback = True

    return {
        "vp_pa": selected.vp_pa,
        "vp_mmhg": selected.vp_mmhg,
        "vp_log10_pa": selected.vp_log10_pa,
        "vp_temperature_c": selected.vp_temperature_c,
        "vp_status": status,
        "vp_fallback_used": fallback,
        "vp_raw_selected": selected.vp_raw_selected,
    }


def clean_vapor_pressure_dataframe(df: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    cleaned = df[raw_col].apply(clean_vp_entry)
    cleaned_df = pd.DataFrame(cleaned.tolist())
    return pd.concat([df.reset_index(drop=True), cleaned_df], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean PubChem vapor-pressure values near room temperature."
    )
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument(
        "--raw-col",
        default="vp_raw_pubchem_clean",
        help="Column containing raw PubChem vapor-pressure text",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.raw_col not in df.columns:
        raise ValueError(
            f"Column '{args.raw_col}' not found. Available columns: {list(df.columns)}"
        )

    out_df = clean_vapor_pressure_dataframe(df, raw_col=args.raw_col)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Saved: {args.output_csv}")
    print()
    print("vp_status counts:")
    print(out_df["vp_status"].value_counts(dropna=False))
    print()
    print("vp_fallback_used counts:")
    print(out_df["vp_fallback_used"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
