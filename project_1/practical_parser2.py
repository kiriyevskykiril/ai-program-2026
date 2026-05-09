import re
from typing import Any, Dict, List, Optional

MMHG_PER_ATM = 760.0
PA_PER_MMHG = 133.322


def _is_estimated(text: str) -> bool:
    return bool(re.search(r"\b(est|estimated|estimate|calc|calculated|predicted)\b", text, re.I))


def _to_celsius(value: float, unit: str) -> float:
    unit = unit.upper()
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0
    return value


def _pressure_to_mmhg(value: float, unit: str) -> Optional[float]:
    unit = unit.lower().replace(" ", "")
    if unit in {"mmhg", "torr"}:
        return value
    if unit == "pa":
        return value / PA_PER_MMHG
    if unit == "kpa":
        return value * 1000.0 / PA_PER_MMHG
    if unit == "atm":
        return value * MMHG_PER_ATM
    return None


def _extract_temperature_c(text: str, start: int = 0) -> Optional[float]:
    """Return the first temperature after start, converted to Celsius."""
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\b", text[start:], re.I)
    if not m:
        return None
    return round(_to_celsius(float(m.group(1)), m.group(2)), 3)


def _extract_pressure_mmhg(text: str) -> Optional[float]:
    m = re.search(
        r"(?:at|@)\s*(\d+(?:\.\d+)?)\s*(mm\s*hg|mmhg|torr|pa|kpa|atm)\b",
        text,
        re.I,
    )
    if not m:
        return None
    return _pressure_to_mmhg(float(m.group(1)), m.group(2))


def parse_boiling_point(values: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """
    Parse boiling-point strings from PubChem.

    Selection rule:
    1. Prefer non-estimated values.
    2. Prefer Celsius values over converted Fahrenheit values.
    3. Prefer values measured at standard pressure, 760 mmHg.
    4. Prefer ranges at 760 mmHg, because they usually represent a curated experimental interval.

    Returns value in Celsius. If the best record is a range, value is the midpoint and
    range_min_c / range_max_c are also returned.
    """
    if not values:
        return None

    candidates: List[Dict[str, Any]] = []

    range_pattern = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(?:to|-|–)\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\b",
        re.I,
    )
    single_pattern = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\b", re.I)

    for order, text in enumerate(values):
        pressure_mmhg = _extract_pressure_mmhg(text)
        standard_pressure = pressure_mmhg is not None and abs(pressure_mmhg - 760.0) <= 2.0
        estimated = _is_estimated(text)

        for m in range_pattern.finditer(text):
            lo = _to_celsius(float(m.group(1)), m.group(3))
            hi = _to_celsius(float(m.group(2)), m.group(3))
            if lo > hi:
                lo, hi = hi, lo
            candidates.append({
                "value": round((lo + hi) / 2.0, 3),
                "unit": "°C",
                "range_min_c": round(lo, 3),
                "range_max_c": round(hi, 3),
                "pressure_mmhg": round(pressure_mmhg, 3) if pressure_mmhg is not None else None,
                "standard_pressure": standard_pressure,
                "estimated": estimated,
                "raw": text,
                "_score": (
                    100 * (not estimated)
                    + 70 * standard_pressure
                    + 25 * (m.group(3).upper() == "C")
                    + 15  # range bonus
                    - order * 0.01
                ),
            })

        # Add single values only if this text did not contain a range.
        if not range_pattern.search(text):
            for m in single_pattern.finditer(text):
                value_c = _to_celsius(float(m.group(1)), m.group(2))
                candidates.append({
                    "value": round(value_c, 3),
                    "unit": "°C",
                    "range_min_c": None,
                    "range_max_c": None,
                    "pressure_mmhg": round(pressure_mmhg, 3) if pressure_mmhg is not None else None,
                    "standard_pressure": standard_pressure,
                    "estimated": estimated,
                    "raw": text,
                    "_score": (
                        100 * (not estimated)
                        + 70 * standard_pressure
                        + 25 * (m.group(2).upper() == "C")
                        - order * 0.01
                    ),
                })

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x["_score"])
    best.pop("_score", None)
    return best


def _pressure_candidates_from_text(text: str, order: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    estimated = _is_estimated(text)

    # Common form: "0.094 mm Hg at 25 °C" or "0.1 mmHg at 68 °F".
    value_unit_pattern = re.compile(
        r"(\d+(?:\.\d+)?(?:[xX]\s*10[+-]?\d+)?)\s*(?:\[)?\s*(mm\s*hg|mmhg|torr|pa|kpa|atm)(?:\])?",
        re.I,
    )

    for m in value_unit_pattern.finditer(text):
        raw_value = m.group(1)
        if re.search(r"[xX]\s*10", raw_value):
            base, exp = re.split(r"[xX]\s*10", raw_value)
            value = float(base) * (10 ** int(exp))
        else:
            value = float(raw_value)

        value_mmhg = _pressure_to_mmhg(value, m.group(2))
        if value_mmhg is None:
            continue

        temperature_c = _extract_temperature_c(text, m.end())
        candidates.append({
            "value_mmhg": round(value_mmhg, 6),
            "temperature_c": temperature_c,
            "estimated": estimated,
            "raw": text,
            "_order": order,
        })

    # PubChem sometimes has: "Vapor pressure, Pa at 20 °C: 13.2".
    unit_before_value_pattern = re.compile(
        r"\b(mm\s*hg|mmhg|torr|pa|kpa|atm)\b\s*at\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\b\s*:\s*(\d+(?:\.\d+)?)",
        re.I,
    )

    for m in unit_before_value_pattern.finditer(text):
        value_mmhg = _pressure_to_mmhg(float(m.group(4)), m.group(1))
        if value_mmhg is None:
            continue
        candidates.append({
            "value_mmhg": round(value_mmhg, 6),
            "temperature_c": round(_to_celsius(float(m.group(2)), m.group(3)), 3),
            "estimated": estimated,
            "raw": text,
            "_order": order,
        })

    return candidates


def parse_vapor_pressure(values: Optional[List[str]], preferred_temperature_c: float = 25.0) -> Optional[Dict[str, Any]]:
    """
    Parse vapor-pressure strings from PubChem.

    Selection rule:
    1. Prefer non-estimated values.
    2. Prefer records with temperature explicitly stated.
    3. Prefer temperature closest to preferred_temperature_c, default 25 °C.
    4. Prefer cleaner/single-value records over multi-value records when scores tie.

    Returns vapor pressure normalized to mmHg.
    """
    if not values:
        return None

    candidates: List[Dict[str, Any]] = []
    for order, text in enumerate(values):
        candidates.extend(_pressure_candidates_from_text(text, order))

    if not candidates:
        return None

    for c in candidates:
        temp = c["temperature_c"]
        temp_distance = abs(temp - preferred_temperature_c) if temp is not None else 999.0
        has_temp = temp is not None
        multi_value_penalty = text_count = len(re.findall(r"\d+(?:\.\d+)?\s*(?:\[)?\s*(?:mm\s*hg|mmhg|torr|pa|kpa|atm)", c["raw"], re.I))
        c["_score"] = (
            1000 * (not c["estimated"])
            + 500 * has_temp
            - 20 * temp_distance
            - 2 * multi_value_penalty
            - c["_order"] * 0.01
        )

    best = max(candidates, key=lambda x: x["_score"])
    best.pop("_score", None)
    best.pop("_order", None)
    # Keep the normalized value compact, but do not over-round small values.
    best["value_mmhg"] = round(best["value_mmhg"], 4)
    return best
