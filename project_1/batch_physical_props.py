"""
Batch-fetch boiling point and vapor pressure from PubChem for many CIDs.

Uses the same pipeline as the notebook: ``pubchem_retriever2.get_pubchem_physical_props``
→ ``practical_parser2`` parsers. Results are cached on disk so you can resume after
interrupts and avoid re-hitting PubChem for compounds you already fetched.

Typical use from a notebook::

    from pathlib import Path
    import pandas as pd
    from batch_physical_props import fetch_props_for_dataframe, merge_props

    df = pd.read_csv(Path("data/waka_dragon_merged.csv"))
    props = fetch_props_for_dataframe(df, cid_column="CID")
    out = merge_props(df, props)
    out.to_csv(Path("data/waka_dragon_with_props.csv"), index=False)

PubChem: keep ``delay_s`` at ~0.2–0.5 s between compounds to stay polite; use
``resume=True`` and a ``cache_path`` so reruns are cheap.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

from pubchem_retriever2 import get_pubchem_physical_props

logger = logging.getLogger(__name__)

_DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "pubchem_physical_props_cache.csv"


def _parsed_bp_c_vp_mmhg(result: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Extract scalar features from ``get_pubchem_physical_props`` output."""
    bp_parsed = result.get("boiling_point_parsed")
    vp_parsed = result.get("vapor_pressure_parsed")

    bp_c: Optional[float] = None
    if isinstance(bp_parsed, dict) and "value" in bp_parsed:
        bp_c = float(bp_parsed["value"])

    vp_mmhg: Optional[float] = None
    if isinstance(vp_parsed, dict) and "value_mmhg" in vp_parsed:
        vp_mmhg = float(vp_parsed["value_mmhg"])

    return bp_c, vp_mmhg


def _normalize_cid(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.is_file():
        return pd.DataFrame(
            columns=[
                "cid",
                "boiling_point_c",
                "vapor_pressure_mmhg",
                "error",
            ]
        )
    df = pd.read_csv(cache_path)
    if "cid" not in df.columns:
        raise ValueError(f"Cache {cache_path} must contain a 'cid' column")
    return df


def _append_cache_row(cache_path: Path, row: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    header = not cache_path.is_file()
    df.to_csv(cache_path, mode="a", header=header, index=False)


def fetch_props_for_cids(
    cids: Iterable[int],
    *,
    delay_s: float = 0.35,
    max_retries: int = 4,
    retry_backoff_s: float = 2.0,
    preferred_vp_temperature_c: float = 25.0,
    cache_path: Optional[Path] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Fetch parsed boiling point (°C) and vapor pressure (mmHg) for each unique CID.

    Parameters
    ----------
    cids
        PubChem compound IDs (duplicates are ignored).
    delay_s
        Sleep between successful requests (PubChem etiquette).
    max_retries / retry_backoff_s
        Retries on HTTP errors and ``requests`` failures with exponential backoff.
    preferred_vp_temperature_c
        Passed through to ``parse_vapor_pressure`` (default 25 °C).
    cache_path
        CSV path for one-row-per-cid cache; defaults to ``data/pubchem_physical_props_cache.csv``.
    resume
        If True, skip CIDs already present in the cache file.

    Returns
    -------
    DataFrame with columns: cid, boiling_point_c, vapor_pressure_mmhg, error
    (error is empty string on success; otherwise a short message).
    """
    cache_path = Path(cache_path) if cache_path is not None else _DEFAULT_CACHE
    seen: List[int] = []
    for c in cids:
        if c is None:
            continue
        ci = int(c)
        if ci not in seen:
            seen.append(ci)

    if not resume and cache_path.is_file():
        cache_path.unlink()

    cached = _load_cache(cache_path) if resume else pd.DataFrame()
    done: set[int] = set()
    if resume and not cached.empty:
        done = set(cached["cid"].astype(int).tolist())

    rows: List[Dict[str, Any]] = []
    if resume and not cached.empty:
        rows.extend(cached.to_dict("records"))

    def fetch_one(cid: int) -> Dict[str, Any]:
        last_err = ""
        for attempt in range(max_retries):
            try:
                result = get_pubchem_physical_props(
                    cid,
                    parse=True,
                    preferred_vp_temperature_c=preferred_vp_temperature_c,
                )
                bp_c, vp_mmhg = _parsed_bp_c_vp_mmhg(result)
                return {
                    "cid": cid,
                    "boiling_point_c": bp_c,
                    "vapor_pressure_mmhg": vp_mmhg,
                    "error": "",
                }
            except requests.HTTPError as e:
                last_err = f"HTTPError {e.response.status_code if e.response else '?'}"
                code = e.response.status_code if e.response else None
                wait = retry_backoff_s * (2**attempt)
                if code == 429:
                    wait = max(wait, 30.0)
                logger.warning("CID %s: %s (attempt %s, sleeping %.1fs)", cid, last_err, attempt + 1, wait)
                time.sleep(wait)
            except (requests.RequestException, ValueError, KeyError, TypeError) as e:
                last_err = repr(e)
                wait = retry_backoff_s * (2**attempt)
                logger.warning("CID %s: %s (attempt %s, sleeping %.1fs)", cid, last_err, attempt + 1, wait)
                time.sleep(wait)

        return {
            "cid": cid,
            "boiling_point_c": None,
            "vapor_pressure_mmhg": None,
            "error": last_err or "fetch_failed",
        }

    fetched = 0
    for cid in seen:
        if cid in done:
            continue
        row = fetch_one(cid)
        rows.append(row)
        _append_cache_row(cache_path, row)
        done.add(cid)
        fetched += 1
        logger.info(
            "Fetched CID %s (%s of %s unique CIDs this run): bp=%s vp=%s err=%s",
            cid,
            fetched,
            len(seen),
            row["boiling_point_c"],
            row["vapor_pressure_mmhg"],
            row["error"] or "-",
        )
        time.sleep(delay_s)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["cid"], keep="last").sort_values("cid").reset_index(drop=True)
    return out


def fetch_props_for_dataframe(
    df: pd.DataFrame,
    *,
    cid_column: str = "CID",
    **kwargs: Any,
) -> pd.DataFrame:
    """Collect unique CIDs from ``df[cid_column]`` and run :func:`fetch_props_for_cids`."""
    raw = df[cid_column].map(_normalize_cid)
    cids_list = [c for c in raw.dropna().unique().tolist() if c is not None]
    cids_int = [int(c) for c in cids_list]
    return fetch_props_for_cids(cids_int, **kwargs)


def merge_props(
    waka_like: pd.DataFrame,
    props: pd.DataFrame,
    *,
    cid_column: str = "CID",
    bp_column: str = "bp_c_pubchem",
    vp_column: str = "vp_mmhg_pubchem",
) -> pd.DataFrame:
    """Left-merge parsed props onto a Wakayama/Dragon frame by CID."""
    p = props.rename(
        columns={
            "cid": cid_column,
            "boiling_point_c": bp_column,
            "vapor_pressure_mmhg": vp_column,
        }
    )
    # Drop error column from merge surface (keep only if user wants it)
    if "error" in p.columns:
        err_col = f"{bp_column}_fetch_error"
        p = p.rename(columns={"error": err_col})
    out = waka_like.merge(p, on=cid_column, how="left")
    return out
