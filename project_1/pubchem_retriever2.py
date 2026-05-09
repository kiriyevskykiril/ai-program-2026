import requests

from practical_parser2 import parse_boiling_point, parse_vapor_pressure


def extract_pubchem_heading_value(cid, heading):
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
        f"{cid}/JSON?heading={heading.replace(' ', '+')}"
    )

    r = requests.get(url, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    data = r.json()
    values = []

    def walk(obj):
        if isinstance(obj, dict):
            if "StringWithMarkup" in obj:
                for item in obj["StringWithMarkup"]:
                    text = item.get("String")
                    if text:
                        values.append(text)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return values or None


def get_pubchem_physical_props(cid, parse=True, preferred_vp_temperature_c=25.0):
    """
    Retrieve boiling point and vapor pressure annotations from PubChem PUG-View.

    By default, the function returns both raw annotations and parsed normalized values:
    - boiling_point_parsed: Celsius, preferring values at 760 mmHg.
    - vapor_pressure_parsed: mmHg, preferring values measured closest to 25 °C.

    Set parse=False to return only the raw PubChem annotation lists.
    """
    boiling_point_raw = extract_pubchem_heading_value(cid, "Boiling Point")
    vapor_pressure_raw = extract_pubchem_heading_value(cid, "Vapor Pressure")

    result = {
        "cid": cid,
        "boiling_point": boiling_point_raw,
        "vapor_pressure": vapor_pressure_raw,
    }

    if parse:
        result.update({
            "boiling_point_parsed": parse_boiling_point(boiling_point_raw),
            "vapor_pressure_parsed": parse_vapor_pressure(
                vapor_pressure_raw,
                preferred_temperature_c=preferred_vp_temperature_c,
            ),
        })

    return result
