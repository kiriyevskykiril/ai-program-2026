import pandas as pd
from pathlib import Path
import requests
import re, time, requests


PATH_TO_DATA_FOLDER = Path(r'C:\src\ai-program-2026\project_1\data')

wakayama_file_name = 'Wakayama_db.csv'
dragon_file_name = 'raw_dragon_matrix.csv'

wakayama_file_path = PATH_TO_DATA_FOLDER / wakayama_file_name
dragon_file_path = PATH_TO_DATA_FOLDER / dragon_file_name

def cas_to_cid(cas: str) -> int | None:
    """ Return Pubchem CID for a given CAS."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/cids/TXT"
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.text.strip():
            return int(r.text.strip().split()[0])
    except Exception:
        pass
    return None


def get_waka_df(describe=False, add_cid=True) -> pd.DataFrame:
    df = pd.read_csv(wakayama_file_path)
    if describe:
        print(f"Wakayama df shape: {df.shape}")
        print(f"The number of unique CASes are {len(df['CAS'].unique())}")

    if add_cid:
        print("Fetching Pubchem CIDs (this may take time)")
        df['CID'] = df['CAS'].apply(lambda x: cas_to_cid(str(x)))
        df['CID'] = df['CID'].astype('Int64')
        columns = ['CID'] + [c for c in df.columns if c != 'CID']
        df = df[columns]

    # show how many None/NaN values
    missing_count = df['CID'].isna().sum()
    print(f"🔍 Missing CID count: {missing_count} out of {len(df)} rows")

    df = df.dropna(subset=['CID'])
    print("Rows with missing CID were deleted")
    return df

def get_dragon_df():
    df = pd.read_csv(dragon_file_path)
    return df


def nist_cas_to_inchikey(cas: str) -> str | None:
    """Get InChIKey from NIST WebBook by CAS."""
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://webbook.nist.gov/cgi/cbook.cgi?ID={cas}&Units=SI",
        f"https://webbook.nist.gov/cgi/cbook.cgi?ID={cas}&Units=SI&Mask=200#Names",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.ok:
                m = re.search(r'InChIKey=([A-Z0-9\-]{27})', r.text)
                if m:
                    return m.group(1)
        except Exception:
            pass
        time.sleep(0.2)
    return None

def inchikey_to_cid_pubchem(inchikey: str) -> int | None:
    """Resolve PubChem CID by InChIKey."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/cids/TXT"
        r = requests.get(url, timeout=12)
        if r.ok and r.text.strip():
            return int(r.text.strip().split()[0])
    except Exception:
        pass
    return None




