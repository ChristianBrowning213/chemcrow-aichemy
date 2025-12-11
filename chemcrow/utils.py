import re
import time

import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def is_smiles(text):
    try:
        m = Chem.MolFromSmiles(text, sanitize=False)
        if m is None:
            return False
        return True
    except:
        return False


def is_multiple_smiles(text):
    if is_smiles(text):
        return "." in text
    return False


def split_smiles(text):
    return text.split(".")


def is_cas(text):
    pattern = r"^\d{2,7}-\d{2}-\d$"
    return re.match(pattern, text) is not None


def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    while not is_smiles(ss[-1]):
        rm = ss[-1]
        ss.remove(rm)
    return ss[-1]


def canonical_smiles(smiles):
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        return smi
    except Exception:
        return "Invalid SMILES string"


def tanimoto(s1, s2):
    """Calculate the Tanimoto similarity of two SMILES strings."""
    try:
        mol1 = Chem.MolFromSmiles(s1)
        mol2 = Chem.MolFromSmiles(s2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except (TypeError, ValueError, AttributeError):
        return "Error: Not a valid SMILES string"


def pubchem_query2smiles(
    query: str,
    url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}",
) -> str:
    if is_smiles(query):
        if not is_multiple_smiles(query):
            return query
        else:
            raise ValueError(
                "Multiple SMILES strings detected, input one molecule at a time."
            )
    
    errors = []
    
    # **Attempt 1: PubChem API** (most reliable for chemical names)
    print(f"[DEBUG] Attempt 1: Querying PubChem for '{query}'...")
    try:
        if url is None:
            url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
        
        r = requests.get(
            url.format(query, "property/IsomericSMILES/JSON"),
            timeout=10
        )
        r.raise_for_status()
        
        data = r.json()
        
        # Get SMILES from response (try IsomericSMILES first, then SMILES)
        if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
            props = data["PropertyTable"]["Properties"][0]
            smi = props.get("IsomericSMILES") or props.get("SMILES")
            
            if smi:
                print(f"[DEBUG] ✓ PubChem found: {smi}")
                return str(Chem.CanonSmiles(largest_mol(smi)))
        
        errors.append("PubChem: No SMILES data in response")
        
    except requests.exceptions.RequestException as e:
        errors.append(f"PubChem API error: {str(e)[:50]}")
    except Exception as e:
        errors.append(f"PubChem parse error: {str(e)[:50]}")
    
    # **Attempt 2: Cactus NCI API** (better for brand names)
    print(f"[DEBUG] Attempt 2: Querying Cactus NCI for '{query}'...")
    try:
        # Cactus is very simple - just returns SMILES as text
        cactus_url = f"https://cactus.nci.nih.gov/chemical/structure/{query}/smiles"
        r = requests.get(cactus_url, timeout=10)
        r.raise_for_status()
        
        smi = r.text.strip()
        
        # Validate it's valid SMILES
        if smi and is_smiles(smi):
            print(f"[DEBUG] ✓ Cactus found: {smi}")
            return str(Chem.CanonSmiles(largest_mol(smi)))
        else:
            errors.append(f"Cactus: Invalid SMILES '{smi}'")
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            errors.append("Cactus: Molecule not found (404)")
        else:
            errors.append(f"Cactus HTTP error: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        errors.append(f"Cactus API error: {str(e)[:50]}")
    except Exception as e:
        errors.append(f"Cactus parse error: {str(e)[:50]}")
    
    # **All APIs failed**
    error_summary = " | ".join(errors)
    print(f"[DEBUG] ✗ All APIs failed: {error_summary}")
    
    raise ValueError(
        f"Could not find a molecule matching '{query}'. "
        f"Tried: PubChem API, Cactus NCI API. "
        f"Errors: {error_summary}. "
        f"Please verify the molecule name is correct."
    )


def query2cas(query: str, url_cid: str, url_data: str):
    try:
        mode = "name"
        if is_smiles(query):
            if is_multiple_smiles(query):
                raise ValueError(
                    "Multiple SMILES strings detected, input one molecule at a time."
                )
            mode = "smiles"
        url_cid = url_cid.format(mode, query)
        cid = requests.get(url_cid).json()["IdentifierList"]["CID"][0]
        url_data = url_data.format(cid)
        data = requests.get(url_data).json()
    except (requests.exceptions.RequestException, KeyError):
        raise ValueError("Invalid molecule input, no Pubchem entry")

    try:
        for section in data["Record"]["Section"]:
            if section.get("TOCHeading") == "Names and Identifiers":
                for subsection in section["Section"]:
                    if subsection.get("TOCHeading") == "Other Identifiers":
                        for subsubsection in subsection["Section"]:
                            if subsubsection.get("TOCHeading") == "CAS":
                                return subsubsection["Information"][0]["Value"][
                                    "StringWithMarkup"
                                ][0]["String"]
    except KeyError:
        raise ValueError("Invalid molecule input, no Pubchem entry")

    raise ValueError("CAS number not found")


def smiles2name(smi, single_name=True):
    """This function queries the given molecule smiles and returns a name record or iupac"""

    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)
    except Exception:
        raise ValueError("Invalid SMILES string")
    # query the PubChem database
    r = requests.get(
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        + smi
        + "/synonyms/JSON"
    )
    # convert the response to a json object
    data = r.json()
    # return the SMILES string
    try:
        if single_name:
            index = 0
            names = data["InformationList"]["Information"][0]["Synonym"]
            while is_cas(name := names[index]):
                index += 1
                if index == len(names):
                    raise ValueError("No name found")
        else:
            name = data["InformationList"]["Information"][0]["Synonym"]
    except KeyError:
        raise ValueError("Unknown Molecule")
    return name


