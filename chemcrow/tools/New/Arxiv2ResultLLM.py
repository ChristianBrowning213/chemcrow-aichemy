import os
import re
import time
import requests
import xml.etree.ElementTree as ET

import langchain
import paperqa
from langchain.base_language import BaseLanguageModel
from langchain.tools import BaseTool
from langchain.embeddings.openai import OpenAIEmbeddings
from pypdf.errors import PdfReadError

# ------------ arXiv API config ------------ #

ARXIV_API_URL = "https://export.arxiv.org/api/query"

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "ChemCrow-ArxivTool/0.1 "
            "(https://example.org; mailto:you@example.org)"
        )
    }
)

_LAST_API_CALL = 0.0
_API_MIN_INTERVAL = 3.0  # seconds

# Error prefix so the agent can detect tool failures
ARXIV_ERROR_PREFIX = "[ARXIV_TOOL_ERROR]"

def _safe_dir_name(name: str, max_len: int = 64) -> str:
    """
    Turn an arbitrary string into a filesystem-safe directory name.

    - Removes leading/trailing quotes.
    - Replaces any character not in [A-Za-z0-9._-] with '_'.
    - Collapses multiple '_' in a row.
    - Truncates to max_len characters.
    - Falls back to 'default' if empty.
    """
    # Strip common wrapping quotes
    name = name.strip().strip('"').strip("'")

    # Replace any disallowed character with '_'
    safe = []
    for ch in name:
        if (
            "A" <= ch <= "Z"
            or "a" <= ch <= "z"
            or "0" <= ch <= "9"
            or ch in "._-"
        ):
            safe.append(ch)
        else:
            safe.append("_")

    safe_str = "".join(safe)

    # Collapse multiple underscores
    collapsed = []
    prev = ""
    for ch in safe_str:
        if ch == "_" and prev == "_":
            continue
        collapsed.append(ch)
        prev = ch

    safe_str = "".join(collapsed)

    # Truncate to max_len
    if len(safe_str) > max_len:
        safe_str = safe_str[:max_len]

    # Fallback if everything was stripped
    if not safe_str:
        safe_str = "default"

    return safe_str


def _fail(msg: str) -> str:
    """
    Return a compact, structured error string for the agent to detect.
    """
    # Keep this message short so it does not bloat context.
    return f"{ARXIV_ERROR_PREFIX} {msg}"


# ------------ ArXiv helper functions (raw API) ------------ #

def _throttled_arxiv_get(params: dict) -> requests.Response:
    """
    Call the arXiv export API with a simple 3s throttle.
    Raises requests.RequestException on network errors.
    """
    global _LAST_API_CALL
    now = time.time()
    elapsed = now - _LAST_API_CALL
    if elapsed < _API_MIN_INTERVAL:
        time.sleep(_API_MIN_INTERVAL - elapsed)
    _LAST_API_CALL = time.time()

    resp = SESSION.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def _arxiv_api_query(
    search: str,
    max_results: int = 20,
) -> list[dict]:
    """
    Call the arXiv export API directly and parse results.

    Returns a list of dicts with keys:
      'id', 'title', 'authors', 'summary', 'published', 'pdf_url'

    On any network or parsing error, returns an empty list.
    """
    params = {
        "search_query": f"all:{search}",
        "start": 0,
        "max_results": max_results,
    }

    try:
        resp = _throttled_arxiv_get(params)
    except requests.RequestException:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    entries: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        id_el = entry.find("atom:id", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)

        title = (title_el.text or "").strip() if title_el is not None else ""
        entry_id = (id_el.text or "").strip() if id_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""

        authors_list = []
        for a in entry.findall("atom:author", ns):
            name_el = a.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors_list.append(name_el.text.strip())
        authors = ", ".join(authors_list)

        pdf_url = ""
        if "/abs/" in entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"

        entries.append(
            {
                "id": entry_id,
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
                "pdf_url": pdf_url,
            }
        )

    return entries


def arxiv_scraper(
    search: str,
    pdir: str = "arxiv_query",
    max_results: int = 20,
) -> dict:
    """
    Use the raw arXiv API to find up to max_results papers matching `search`,
    download their PDFs, and return a mapping:

        path -> {"citation": <citation_string>}

    On total failure, returns an empty dict.
    """
    if not os.path.isdir(pdir):
        os.makedirs(pdir, exist_ok=True)

    try:
        results = _arxiv_api_query(search, max_results=max_results)
    except Exception:
        return {}

    papers: dict = {}

    for r in results:
        if not r.get("pdf_url"):
            continue

        try:
            arxiv_id = r["id"].split("/")[-1] if r["id"] else "arxiv"
            filename = f"{arxiv_id}.pdf"
            path = os.path.join(pdir, filename)

            pdf_resp = SESSION.get(r["pdf_url"], timeout=30)
            pdf_resp.raise_for_status()

            with open(path, "wb") as f:
                f.write(pdf_resp.content)

            citation = (
                f"Title: {r['title']}\n"
                f"Authors: {r['authors']}\n"
                f"Published: {r['published']}\n"
                f"ArXiv ID: {arxiv_id}\n"
                f"URL: {r['id']}\n\n"
                f"Abstract: {r['summary']}"
            )

            papers[path] = {"citation": citation}
        except Exception:
            continue

    return papers


# ------------ LLM wrapper functions (ChemCrow-style) ------------ #
def arxiv_paper_search(llm, query, max_results=20):
    """
    Use an LLM to compress the user query to a short arXiv search string,
    then run an arXiv search and download the PDFs.

    On any failure in query compression, returns an empty dict.
    """
    prompt = langchain.prompts.PromptTemplate(
        input_variables=["question"],
        template=(
            "I would like to find scholarly papers to answer "
            "this question: {question}. Your response must be at "
            "most 10 words long.\n"
            "A search query that would bring up papers that can answer "
            "this question would be: "
        ),
    )

    query_chain = langchain.chains.llm.LLMChain(llm=llm, prompt=prompt)

    base_dir = "arxiv_query"
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    try:
        search = query_chain.run(query).strip()
    except Exception:
        return {}

    if not search:
        return {}

    print("\nArXiv search:", search)

    # Make a filesystem-safe subdirectory name
    subdir = _safe_dir_name(search)
    search_dir = os.path.join(base_dir, subdir)
    if not os.path.isdir(search_dir):
        try:
            os.mkdir(search_dir)
        except OSError:
            # If directory creation fails, bail out cleanly
            return {}

    papers = arxiv_scraper(search, pdir=search_dir, max_results=max_results)
    return papers


def arxiv2result_llm(
    llm,
    query,
    k: int = 5,
    max_sources: int = 2,
    openai_api_key: str = None,
    max_results: int = 20,
):
    """
    Failure-aware ArXiv-based QA:
    - Use LLM to generate a focused search query
    - Search arXiv and download PDFs
    - Use paperqa to answer the question from those PDFs

    On any failure, returns a short error string with ARXIV_ERROR_PREFIX.
    """
    try:
        papers = arxiv_paper_search(llm, query, max_results=max_results)
    except Exception as e:
        return _fail(
            f"Failed during arXiv search step: {type(e).__name__}: {e}"
        )

    if not papers:
        return _fail("No usable arXiv papers found for this query.")

    try:
        docs = paperqa.Docs(
            llm=llm,
            summary_llm=llm,
            embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key),
        )
    except Exception as e:
        return _fail(
            f"Failed to initialise paperqa Docs: {type(e).__name__}: {e}"
        )

    not_loaded = 0
    loaded = 0

    for path, data in papers.items():
        try:
            docs.add(path, data["citation"])
            loaded += 1
        except (ValueError, FileNotFoundError, PdfReadError):
            not_loaded += 1
        except Exception:
            not_loaded += 1

    if loaded == 0:
        return _fail(
            "Downloaded arXiv PDFs but could not load any of them. "
            "Try a broader or different query."
        )

    if not_loaded > 0:
        print(
            f"\nFound {len(papers)} arXiv papers but could not load {not_loaded}."
        )

    try:
        answer = docs.query(query, k=k, max_sources=max_sources).formatted_answer
    except Exception as e:
        return _fail(
            f"Failed while querying loaded papers: {type(e).__name__}: {e}"
        )

    return answer



# ------------ ChemCrow tool wrapper ------------ #

class Arxiv2ResultLLM(BaseTool):
    """
    ChemCrow tool for answering technical questions using primary literature
    from arxiv.org (physics, CS, maths, etc.).

    Pipeline:
      1. Compresses the user question to a short search string (≤ 10 words)
         using the LLM itself.
      2. Calls the arXiv export API with 'all:<search>' and downloads up to
         'max_results' PDFs into a query-specific folder.
      3. Builds a paperqa.Docs index using OpenAI embeddings and the same LLM.
      4. Asks paperqa to answer the original question using those papers.

    Why this is useful:
      - It gives you grounded answers that quote/aggregate actual papers, rather
        than pure parametric LLM knowledge.
      - Great for questions like "what architectures work best for COF property
        prediction?" or "recent progress in motif-based crystal graphs".

    Search behaviour / limitations:
      - The arXiv API search here is quite literal: it primarily matches your
        compressed query against the TITLE, ABSTRACT and CATEGORY fields.
      - That means your question MUST contain the right technical keywords and
        acronyms (e.g. 'COF gas adsorption', 'weighted automata sequence model',
        'MOF Bayesian optimisation') or you will get weak/no results.
      - Very broad, informal or conversational queries often fail; in that case,
        rephrase the question to mention specific methods, materials or tasks.
    """

    name = "ArxivLiteratureSearch"
    description = (
        "Search arxiv.org for relevant papers and answer a technical question by "
        "actually reading those papers. Best used for physics/CS/math/chemistry "
        "topics where arXiv has good coverage. The tool first condenses your "
        "question into a short keyword query, then calls the arXiv API with "
        "'all:<query>', downloads the top PDFs, and uses paperqa + embeddings to "
        "synthesise an answer. "
        "Important: the search is SIMPLE and keyword-driven — it works best when "
        "your question names concrete concepts (e.g. 'COF gas adsorption MCMC', "
        "'graph neural networks for crystal structures', 'Bayesian optimisation "
        "for materials discovery'). Vague questions without the right terms may "
        "return no useful papers. The tool returns a natural-language answer "
        "grounded in the retrieved literature, or a short error string starting "
        "with '[ARXIV_TOOL_ERROR]' if something fails."
    )

    
    llm: BaseLanguageModel = None
    openai_api_key: str = None
    max_results: int = 20

    def __init__(
        self,
        llm: BaseLanguageModel,
        openai_api_key: str = None,
        max_results: int = 20,
    ):
        super().__init__()
        self.llm = llm
        self.openai_api_key = openai_api_key
        self.max_results = max_results

    def _run(self, query: str) -> str:
        try:
            return arxiv2result_llm(
                self.llm,
                query,
                openai_api_key=self.openai_api_key,
                max_results=self.max_results,
            )
        except Exception as e:
            return _fail(f"Unexpected tool-level error: {type(e).__name__}")

    async def _arun(self, query: str) -> str:
        """
        Async version is not implemented for this tool.
        """
        raise NotImplementedError("this tool does not support async")
