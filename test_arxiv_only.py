# test_arxiv_only.py
"""
Minimal end-to-end test script that tells ChemCrow to exercise ONLY the
ArxivLiteratureSearch tool (your Arxiv2ResultLLM wrapper).

Assumes:
  - clean_chemcrow_minimal.py defines:
        build_clean_chemcrow
  - build_clean_chemcrow registers a tool with name 'ArxivLiteratureSearch'
    implemented by chemcrow.tools.New.Arxiv2ResultLLM.Arxiv2ResultLLM.
"""

import os

from clean_chemcrow_minimal import build_clean_chemcrow


def main():
    # Make sure OpenAI key is visible (Arxiv2ResultLLM will call OpenAI)
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY is not set. ArxivLiteratureSearch may fail.\n")

    chem_model = build_clean_chemcrow()

    # Try to introspect tools, but DO NOT abort if we fail
    tools_attr = getattr(chem_model, "tools", None)

    print("\n=== TOOLS SEEN BY ChemCrow (chem_model.tools if present) ===")
    if tools_attr is None:
        print("  (chem_model has no 'tools' attribute exposed; continuing anyway.)")
        tool_names = set()
    else:
        tool_names = {t.name for t in tools_attr}
        if not tool_names:
            print("  (No tools found on chem_model.tools; continuing anyway.)")
        else:
            for n in sorted(tool_names):
                print(f"  - {n}")

    if "ArxivLiteratureSearch" not in tool_names:
        print(
            "\nNOTE: ArxivLiteratureSearch not found via chem_model.tools, "
            "but build_clean_chemcrow printed it as loaded. "
            "Proceeding with the Arxiv-only test regardless.\n"
        )

    # Build the natural-language prompt for the agent
    test_prompt = """
You are wired into a ChemCrow environment that includes at least the following tools:

  - Python_REPL
  - Wikipedia
  - ArxivLiteratureSearch

Other chemistry tools may be present, but for THIS TASK you must rely
EXCLUSIVELY on **ArxivLiteratureSearch** for external scientific information.
Do not use Wikipedia or your own training data to answer the scientific parts;
you may only use your own reasoning to summarise and synthesise what
ArxivLiteratureSearch returns.

Your job is to call **ArxivLiteratureSearch** at least three times, to answer
the following research questions:

1) COF motif mining:
   \"\"\"
   What are current machine-learning approaches for motif or building-block
   discovery in COFs and MOFs, especially methods that operate directly on
   crystal structures or local environments (for example graph neural networks,
   unsupervised motif discovery, or local-environment clustering)?
   \"\"\"

2) Agentic AI + materials:
   \"\"\"
   What recent Arxiv work discusses agentic or autonomous AI systems applied
   to materials discovery, crystal structure search, or high-throughput
   computational screening?
   \"\"\"

3) Fun control query:
   \"\"\"
   Find recent Arxiv papers that describe interesting biological or behavioural
   properties of mammals, preferably in a quantitative or modelling context.
   \"\"\"

For each question:

  - Call **ArxivLiteratureSearch** with a query that you think will retrieve
    a useful set of papers.
  - If necessary, you may call the tool multiple times per question with
    different queries; just keep the total number of calls reasonable.
  - Based ONLY on the outputs of ArxivLiteratureSearch, write a short,
    technically accurate summary (5–10 sentences) answering the question.
  - Explicitly name 3–5 key papers that you found for that question, and
    briefly state why each paper is relevant (one sentence per paper).

At the end, provide a brief overall synthesis (5–8 sentences) comparing what
you learned across the three questions:
  - How mature is motif-based analysis in porous materials?
  - How far along is agentic or autonomous AI for materials discovery?
  - Where do you see obvious research gaps that could be interesting to
    explore next?

Remember:
  - You MUST use the ArxivLiteratureSearch tool for all literature facts.
  - Do not fabricate paper titles or claims that are not supported by the
    Arxiv tool's output.
"""

    print("\n=== TEST PROMPT (Arxiv-only system) ===\n")
    print(test_prompt)
    print("\n=== MODEL RESPONSE ===\n")

    answer = chem_model.run(test_prompt)
    print(answer)


if __name__ == "__main__":
    main()
