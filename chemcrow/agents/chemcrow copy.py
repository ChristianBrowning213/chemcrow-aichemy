from typing import Optional

import langchain
from dotenv import load_dotenv
from langchain import PromptTemplate, chains
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import ValidationError
from rmrkl import ChatZeroShotAgent, RetryAgentExecutor

from .prompts import FORMAT_INSTRUCTIONS, QUESTION_PROMPT, REPHRASE_TEMPLATE, SUFFIX
from .tools import make_tools


# ======================================================================
# HARD PATCH: Fix legacy ChatOpenAI max_tokens / max_completion_tokens
# ======================================================================
# The goal of this patch:
#   - Make ChatOpenAI._default_params NOT include:
#       max_tokens=None or max_completion_tokens=None
#   - So that whatever we pass via _make_llm(...) is the only thing sent.
#   - Avoid "max_tokens is null" and "unsupported parameter max_tokens" errors.

from langchain.chat_models import openai as lc_openai

_old_default_params_attr = lc_openai.ChatOpenAI._default_params

# Handle both "property" and "plain callable" cases safely
if isinstance(_old_default_params_attr, property):
    _old_default_params_fget = _old_default_params_attr.fget
else:
    _old_default_params_fget = _old_default_params_attr


def _patched_default_params(self):
    """
    Wrap the original _default_params but:
      - Ensure we always return a dict
      - Strip any token-related keys so they don't send nulls
    """
    original = _old_default_params_fget(self)
    # Make sure we have a real dict to mutate
    params = dict(original) if isinstance(original, dict) else dict(original)

    # Remove any token keys that might be None by default
    params.pop("max_tokens", None)
    params.pop("max_completion_tokens", None)

    return params


# Reinstall as a @property so `self._default_params` returns a mapping
lc_openai.ChatOpenAI._default_params = property(_patched_default_params)


# ======================================================================
# Modern OpenAI LLM Wrapper (2025-compatible)
# ======================================================================

def _make_llm(model, temp, api_key, streaming: bool = False):
    """
    Unified handler for ALL modern OpenAI chat models.

    Behaviour:
      - For chat models:
          * gpt-3.5-*
          * gpt-4-*, gpt-4o, gpt-4.1-*, gpt-4.2-*
          * gpt-5-*, gpt-5.1-*
          * o1, o1-mini, o1-preview
        use ChatOpenAI and explicitly set:
          - max_tokens            for 3.x / 4.x models
          - max_completion_tokens for 5.x / o1 models

      - For legacy "text-*" models, use OpenAI (completion API) with no
        special token handling (old behaviour).
    """
    from langchain.chat_models import ChatOpenAI
    from langchain import OpenAI

    # Models that require max_completion_tokens instead of max_tokens
    models_using_completion_tokens = (
        "gpt-5",
        "gpt-5.1",
        "o1",
    )
    needs_completion_tokens = any(
        model.startswith(prefix) for prefix in models_using_completion_tokens
    )

    # Recognised chat model prefixes
    chat_prefixes = (
        "gpt-3.5",
        "gpt-4",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.2",
        "gpt-5",
        "gpt-5.1",
        "o1",
    )

    # --- Chat models (ChatOpenAI) ---
    if any(model.startswith(p) for p in chat_prefixes):
        base_kwargs = dict(
            temperature=temp,
            model_name=model,
            request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
        )

        # Only ONE of these is sent; _default_params no longer adds nulls.
        if needs_completion_tokens:
            base_kwargs["max_completion_tokens"] = 4096
        else:
            base_kwargs["max_tokens"] = 4096

        return ChatOpenAI(**base_kwargs)

    # --- Legacy text completion models (OpenAI) ---
    if model.startswith("text-"):
        # Old completion API still uses max_tokens internally; we don't
        # need to touch it here, leaving behaviour as in original ChemCrow.
        return OpenAI(
            temperature=temp,
            model_name=model,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
            openai_api_key=api_key,
        )

    # Anything else is considered invalid for this ChemCrow fork
    raise ValueError(f"Unsupported model: {model}")


# ======================================================================
# ChemCrow Agent Definition
# ======================================================================

class ChemCrow:
    def __init__(
        self,
        tools=None,
        model="gpt-4.1-mini",
        tools_model="gpt-4.1-mini",
        temp=0.1,
        max_iterations=40,
        verbose=True,
        streaming: bool = True,
        openai_api_key: Optional[str] = None,
        api_keys: dict = {},
        local_rxn: bool = False,
    ):
        """Initialize ChemCrow agent with modern OpenAI compatibility."""

        load_dotenv()

        try:
            self.llm = _make_llm(model, temp, openai_api_key, streaming)
        except ValidationError:
            raise ValueError("Invalid or missing OpenAI API key")

        # ------------------------------------------------------------------
        # Build tools if not provided
        # ------------------------------------------------------------------
        if tools is None:
            api_keys["OPENAI_API_KEY"] = openai_api_key
            tools_llm = _make_llm(tools_model, temp, openai_api_key, streaming)
            tools = make_tools(
                tools_llm,
                api_keys=api_keys,
                local_rxn=local_rxn,
                verbose=verbose,
            )

        # ------------------------------------------------------------------
        # Agent executor (Retry + Tool-using zero-shot agent)
        # ------------------------------------------------------------------
        self.agent_executor = RetryAgentExecutor.from_agent_and_tools(
            tools=tools,
            agent=ChatZeroShotAgent.from_llm_and_tools(
                self.llm,
                tools,
                suffix=SUFFIX,
                format_instructions=FORMAT_INSTRUCTIONS,
                question_prompt=QUESTION_PROMPT,
            ),
            verbose=True,
            max_iterations=max_iterations,
        )

        # Rephraser chain (used by original ChemCrow interface)
        rephrase = PromptTemplate(
            input_variables=["question", "agent_ans"],
            template=REPHRASE_TEMPLATE,
        )
        self.rephrase_chain = chains.LLMChain(prompt=rephrase, llm=self.llm)

    # ==================================================================
    # Agent RUN method
    # ==================================================================
    def run(self, prompt: str):
        """Executes the ChemCrow agent with the given prompt."""
        outputs = self.agent_executor({"input": prompt})
        return outputs["output"]
