# resume/src/resume_suggestions.py
from __future__ import annotations

import os
from typing import Dict, Optional

# Streamlit is optional here—used only to read secrets if available.
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

# Use your existing prompts/models (but NOT an API key) from constants.py
from constants import (
    OPENAI_MODEL_NAME,
    TEMPLATE_CONTENT,
    comparison_prompt,
    resume_analysis_prompt,
    job_description_analysis_prompt,
    gap_analysis_prompt,
    actionable_steps_prompt,
    experience_enhancement_prompt,
    additional_qualifications_prompt,
    resume_tailoring_prompt,
    relevant_skills_highlight_prompt,
    resume_formatting_prompt,
    resume_length_prompt,
)

# Load .env for local dev (harmless in hosted envs)
load_dotenv()


def _read_api_key() -> str:
    """Read the OpenAI key from Streamlit secrets (if present) or environment."""
    # Prefer Streamlit secrets if available
    if st is not None:
        try:
            v = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
            if v:
                return str(v)
        except Exception:
            pass
    # Fallback to OS/.env
    return os.getenv("OPENAI_API_KEY", "")


def _build_chain(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> ConversationChain:
    """Create a ConversationChain with short memory (k=2)."""
    key = (api_key or _read_api_key() or "").strip()
    if not key or not key.startswith("sk-"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing or invalid. Set it in .env, environment, or Streamlit secrets."
        )

    llm = ChatOpenAI(
        model=(model or OPENAI_MODEL_NAME),
        temperature=temperature,
        api_key=key,
    )

    system_message = SystemMessage(content=TEMPLATE_CONTENT)
    human_message = HumanMessagePromptTemplate.from_template("{history} User:{input} Assistant:")
    prompt_template = ChatPromptTemplate(messages=[system_message, human_message], validate_template=True)
    memory = ConversationBufferWindowMemory(k=2)

    return ConversationChain(llm=llm, prompt=prompt_template, memory=memory, verbose=False)


def _ensure_text(x: Optional[str]) -> str:
    return (x or "").strip()


def generate_response(prompt_input: str, chain: ConversationChain) -> str:
    """One-shot helper to keep parity with your earlier usage."""
    return chain.predict(input=prompt_input)


def generate_report(
    resume_text: Optional[str],
    jd_text: Optional[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, str]:
    """
    Builds the chain and runs all analyses using your existing prompt templates.
    Returns a dict of section -> content.
    """
    resume = _ensure_text(resume_text)
    jd = _ensure_text(jd_text)
    if not resume or not jd:
        raise ValueError("Both resume_text and jd_text must be provided (non-empty strings).")

    chain = _build_chain(api_key=api_key, model=model, temperature=temperature)

    results: Dict[str, str] = {}
    results["comparison_analysis"] = generate_response(comparison_prompt.format(resume, jd), chain)
    results["resume_analysis"] = generate_response(resume_analysis_prompt.format(resume), chain)
    results["job_description_analysis"] = generate_response(job_description_analysis_prompt.format(jd), chain)
    results["gap_analysis"] = generate_response(gap_analysis_prompt.format(resume, jd), chain)
    results["actionable_steps"] = generate_response(actionable_steps_prompt.format(resume, jd), chain)
    results["experience_enhancement"] = generate_response(experience_enhancement_prompt.format(resume, jd), chain)
    results["additional_qualifications"] = generate_response(additional_qualifications_prompt.format(resume, jd), chain)
    results["resume_tailoring"] = generate_response(resume_tailoring_prompt.format(resume, jd), chain)
    results["relevant_skills_highlight"] = generate_response(relevant_skills_highlight_prompt.format(resume, jd), chain)
    results["resume_formatting"] = generate_response(resume_formatting_prompt.format(resume, jd), chain)
    results["resume_length"] = generate_response(resume_length_prompt.format(resume, jd), chain)

    return results


def generate_report_text(
    resume_text: Optional[str],
    jd_text: Optional[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Convenience wrapper: returns a single formatted string suitable for display.
    """
    r = generate_report(
        resume_text=resume_text,
        jd_text=jd_text,
        api_key=api_key,
        model=model,
        temperature=temperature,
    )
    return (
        f"Comparison Analysis:\n{r['comparison_analysis']}\n\n"
        f"Resume Analysis:\n{r['resume_analysis']}\n\n"
        f"Job Description Analysis:\n{r['job_description_analysis']}\n\n"
        f"Gap Analysis:\n{r['gap_analysis']}\n\n"
        f"Actionable Steps:\n{r['actionable_steps']}\n\n"
        f"Experience Enhancement:\n{r['experience_enhancement']}\n\n"
        f"Additional Qualifications:\n{r['additional_qualifications']}\n\n"
        f"Resume Tailoring:\n{r['resume_tailoring']}\n\n"
        f"Relevant Skills Highlight:\n{r['relevant_skills_highlight']}\n\n"
        f"Resume Formatting:\n{r['resume_formatting']}\n\n"
        f"Resume Length:\n{r['resume_length']}"
    )
