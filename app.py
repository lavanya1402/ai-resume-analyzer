# app.py — AI Resume Analyzer (clean)
from __future__ import annotations
import io, os, re, shutil, unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")

from dotenv import load_dotenv
from PIL import Image
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ---------- Optional OCR deps (kept optional so cloud still works) ----------
try:
    from pdf2image import convert_from_bytes  # requires poppler for image-only PDFs
except Exception:
    convert_from_bytes = None  # type: ignore

try:
    import pytesseract as _pt  # requires tesseract for OCR
    pytesseract = _pt  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
# ---------------------------------------------------------------------------


# =============================== Config =====================================
ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=ROOT / ".env", override=True)  # local dev

def _has_local_secrets_file() -> bool:
    return any(p.exists() for p in [
        ROOT / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ])

def read_cfg(name: str, default: str = "") -> str:
    """Priority: .env/OS → st.secrets (only if secrets.toml exists) → default."""
    val = os.getenv(name)
    if val:
        return val
    if _has_local_secrets_file():  # avoids 'no secrets file' warnings locally
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass
    return default

OPENAI_API_KEY = read_cfg("OPENAI_API_KEY")
EMBED_MODEL    = read_cfg("OPENAI_MODEL_EMBED", "text-embedding-3-large")
GPT_MODEL      = read_cfg("OPENAI_MODEL_GPT",   "gpt-4o-mini")
TESSERACT_CMD  = read_cfg("TESSERACT_CMD")      # e.g. /usr/bin/tesseract
POPPLER_PATH   = read_cfg("POPPLER_PATH")       # e.g. /usr/bin

if pytesseract and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD  # type: ignore

client: OpenAI | None = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
# ============================================================================


# ============================== OCR checks ==================================
def ocr_runtime_available() -> Tuple[bool, str]:
    """Return (available, reason)."""
    has_pdf2image = convert_from_bytes is not None
    has_pytesseract = pytesseract is not None
    has_poppler = (shutil.which("pdftoppm") is not None) or bool(POPPLER_PATH)
    has_tesseract = (shutil.which("tesseract") is not None) or bool(TESSERACT_CMD)

    if not (has_pdf2image and has_pytesseract):
        return False, "Missing Python OCR packages (pdf2image/pytesseract)."
    if not has_poppler:
        return False, "Poppler binary not found (pdftoppm)."
    if not has_tesseract:
        return False, "Tesseract binary not found."
    return True, "OCR runtime available."
# ============================================================================


# ============================== Utilities ===================================
def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\x00", " ")
    return re.sub(r"[ \t]+", " ", text).strip()

def parse_pdf(pdf_bytes: bytes) -> str:
    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            t = (p.extract_text() or "").strip()
            if not t:  # try OCR for image-only page
                ok, why = ocr_runtime_available()
                if ok and convert_from_bytes and pytesseract:
                    try:
                        images = convert_from_bytes(
                            pdf_bytes,
                            first_page=p.page_number,
                            last_page=p.page_number,
                            poppler_path=POPPLER_PATH or None
                        )
                        if images:
                            t = pytesseract.image_to_string(images[0])  # type: ignore
                    except Exception:
                        pass  # silent OCR failure
                else:
                    st.info(f"OCR skipped on page {p.page_number}: {why}")
            parts.append(t)
    return _clean("\n".join(parts))

def parse_image(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes))
    if pytesseract and ocr_runtime_available()[0]:
        try:
            return _clean(pytesseract.image_to_string(img))  # type: ignore
        except Exception:
            return ""
    return ""

def parse_any(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(file_bytes)
    if any(name.endswith(e) for e in (".png", ".jpg", ".jpeg", ".webp", ".tiff")):
        text = parse_image(file_bytes)
        if not text:
            st.info("OCR skipped for image: OCR runtime not available on this server.")
        return text
    return _clean(file_bytes.decode("utf-8", errors="ignore"))

def embed(text: str) -> List[float]:
    if client is None:
        raise RuntimeError("OpenAI client not initialized (missing OPENAI_API_KEY).")
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def cos_sim(a, b) -> float:
    return float(cosine_similarity(np.array(a).reshape(1, -1),
                                   np.array(b).reshape(1, -1))[0][0])

def simple_skills(text: str) -> List[str]:
    keys = [
        "python","sql","machine learning","deep learning","nlp","azure","aws","gcp",
        "docker","kubernetes","linux","git","pandas","numpy","scikit-learn",
        "streamlit","fastapi","rest","mlops","llm","langchain","openai","prompt engineering",
        "rag","faiss","vector db","airflow","kafka"
    ]
    t = text.lower()
    return sorted({k for k in keys if k in t})

def suggest(resume: str, jd: str, score: float, skills: List[str]) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not initialized (missing OPENAI_API_KEY).")
    system = "You are a precise, practical resume coach. Be concise, use bullet points, give concrete examples."
    user = f"""Analyze how well this resume aligns with the JD.

RESUME:
---
{resume[:8000]}
---

JOB DESCRIPTION:
---
{jd[:8000]}
---

Context:
- Cosine similarity (0-1): {score:.3f}
- JD skills detected: {skills}

Tasks:
1) 2-line summary of overall fit.
2) Top 8 missing/weak areas mapped to JD bullets.
3) Rewrite 4 resume bullets with quantified impact verbs tailored to the JD.
4) 5-item action plan (courses/projects/metrics) for the next 2 weeks."""
    out = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.3,
        max_tokens=800,
    )
    return out.choices[0].message.content.strip()
# ============================================================================


# ================================= UI =======================================
st.title("🧠 AI Resume Analyzer (Streamlit)")
st.caption("Upload a Resume + JD • Embeddings + Cosine Similarity • GPT Suggestions")

with st.sidebar:
    st.markdown(f"**OpenAI key configured:** {'✅' if OPENAI_API_KEY else '❌'}")
    st.write("Embedding:", EMBED_MODEL)
    st.write("GPT:", GPT_MODEL)
    st.markdown("---")
    ok_ocr, why_ocr = ocr_runtime_available()
    if not ok_ocr:
        st.warning("OCR features limited: " + why_ocr)
    st.markdown("Troubleshooting: set **POPPLER_PATH** for image-only PDFs; set **TESSERACT_CMD** for OCR.")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY (set it in Streamlit → Settings → Secrets, or .env locally).")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    f_res = st.file_uploader("Upload Resume (PDF/Image/TXT)", type=["pdf","png","jpg","jpeg","webp","txt"])
with c2:
    f_jd = st.file_uploader("Upload Job Description (PDF/Image/TXT)", type=["pdf","png","jpg","jpeg","webp","txt"])

use_demo = st.button("Run Demo (Sample Texts)")

resume_txt = jd_txt = ""
if use_demo:
    resume_txt = (
        "Lavanya Srivastava — AI Developer & Trainer\n"
        "- Built agentic AI with LangChain, Streamlit, OpenAI.\n"
        "- Deployed on Azure; Docker + CI/CD.\n"
        "- Python, pandas, numpy, scikit-learn.\n"
        "- RAG with FAISS and vector DBs."
    )
    jd_txt = (
        "We seek a Generative AI Developer with Python, LangChain, LLMs,\n"
        "RAG (FAISS/Chroma), Azure deployment, and MLOps basics.\n"
        "Streamlit or FastAPI preferred. SQL and Docker required."
    )

if f_res is not None:
    resume_txt = parse_any(f_res.read(), f_res.name)
if f_jd is not None:
    jd_txt = parse_any(f_jd.read(), f_jd.name)

if resume_txt and jd_txt:
    with st.expander("Resume (parsed)"): st.text(resume_txt[:10000])
    with st.expander("Job Description (parsed)"): st.text(jd_txt[:10000])

    v_res, v_jd = embed(resume_txt), embed(jd_txt)
    score = cos_sim(v_res, v_jd)
    st.metric("Semantic Match (cosine)", f"{score:.3f}")

    jd_sk = simple_skills(jd_txt)
    rs_sk = simple_skills(resume_txt)
    missing = [s for s in jd_sk if s not in rs_sk]

    st.write("**JD skills detected:**", ", ".join(jd_sk) if jd_sk else "—")
    st.write("**Missing in resume:**", ", ".join(missing) if missing else "None 🎉")

    with st.spinner("Generating suggestions…"):
        tips = suggest(resume_txt, jd_txt, score, jd_sk)

    st.subheader("Personalized Suggestions")
    st.markdown(tips)

    report = (
        f"AI Resume Analyzer Report\n\n"
        f"Score: {score:.3f}\n"
        f"JD Skills: {', '.join(jd_sk) if jd_sk else '—'}\n"
        f"Missing: {', '.join(missing) if missing else 'None'}\n\n"
        f"Suggestions:\n{tips}\n"
    )
    st.download_button("📄 Download report.txt", report.encode("utf-8"),
                       file_name="resume_report.txt", mime="text/plain")
else:
    st.info("Upload both files or click **Run Demo (Sample Texts)**.")
