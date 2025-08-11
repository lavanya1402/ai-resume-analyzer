import os, io, re, unicodedata
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from dotenv import load_dotenv
# Load the .env that sits next to this file, even if CWD changes
load_dotenv(dotenv_path=Path(__file__).with_name('.env'), override=True)

# ---- Settings (.env optional) ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required for OpenAI
EMBED_MODEL    = os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-large")
GPT_MODEL      = os.getenv("OPENAI_MODEL_GPT", "gpt-4o-mini")
TESSERACT_CMD  = os.getenv("TESSERACT_CMD")   # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_PATH   = os.getenv("POPPLER_PATH")    # e.g. C:\...\poppler-xx\Library\bin

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Helpers ----
def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def parse_pdf(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            t = (p.extract_text() or "").strip()
            if not t:
                # OCR this page if blank text layer — use Poppler path if provided
                try:
                    images = convert_from_bytes(
                        pdf_bytes,
                        first_page=p.page_number,
                        last_page=p.page_number,
                        poppler_path=POPPLER_PATH  # <— key addition
                    )
                    if images:
                        t = pytesseract.image_to_string(images[0])
                except Exception:
                    # Poppler not found or conversion failed; leave t as ""
                    pass
            text_parts.append(t)
    return _clean("\n".join(text_parts))

def parse_image(img_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes))
    return _clean(pytesseract.image_to_string(img))

def parse_any(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(file_bytes)
    if any(name.endswith(e) for e in [".png", ".jpg", ".jpeg", ".webp", ".tiff"]):
        return parse_image(file_bytes)
    return _clean(file_bytes.decode("utf-8", errors="ignore"))

def embed(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def cos_sim(a, b) -> float:
    return float(cosine_similarity(np.array(a).reshape(1, -1), np.array(b).reshape(1, -1))[0][0])

def simple_skills(text: str):
    keys = [
        "python","sql","machine learning","deep learning","nlp","azure","aws","gcp",
        "docker","kubernetes","linux","git","pandas","numpy","scikit-learn",
        "streamlit","fastapi","rest","mlops","llm","langchain","openai","prompt engineering",
        "rag","faiss","vector db","airflow","kafka"
    ]
    t = text.lower()
    return sorted({k for k in keys if k in t})

def suggest(resume: str, jd: str, score: float, skills: list[str]) -> str:
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
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.3,
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()

# ---- UI ----
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 AI Resume Analyzer (Streamlit)")
st.caption("Upload a Resume + JD • Embeddings + Cosine Similarity • GPT Suggestions")

with st.sidebar:
    st.markdown("**OpenAI key configured:** " + ("✅" if OPENAI_API_KEY else "❌"))
    st.write("Embedding:", EMBED_MODEL)
    st.write("GPT:", GPT_MODEL)
    st.markdown("---")
    st.markdown("Troubleshooting: set **POPPLER_PATH** in .env for image-only PDFs; set **TESSERACT_CMD** for OCR.")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .env")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    f_res = st.file_uploader("Upload Resume (PDF/Image/TXT)", type=["pdf","png","jpg","jpeg","webp","txt"])
with c2:
    f_jd = st.file_uploader("Upload Job Description (PDF/Image/TXT)", type=["pdf","png","jpg","jpeg","webp","txt"])

use_demo = st.button("Run Demo (Sample Texts)")

resume_txt = jd_txt = ""
if use_demo:
    resume_txt = """Lavanya Srivastava — AI Developer & Trainer
- Built agentic AI with LangChain, Streamlit, OpenAI.
- Deployed on Azure; Docker + CI/CD.
- Python, pandas, numpy, scikit-learn.
- RAG with FAISS and vector DBs."""
    jd_txt = """We seek a Generative AI Developer with Python, LangChain, LLMs,
RAG (FAISS/Chroma), Azure deployment, and MLOps basics.
Streamlit or FastAPI preferred. SQL and Docker required."""
if f_res is not None:
    resume_txt = parse_any(f_res.read(), f_res.name)
if f_jd is not None:
    jd_txt = parse_any(f_jd.read(), f_jd.name)

if resume_txt and jd_txt:
    with st.expander("Resume (parsed)"):
        st.text(resume_txt[:10000])
    with st.expander("Job Description (parsed)"):
        st.text(jd_txt[:10000])

    v_res = embed(resume_txt)
    v_jd = embed(jd_txt)
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

    report = f"""AI Resume Analyzer Report

Score: {score:.3f}
JD Skills: {', '.join(jd_sk) if jd_sk else '—'}
Missing: {', '.join(missing) if missing else 'None'}

Suggestions:
{tips}
"""
    st.download_button("📄 Download report.txt",
                       data=report.encode("utf-8"),
                       file_name="resume_report.txt",
                       mime="text/plain")
else:
    st.info("Upload both files or click **Run Demo (Sample Texts)**.")
