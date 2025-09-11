# directory_reader.py
import io
import os
from glob import glob

import cv2
import numpy as np
import pytesseract
from tqdm import tqdm
from pypdf import PdfReader
from pdf2image import convert_from_bytes, convert_from_path  # both variants


class DirectoryReader:
    """
    Read JDs/resumes and extract text from:
      • text PDFs (via pypdf)
      • scanned PDFs (via Poppler + Tesseract OCR)
      • Streamlit UploadedFile (bytes) or file paths
    """

    def __init__(self, path_to_jds: str, path_to_resumes: str,
                 poppler_path: str | None = None,
                 tesseract_cmd: str | None = None):
        self.path_to_jds = path_to_jds
        self.path_to_resumes = path_to_resumes
        self.jd_data: dict[str, str] = {}
        self.resume_data: dict[str, str] = {}

        # Resolve Poppler/Tesseract from args or env
        self.poppler_path = poppler_path or os.getenv("POPPLER_PATH")
        self.tesseract_cmd = tesseract_cmd or os.getenv("TESSERACT_CMD")
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    # ---------------- JD ---------------- #
    def read_jd_files(self):
        file_list = glob(self.path_to_jds, recursive=True)
        for file in tqdm(file_list):
            with open(file, "r", encoding="utf-8") as f:
                data = f.read().strip().lower()
            job_name = os.path.basename(file).replace(".txt", "")
            self.jd_data[job_name] = data
        return self.jd_data

    # ------------- Helpers ------------- #
    @staticmethod
    def _to_bytes(file_or_path) -> bytes:
        """Accept path/UploadedFile/file-like and return raw bytes."""
        # Streamlit UploadedFile or any file-like
        if hasattr(file_or_path, "read"):
            pos = getattr(file_or_path, "tell", lambda: 0)()
            file_or_path.seek(0)
            data = file_or_path.read()
            try:
                file_or_path.seek(pos)
            except Exception:
                pass
            return data
        # raw bytes
        if isinstance(file_or_path, (bytes, bytearray)):
            return bytes(file_or_path)
        # filesystem path
        with open(file_or_path, "rb") as f:
            return f.read()

    @staticmethod
    def _extract_text_with_pypdf(pdf_bytes: bytes) -> str:
        """Fast text-layer extraction."""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            texts = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                texts.append(txt)
            return "\n".join(texts).strip().lower()
        except Exception:
            return ""

    def _extract_text_with_ocr(self, pdf_source) -> str:
        """
        OCR flow: PDF -> images (Poppler) -> Tesseract.
        pdf_source can be bytes or a file path.
        """
        try:
            if isinstance(pdf_source, (bytes, bytearray)):
                images = convert_from_bytes(pdf_source, poppler_path=self.poppler_path)
            else:
                images = convert_from_path(pdf_source, poppler_path=self.poppler_path)
        except Exception:
            return ""

        texts = []
        for img in images:
            img_np = np.array(img)
            img_np = self.deskew(img_np)
            txt = self.get_text_from_image(img_np)
            texts.append(txt)
        return "\n".join(texts).strip().lower()

    # ------------- Public API ------------- #
    def extract_text_from_pdf(self, file_or_path):
        """
        Try pypdf first; if empty → OCR fallback.
        Supports Streamlit UploadedFile, bytes, or path.
        """
        pdf_bytes = self._to_bytes(file_or_path)
        text = self._extract_text_with_pypdf(pdf_bytes)
        if len(text) > 1:
            return text
        return self._extract_text_with_ocr(pdf_bytes)

    def extract_text_from_image(self, file_or_path):
        """Direct OCR path (useful if you know it's image-only)."""
        return self._extract_text_with_ocr(file_or_path)

    def read_resume_files(self):
        """Bulk read from a directory pattern (e.g., 'resumes/**/*.pdf')."""
        file_list = glob(self.path_to_resumes, recursive=True)
        for file in tqdm(file_list):
            file_parts = os.path.normpath(file).split(os.sep)
            job_title = file_parts[-2].replace(" ", "_").lower()
            resume_name = os.path.basename(file_parts[-1]).replace("-", "_").lower().replace(".pdf", "")
            data = self.extract_text_from_pdf(file)
            self.resume_data[f"{job_title}_{resume_name}"] = data
        return self.resume_data

    # ------------- Image utils ------------- #
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(gray > 0))
        if coords.size == 0:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def get_text_from_image(image: np.ndarray) -> str:
        # Solid defaults for documents
        return pytesseract.image_to_string(image, config="--oem 3 --psm 6")
