import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
CHAPTERS_DIR = ROOT_DIR / "chapters"
REFERENCES_DIR = ROOT_DIR / "references"
METADATA_DIR = ROOT_DIR / "metadata"
CHROMA_DIR = ROOT_DIR / "chroma_db"

load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "novel_chunks")

# Act 텍스트가 이 길이를 넘으면 자동 분할 → 파트별 추출 → LLM 병합으로 처리.
MAX_PART_CHARS = int(os.getenv("MAX_PART_CHARS", "20000"))
