"""2단계: chapters/, references/, metadata/chapters/를 ChromaDB의 3개 컬렉션으로 적재."""

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # langchain meta-package re-export 폴백
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import (
    CHAPTERS_DIR,
    CHAPTERS_METADATA_COLLECTION,
    CHAPTERS_TEXT_COLLECTION,
    CHROMA_DIR,
    METADATA_DIR,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    REFERENCE_COLLECTION,
    REFERENCES_DIR,
)
from .schemas import ActMetadata, ReferenceMetadata, Scene


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

_REFERENCE_KEYWORDS = ("원작", "스크립트", "레퍼")


# ---------- ChromaDB plumbing (lazy imports so dry-run works without chromadb installed) ----------

@lru_cache(maxsize=1)
def _embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


@lru_cache(maxsize=1)
def _client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _vectorstore(name: str):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=name,
        embedding_function=_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def _existing_count(name: str) -> int:
    try:
        return _client().get_collection(name).count()
    except Exception:
        return 0


def _drop_collection(name: str) -> None:
    try:
        _client().delete_collection(name)
    except Exception:
        pass


def _ingest(collection_name: str, docs: List[Document], force: bool) -> int:
    if not docs:
        print(f"[skip] {collection_name}: 적재할 문서 없음")
        return 0

    if force:
        _drop_collection(collection_name)
    else:
        existing = _existing_count(collection_name)
        if existing > 0:
            print(f"[skip] {collection_name}: 이미 {existing}개 존재 (--force로 재생성)")
            return 0

    vs = _vectorstore(collection_name)
    vs.add_documents(docs)
    print(f"[저장] {collection_name}: {len(docs)}개 문서")
    return len(docs)


def _make_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )


# ---------- chapters_text: walk toggle structure → chunked Documents ----------

@dataclass
class _Section:
    act: str            # 파일 stem (예: "act_01")
    section: str        # 상위 토글 제목
    scene: str          # 하위 토글 제목 (없으면 "")
    kind: str           # "author_text" | "reference_original"
    text: str


def _is_reference_original_title(title: str) -> bool:
    return any(kw in title for kw in _REFERENCE_KEYWORDS)


def _walk_act_sections(text: str, act_stem: str) -> List[_Section]:
    """Notion 토글 구조를 walk해서 leaf 섹션 단위로 분리."""
    lines = text.splitlines(keepends=True)

    top_idx = [i for i, ln in enumerate(lines) if ln.startswith("- ")]
    sub_idx = [i for i, ln in enumerate(lines) if ln.startswith("    - ")]

    if not top_idx:
        body = text.strip()
        return (
            [_Section(act=act_stem, section="(전체)", scene="", kind="author_text", text=body)]
            if body
            else []
        )

    sections: List[_Section] = []

    # 첫 토글 앞 서두는 별도 섹션으로 보존.
    if top_idx[0] > 0:
        prelude = "".join(lines[: top_idx[0]]).strip()
        if prelude:
            sections.append(
                _Section(act=act_stem, section="(서두)", scene="", kind="author_text", text=prelude)
            )

    for ti, t_start in enumerate(top_idx):
        t_end = top_idx[ti + 1] if ti + 1 < len(top_idx) else len(lines)
        top_title = lines[t_start].strip()[2:].strip() or "(제목없음)"
        sub_in_range = [si for si in sub_idx if t_start < si < t_end]

        if not sub_in_range:
            body = "".join(lines[t_start + 1 : t_end]).strip()
            if not body:
                continue
            kind = "reference_original" if _is_reference_original_title(top_title) else "author_text"
            sections.append(_Section(act=act_stem, section=top_title, scene="", kind=kind, text=body))
            continue

        # 서브토글 사이의 도입부도 부모 섹션으로 한 조각.
        intro = "".join(lines[t_start + 1 : sub_in_range[0]]).strip()
        if intro:
            kind = "reference_original" if _is_reference_original_title(top_title) else "author_text"
            sections.append(_Section(act=act_stem, section=top_title, scene="(도입)", kind=kind, text=intro))

        for si, s_start in enumerate(sub_in_range):
            s_end = sub_in_range[si + 1] if si + 1 < len(sub_in_range) else t_end
            sub_title = lines[s_start].strip()[2:].strip() or "(제목없음)"
            body = "".join(lines[s_start + 1 : s_end]).strip()
            if not body:
                continue
            kind = (
                "reference_original"
                if _is_reference_original_title(sub_title) or _is_reference_original_title(top_title)
                else "author_text"
            )
            sections.append(
                _Section(act=act_stem, section=top_title, scene=sub_title, kind=kind, text=body)
            )

    return sections


def _chunk_sections_to_docs(sections: List[_Section]) -> List[Document]:
    splitter = _make_text_splitter()
    docs: List[Document] = []
    for sec in sections:
        chunks = splitter.split_text(sec.text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "act": sec.act,
                        "section": sec.section,
                        "scene": sec.scene,
                        "kind": sec.kind,
                        "chunk_idx": i,
                        "chunk_total": len(chunks),
                    },
                )
            )
    return docs


def build_chapters_text_docs() -> List[Document]:
    docs: List[Document] = []
    for txt_path in sorted(CHAPTERS_DIR.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        sections = _walk_act_sections(text, txt_path.stem)
        docs.extend(_chunk_sections_to_docs(sections))
    return docs


# ---------- chapters_metadata: stage 1 JSON → searchable Documents ----------

def _format_scene_doc(scene: Scene, act_stem: str, act_title: str) -> Document:
    parts = [f"[Act: {act_title} / Scene: {scene.scene_id} {scene.scene_title}]"]
    parts.append(f"상태: {scene.status}")
    parts.append(f"요약: {scene.summary}")

    if scene.characters:
        parts.append("인물: " + "; ".join(f"{c.name}({c.description})" for c in scene.characters))

    if scene.emotional_states:
        parts.append(
            "감정: "
            + "; ".join(
                f"{e.character} — {e.change_from} → {e.state}" if e.change_from else f"{e.character}: {e.state}"
                for e in scene.emotional_states
            )
        )

    if scene.relationship_changes:
        parts.append(
            "관계 변화: "
            + "; ".join(f"{r.character_a}-{r.character_b}: {r.change}" for r in scene.relationship_changes)
        )

    if scene.foreshadowing:
        parts.append(
            "복선: "
            + "; ".join(
                f"[recovered ← {f.related_to}] {f.description}"
                if f.status == "recovered"
                else f"[planted] {f.description}"
                for f in scene.foreshadowing
            )
        )

    if scene.reference_originals:
        parts.append(
            "참고 원작: "
            + "; ".join(f"{r.title}: {r.summary}" for r in scene.reference_originals)
        )

    return Document(
        page_content="\n".join(parts),
        metadata={
            "act": act_stem,
            "doc_kind": "scene",
            "scene_id": scene.scene_id,
            "scene_title": scene.scene_title,
            "status": scene.status,
            "characters": ", ".join(c.name for c in scene.characters),
            "has_reference_original": bool(scene.reference_originals),
        },
    )


def _format_act_doc(act_md: ActMetadata, act_stem: str) -> Document:
    scene_list = " / ".join(f"{s.scene_id} {s.scene_title}" for s in act_md.scenes)
    text = "\n".join(
        [
            f"[Act: {act_md.act_title}]",
            f"전체 흐름: {act_md.overall_summary}",
            f"구성 장면: {scene_list}",
        ]
    )
    return Document(
        page_content=text,
        metadata={
            "act": act_stem,
            "doc_kind": "act_summary",
            "act_title": act_md.act_title,
            "scene_count": len(act_md.scenes),
        },
    )


def build_chapters_metadata_docs() -> List[Document]:
    chapters_meta_dir = METADATA_DIR / "chapters"
    if not chapters_meta_dir.exists():
        return []
    docs: List[Document] = []
    for json_path in sorted(chapters_meta_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        act_md = ActMetadata.model_validate(data)
        docs.append(_format_act_doc(act_md, json_path.stem))
        for scene in act_md.scenes:
            docs.append(_format_scene_doc(scene, json_path.stem, act_md.act_title))
    return docs


# ---------- reference: chunked text + file_type tag from stage 1 metadata ----------

def _resolve_file_type(stem: str) -> str:
    meta_path = METADATA_DIR / "references" / f"{stem}.json"
    if not meta_path.exists():
        return "unknown"
    try:
        ref_md = ReferenceMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        return ref_md.file_type
    except Exception:
        return "unknown"


def build_reference_docs() -> List[Document]:
    splitter = _make_text_splitter()
    docs: List[Document] = []
    for txt_path in sorted(REFERENCES_DIR.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        file_type = _resolve_file_type(txt_path.stem)
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": txt_path.stem,
                        "file_type": file_type,
                        "chunk_idx": i,
                        "chunk_total": len(chunks),
                    },
                )
            )
    return docs


# ---------- Public ingest entry points ----------

def ingest_chapters_text(force: bool) -> int:
    return _ingest(CHAPTERS_TEXT_COLLECTION, build_chapters_text_docs(), force)


def ingest_chapters_metadata(force: bool) -> int:
    return _ingest(CHAPTERS_METADATA_COLLECTION, build_chapters_metadata_docs(), force)


def ingest_references(force: bool) -> int:
    return _ingest(REFERENCE_COLLECTION, build_reference_docs(), force)
