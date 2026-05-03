"""2단계: chapters/, references/, metadata/chapters/를 ChromaDB의 3개 컬렉션으로 적재."""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import (
    build_chapters_metadata_docs,
    build_chapters_text_docs,
    build_reference_docs,
    ingest_chapters_metadata,
    ingest_chapters_text,
    ingest_references,
)


COLLECTIONS = ("chapters_text", "chapters_metadata", "reference")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="장편 소설 RAG 2단계: 텍스트와 메타데이터를 ChromaDB에 적재"
    )
    parser.add_argument(
        "--collection",
        choices=COLLECTIONS + ("all",),
        default="all",
        help="적재할 컬렉션 (기본: all)",
    )
    parser.add_argument("--force", action="store_true", help="이미 적재된 컬렉션도 재생성")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="임베딩/저장 없이 청크 생성만 하고 결과 통계만 출력",
    )
    args = parser.parse_args()

    if args.dry_run:
        return _run_dry(args.collection)

    try:
        if args.collection in ("chapters_text", "all"):
            print("\n=== chapters_text ===")
            ingest_chapters_text(args.force)
        if args.collection in ("chapters_metadata", "all"):
            print("\n=== chapters_metadata ===")
            ingest_chapters_metadata(args.force)
        if args.collection in ("reference", "all"):
            print("\n=== reference ===")
            ingest_references(args.force)
    except Exception as e:
        print(f"실패: {e}")
        traceback.print_exc()
        return 1

    return 0


def _run_dry(target: str) -> int:
    plans = []
    if target in ("chapters_text", "all"):
        plans.append(("chapters_text", build_chapters_text_docs))
    if target in ("chapters_metadata", "all"):
        plans.append(("chapters_metadata", build_chapters_metadata_docs))
    if target in ("reference", "all"):
        plans.append(("reference", build_reference_docs))

    for name, builder in plans:
        print(f"\n=== {name} (dry-run) ===")
        try:
            docs = builder()
        except Exception as e:
            print(f"  실패: {e}")
            traceback.print_exc()
            continue
        print(f"  문서 {len(docs)}개")
        if docs:
            sizes = [len(d.page_content) for d in docs]
            print(f"  크기: min={min(sizes)}자 / median={sorted(sizes)[len(sizes)//2]}자 / max={max(sizes)}자")
            keys = sorted({k for d in docs for k in d.metadata.keys()})
            print(f"  메타데이터 필드: {keys}")
            print(f"  샘플 메타데이터: {docs[0].metadata}")
            print(f"  샘플 본문 (첫 청크):\n    {docs[0].page_content[:200]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
