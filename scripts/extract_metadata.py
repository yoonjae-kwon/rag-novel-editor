"""1단계: chapters/와 references/에서 LLM으로 메타데이터를 추출하여 metadata/에 JSON으로 저장.

큰 chapters는 토글 구조 기반 자동 분할 → 파트별 추출 → LLM 병합으로 처리.
파트별 중간 결과는 metadata/chapters/parts/에 저장 (디버깅용).
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CHAPTERS_DIR, METADATA_DIR, REFERENCES_DIR, ROOT_DIR
from src.metadata_extractor import extract_act_metadata, extract_reference_metadata


def _read_text(txt_path: Path) -> str | None:
    text = txt_path.read_text(encoding="utf-8").strip()
    return text or None


def process_chapters(force: bool) -> tuple[int, int, int]:
    processed = skipped = failed = 0
    if not CHAPTERS_DIR.exists():
        print(f"[skip] {CHAPTERS_DIR} 없음")
        return processed, skipped, failed

    out_dir = METADATA_DIR / "chapters"
    parts_dir = out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(CHAPTERS_DIR.glob("*.txt"))
    if not txt_files:
        print("[skip] chapters/ 에 .txt 파일 없음")
        return processed, skipped, failed

    for txt_path in txt_files:
        out_path = out_dir / f"{txt_path.stem}.json"

        if out_path.exists() and not force:
            print(f"[skip] {txt_path.name} (이미 존재 — --force로 재생성)")
            skipped += 1
            continue

        text = _read_text(txt_path)
        if text is None:
            print(f"[skip] {txt_path.name} (빈 파일)")
            skipped += 1
            continue

        print(f"[처리 중] {txt_path.name} ({len(text):,}자)")
        try:
            result = extract_act_metadata(
                text,
                filename=txt_path.name,
                parts_output_dir=parts_dir,
            )
            out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"  → {out_path.relative_to(ROOT_DIR)}")
            processed += 1
        except Exception as e:
            print(f"  실패: {e}")
            traceback.print_exc()
            failed += 1

    return processed, skipped, failed


def process_references(force: bool) -> tuple[int, int, int]:
    processed = skipped = failed = 0
    if not REFERENCES_DIR.exists():
        print(f"[skip] {REFERENCES_DIR} 없음")
        return processed, skipped, failed

    out_dir = METADATA_DIR / "references"
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(REFERENCES_DIR.glob("*.txt"))
    if not txt_files:
        print("[skip] references/ 에 .txt 파일 없음")
        return processed, skipped, failed

    for txt_path in txt_files:
        out_path = out_dir / f"{txt_path.stem}.json"

        if out_path.exists() and not force:
            print(f"[skip] {txt_path.name} (이미 존재 — --force로 재생성)")
            skipped += 1
            continue

        text = _read_text(txt_path)
        if text is None:
            print(f"[skip] {txt_path.name} (빈 파일)")
            skipped += 1
            continue

        print(f"[처리 중] {txt_path.name} ({len(text):,}자) ...", end=" ", flush=True)
        try:
            result = extract_reference_metadata(text, filename=txt_path.name)
            out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"→ {out_path.relative_to(ROOT_DIR)}")
            processed += 1
        except Exception as e:
            print(f"실패: {e}")
            traceback.print_exc()
            failed += 1

    return processed, skipped, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="장편 소설 RAG 1단계: 본문(chapters)과 설정 자료(references)에서 메타데이터 추출"
    )
    parser.add_argument("--chapters-only", action="store_true", help="chapters/만 처리")
    parser.add_argument("--references-only", action="store_true", help="references/만 처리")
    parser.add_argument("--force", action="store_true", help="이미 존재하는 메타데이터도 재생성")
    args = parser.parse_args()

    if args.chapters_only and args.references_only:
        parser.error("--chapters-only 와 --references-only 는 동시에 사용할 수 없습니다.")

    do_chapters = not args.references_only
    do_references = not args.chapters_only

    totals = [0, 0, 0]

    if do_chapters:
        print("\n=== chapters/ 메타데이터 추출 ===")
        p, s, f = process_chapters(args.force)
        totals = [totals[0] + p, totals[1] + s, totals[2] + f]

    if do_references:
        print("\n=== references/ 메타데이터 추출 ===")
        p, s, f = process_references(args.force)
        totals = [totals[0] + p, totals[1] + s, totals[2] + f]

    print(f"\n완료: 처리 {totals[0]} / 스킵 {totals[1]} / 실패 {totals[2]}")
    return 1 if totals[2] else 0


if __name__ == "__main__":
    sys.exit(main())
