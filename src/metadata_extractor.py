from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_openai import ChatOpenAI

from .config import MAX_PART_CHARS, OPENAI_API_KEY, OPENAI_LLM_MODEL
from .schemas import ActMetadata, ReferenceMetadata
from .text_splitter import split_act_text


CHAPTER_SYSTEM_PROMPT = """당신은 장편 소설 편집자입니다. 주어진 텍스트는 한 부(Act) 또는 그 일부이며, 내부에 여러 장면이 포함되어 있습니다. 노션에서 복사된 형식이라 헤더 표기(##, —, 글머리 기호, 빈 줄 수 등)가 일관되지 않을 수 있습니다.

다음 원칙으로 구조화된 메타데이터를 추출하십시오.

1. 장면 구분
   - 텍스트 내에 '## 장면 X.X' 형태의 헤더가 있으면 그것을 기준으로 장면을 나누십시오.
   - 헤더가 없거나 일관되지 않으면 내용 흐름(시간/장소/시점 전환)으로 자연스러운 단위로 나누십시오.
   - scene_id는 헤더에 명시된 값을 그대로 사용하고, 명시가 없으면 'A.1', 'A.2' 식으로 부여하십시오.

2. 장면 상태(status)
   - complete: 묘사와 대사가 완결된 산문 형태.
   - synopsis: 글머리 기호, 짧은 메모, 장면 개요로만 채워진 형태.
   - 헤더에 '(완성)' / '(시놉시스)' 같은 표기가 있으면 우선 따르되, 실제 본문 형태와 어긋나면 본문 형태를 우선하십시오.

3. characters
   - 그 장면에 실제로 등장하거나 의미 있게 언급되는 인물만 포함.
   - description은 그 장면 안에서의 역할/상태 한 줄.

4. foreshadowing
   - planted: 의도적으로 심어진 단서로 보이는 묘사/대사/사물.
   - recovered: 앞 장면에서 심어진 단서를 회수·해명하는 경우. related_to에 어떤 단서를 회수하는지 적으십시오.
   - 단순 분위기 묘사는 복선이 아닙니다. 명확히 의미가 나중으로 연결되는 것만.

5. emotional_states
   - 인물의 그 장면 안 감정 상태. 장면 내에서 변화가 있다면 change_from에 이전 상태도 함께.

6. relationship_changes
   - 두 인물 간 거리감/신뢰/긴장이 이동한 경우만 포함. 단순 인사·안부는 제외.

7. summary
   - 그 장면의 핵심 사건만 2-4문장으로 압축. 분위기 묘사는 빼고 사건 중심.

8. 원작 스크립트 vs 작가 본문 구분 (중요)
   - 이 텍스트는 노션에서 복사된 토글 구조입니다. 작가 본인이 쓴 본문과, 게임 원작에서 가져온 '원작 스크립트' 토글이 같은 파일 안에 섞여 있을 수 있습니다.
   - 다음에 해당하면 원작 스크립트로 간주하십시오:
     • 토글 헤더에 '원작', '원작 스크립트', '스크립트', '레퍼' 같은 표지가 포함된 경우 (예: '원작 스크립트', '조물원 전(원작 스크립트 참고 레퍼)').
     • 내부 내용이 게임 시나리오·대본 형태로, 작가의 본문 문체와 명확히 다른 경우 (시스템 메시지·UI 표기·번역체 대사·캐릭터 이름 콜론 표기 등).
   - 작가의 메모/주석/시놉시스(예: '감정선', '메모', '초안', '블루퍼', 괄호 안 작가 자문)는 원작이 아닙니다 — 작가 본문의 일부로 간주하되 status는 synopsis로 분류하십시오.
   - 모든 메타데이터(characters, summary, foreshadowing, emotional_states, relationship_changes, status)는 오직 작가 본문에서만 추출하십시오. 원작 스크립트의 사건·인물·감정을 작가 본문 메타데이터에 섞지 마십시오.
   - 원작 스크립트는 가장 가까운 작가 본문 장면의 reference_originals 필드에 별도 태깅하십시오. title은 토글 제목을 그대로, summary는 1-3문장 요약 (원문 통째 인용 금지). 어떤 작가 본문 장면과 연결되는지 명확하지 않으면 직전 작가 장면에 붙이십시오.

주관적 해석을 자제하고, 텍스트에 근거 있는 것만 추출하십시오. 텍스트에 없는 정보는 추측하지 마십시오."""


REFERENCE_SYSTEM_PROMPT = """당신은 장편 소설 편집자의 어시스턴트입니다. 주어진 문서는 작가가 작성한 설정 자료입니다 — 시놉시스, 트리트먼트, 재료/아이디어 메모, 작가 노트 등.

다음 원칙으로 구조화된 메타데이터를 추출하십시오.

1. file_type
   - 문서의 유형을 다음 중 하나로 분류: synopsis / treatment / ingredients / notes / other.
   - 파일명도 단서로 활용하되, 실제 내용이 우선입니다.

2. summary
   - 문서 전체의 목적과 핵심 내용을 한 문단으로.

3. structure_notes
   - 전체 서사 구조에 대한 정보 (부 단위 흐름, 테마, 톤, 장르, 분량 등).
   - section 필드에 어느 부분에 해당하는지 명시 ('제 1부', '전체 테마', '결말' 등).

4. unused_ideas
   - 본문에 아직 반영되지 않은 아이디어/재료/미장면/모티프.
   - 작가가 '나중에 쓰려고' 적어둔 항목들.
   - intended_placement에 배치 의도가 명시되어 있으면 그대로 옮기고, 없으면 null.

5. character_profiles
   - 캐릭터의 백스토리, 성격, 관계 배경 등 본문 외 설정 정보.
   - 본문에 이미 드러난 행동·대사를 단순 재기술한 것은 제외.

6. other_notes
   - 위 카테고리에 들어가지 않는 작가 메모, 결정 사항, 미결정 질문, 자료조사 todo 등.

빈 카테고리는 빈 배열로 두십시오. 추측·창작은 금물입니다."""


MERGE_SYSTEM_PROMPT = """당신은 장편 소설 편집자입니다. 한 부(Act)의 메타데이터가 텍스트 길이 때문에 여러 파트로 나뉘어 추출되었습니다. 이를 하나의 통합 Act 메타데이터로 병합하십시오.

병합 원칙:

1. act_title
   - 부 전체를 가리키는 적절한 제목. 입력 파트들의 제목과 내용을 종합.

2. overall_summary
   - 모든 파트를 관통하는 부 전체의 흐름을 3-5문장으로.

3. scenes
   - 모든 파트의 장면을 입력 순서대로 통합.
   - 동일 scene_id가 다른 파트에 있으면 충돌하지 않도록 적절히 변경 (예: 'P1.A.1', 'P2.A.1').
   - 각 장면의 characters / emotional_states / relationship_changes / reference_originals / summary / status는 입력 그대로 유지하되, 명백한 표기 차이만 통일.

4. 인물 표기
   - 동일 인물의 표기 차이가 있으면 가장 일반적인 표기로 통일하되, 인물을 추가/제거하지 말 것.

5. 복선 연결 (중요)
   - 한 파트에서 'planted'로 표시된 복선이 다른 파트의 장면에서 회수된다면, 회수 장면의 foreshadowing을 status='recovered' + related_to에 어떤 단서를 회수하는지 명시하여 갱신.
   - 이미 적절히 표시된 항목은 그대로 둘 것. 회수가 분명하지 않은 planted는 그대로 두십시오.

6. 감정·관계 흐름
   - 인접 장면 사이에서 동일 인물의 감정 흐름이 자연스럽게 이어지면 change_from을 보강.
   - 추측·창작 금지. 입력에 근거 있는 것만 활용.

7. reference_originals
   - 각 장면에 붙어 있던 그대로 유지. 새로 만들거나 옮기지 말 것.

입력에 없는 정보를 추가하지 말고, 입력에 있는 정보를 누락하지 마십시오."""


@lru_cache(maxsize=1)
def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )


def _extract_act_single(text: str, filename: str | None) -> ActMetadata:
    chain = _llm().with_structured_output(ActMetadata)
    user_content = f"[파일명: {filename}]\n\n{text}" if filename else text
    return chain.invoke(
        [
            {"role": "system", "content": CHAPTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )


def merge_act_parts(parts: List[ActMetadata], filename: str | None = None) -> ActMetadata:
    chain = _llm().with_structured_output(ActMetadata)
    parts_json = "\n\n========\n\n".join(
        f"### PART {i} (제목: {p.act_title})\n{p.model_dump_json(indent=2)}"
        for i, p in enumerate(parts, 1)
    )
    user_content = f"[파일명: {filename}]\n\n{parts_json}" if filename else parts_json
    return chain.invoke(
        [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )


def extract_act_metadata(
    text: str,
    filename: str | None = None,
    *,
    parts_output_dir: Path | None = None,
) -> ActMetadata:
    """Act 메타데이터 추출. 길이 초과 시 자동 분할 → 파트별 추출 → LLM 병합."""
    if len(text) <= MAX_PART_CHARS:
        return _extract_act_single(text, filename)

    parts = split_act_text(text, max_chars=MAX_PART_CHARS)

    if len(parts) == 1:
        return _extract_act_single(text, filename)

    print(f"  분할: {len(parts)} 파트")

    stem = Path(filename).stem if filename else None
    if parts_output_dir and stem:
        parts_output_dir.mkdir(parents=True, exist_ok=True)
        for old in parts_output_dir.glob(f"{stem}_part_*.json"):
            old.unlink()

    part_metadatas: List[ActMetadata] = []
    for i, part in enumerate(parts, 1):
        print(
            f"  ├ 파트 {i}/{len(parts)} [{part.title}] ({len(part):,}자) ...",
            end=" ",
            flush=True,
        )
        sub_filename = (
            f"{filename or '?'} (파트 {i}/{len(parts)}: {part.title})"
        )
        part_md = _extract_act_single(part.text, sub_filename)
        part_metadatas.append(part_md)
        print(f"장면 {len(part_md.scenes)}개")

        if parts_output_dir and stem:
            part_path = parts_output_dir / f"{stem}_part_{i:02d}.json"
            part_path.write_text(part_md.model_dump_json(indent=2), encoding="utf-8")

    print("  └ 파트 통합 중 ...", end=" ", flush=True)
    merged = merge_act_parts(part_metadatas, filename=filename)
    print(f"장면 {len(merged.scenes)}개 (전체)")
    return merged


def extract_reference_metadata(text: str, filename: str | None = None) -> ReferenceMetadata:
    chain = _llm().with_structured_output(ReferenceMetadata)
    user_content = f"[파일명: {filename}]\n\n{text}" if filename else text
    return chain.invoke(
        [
            {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
