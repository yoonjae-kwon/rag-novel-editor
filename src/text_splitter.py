"""Notion 토글 구조를 인식해 Act 텍스트를 LLM 호출 단위로 분할."""

from dataclasses import dataclass
from typing import List


@dataclass
class TextPart:
    title: str
    text: str

    def __len__(self) -> int:
        return len(self.text)


def split_act_text(text: str, max_chars: int) -> List[TextPart]:
    """Act 텍스트를 max_chars 이하의 파트로 분할.

    1) 컬럼0 토글('- 제목')과 헤더('## 제목')를 기준으로 1차 분할.
    2) max_chars를 초과하는 파트는 4-스페이스 들여쓰기 서브토글('    - 제목') 기준으로
       부모 헤더를 prepend한 채 그리디 그룹핑.
    3) 그래도 큰 파트는 줄/단어/문자 경계에서 강제 분할.
    """
    if len(text) <= max_chars:
        return [TextPart(title="(전체)", text=text)]

    parts = _split_by_top_sections(text)

    out: List[TextPart] = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
        else:
            out.extend(_split_by_sub_toggles(p, max_chars))

    final: List[TextPart] = []
    for p in out:
        if len(p) <= max_chars:
            final.append(p)
        else:
            final.extend(_hard_split(p, max_chars))

    return final


def _is_top_section(line: str) -> bool:
    # Notion 토글 ("- 제목"), 또는 마크다운 헤더 ("## 제목" / "# 제목").
    if line.startswith("- "):
        return True
    if line.startswith("## ") or line.startswith("# "):
        return True
    return False


def _section_title(line: str) -> str:
    stripped = line.strip()
    for prefix in ("- ", "## ", "# "):
        if stripped.startswith(prefix):
            return stripped[len(prefix):].strip() or "(제목없음)"
    return stripped or "(제목없음)"


def _split_by_top_sections(text: str) -> List[TextPart]:
    lines = text.splitlines(keepends=True)
    parts: List[TextPart] = []
    current_title = "(서두)"
    current_lines: List[str] = []

    for line in lines:
        if _is_top_section(line):
            if current_lines and "".join(current_lines).strip():
                parts.append(TextPart(title=current_title, text="".join(current_lines)))
            current_title = _section_title(line)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines and "".join(current_lines).strip():
        parts.append(TextPart(title=current_title, text="".join(current_lines)))

    if not parts:
        return [TextPart(title="(전체)", text=text)]

    # 첫 토글 앞의 짧은 서두는 다음 파트와 합쳐 컨텍스트 보존.
    if len(parts) >= 2 and parts[0].title == "(서두)" and len(parts[0].text.strip()) < 200:
        merged_text = parts[0].text + parts[1].text
        parts = [TextPart(title=parts[1].title, text=merged_text)] + parts[2:]

    return parts


def _split_by_sub_toggles(part: TextPart, max_chars: int) -> List[TextPart]:
    lines = part.text.splitlines(keepends=True)
    sub_idx = [i for i, ln in enumerate(lines) if ln.startswith("    - ")]

    if len(sub_idx) < 2:
        return [part]

    # 첫 서브토글 앞부분(부모 토글 헤더 + 서두)은 모든 서브청크에 prepend해서 컨텍스트 유지.
    header_prefix = "".join(lines[: sub_idx[0]])

    sections: List[tuple[str, str]] = []
    for i, idx in enumerate(sub_idx):
        end = sub_idx[i + 1] if i + 1 < len(sub_idx) else len(lines)
        title = lines[idx].strip()[2:].strip() or "(제목없음)"
        body = "".join(lines[idx:end])
        sections.append((title, body))

    sub_parts: List[TextPart] = []
    bucket_titles: List[str] = []
    bucket_body = ""

    for title, body in sections:
        candidate_total = len(header_prefix) + len(bucket_body) + len(body)
        if bucket_body and candidate_total > max_chars:
            sub_parts.append(_compose(part.title, bucket_titles, header_prefix, bucket_body))
            bucket_titles = [title]
            bucket_body = body
        else:
            bucket_titles.append(title)
            bucket_body += body

    if bucket_body:
        sub_parts.append(_compose(part.title, bucket_titles, header_prefix, bucket_body))

    return sub_parts or [part]


def _compose(parent_title: str, sub_titles: List[str], header_prefix: str, body: str) -> TextPart:
    title = f"{parent_title} / {' + '.join(sub_titles)}" if sub_titles else parent_title
    return TextPart(title=title, text=header_prefix + body)


def _hard_split(part: TextPart, max_chars: int) -> List[TextPart]:
    """줄→공백→문자 경계 순으로 max_chars 이하가 되도록 강제 분할."""
    text = part.text
    chunks: List[str] = []
    while len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = text.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        chunks.append(text[:cut])
        text = text[cut:]
    if text.strip():
        chunks.append(text)

    return [
        TextPart(title=f"{part.title} (분할 {i + 1}/{len(chunks)})", text=c)
        for i, c in enumerate(chunks)
    ]
