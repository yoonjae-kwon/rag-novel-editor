from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------- Chapter (Act / Scene) ----------

class Character(BaseModel):
    name: str = Field(description="장면에 등장하거나 의미 있게 언급되는 인물의 이름.")
    description: str = Field(description="이 장면에서의 역할/상태에 대한 한 줄 묘사.")


class EmotionalState(BaseModel):
    character: str
    state: str = Field(description="이 장면에서 인물의 감정 상태.")
    change_from: Optional[str] = Field(
        default=None,
        description="장면 내에서 감정이 이동했다면 그 이전 상태. 변화 없으면 null.",
    )


class RelationshipChange(BaseModel):
    character_a: str
    character_b: str
    change: str = Field(description="두 인물 사이의 거리감/신뢰/긴장이 어떻게 이동했는지.")


class Foreshadow(BaseModel):
    description: str = Field(description="복선의 내용 — 무엇을 암시하거나 회수하는지.")
    status: Literal["planted", "recovered"] = Field(
        description="planted=새로 심어진 단서, recovered=앞 장면 단서가 회수/해명된 경우."
    )
    related_to: Optional[str] = Field(
        default=None,
        description="recovered인 경우, 어떤 앞선 단서를 회수하는지. planted면 null.",
    )


class ReferenceOriginal(BaseModel):
    title: str = Field(
        description="원작 스크립트 토글의 제목을 그대로 (예: '원작 스크립트', '조물원 전(원작 스크립트 참고 레퍼)')."
    )
    summary: str = Field(
        description="원작 스크립트 내용의 핵심을 1-3문장으로 요약. 원문 전체 인용은 금지."
    )


class Scene(BaseModel):
    scene_id: str = Field(description="텍스트 내 헤더에 표시된 ID(예: '1.1'). 없으면 'A.1', 'A.2' 식으로 부여.")
    scene_title: str = Field(description="장면 제목. 헤더가 없으면 내용 기반으로 짧게 부여.")
    status: Literal["complete", "synopsis"] = Field(
        description="complete=묘사·대사가 완결된 산문, synopsis=글머리 기호/짧은 메모/장면 개요. 작가 본문 기준으로 판단."
    )
    summary: str = Field(description="이 장면의 핵심 사건을 2-4문장으로 압축. 작가 본문 기준.")
    characters: List[Character] = Field(default_factory=list)
    emotional_states: List[EmotionalState] = Field(default_factory=list)
    relationship_changes: List[RelationshipChange] = Field(default_factory=list)
    foreshadowing: List[Foreshadow] = Field(default_factory=list)
    reference_originals: List[ReferenceOriginal] = Field(
        default_factory=list,
        description="이 장면과 연결되는 원작 스크립트 토글들. 작가 본문이 아니라 게임 원작에서 가져온 참고 자료.",
    )


class ActMetadata(BaseModel):
    act_title: str = Field(description="부의 제목 (예: '제 1부 — 발견').")
    overall_summary: str = Field(description="이 부 전체의 흐름을 3-5문장으로 요약.")
    scenes: List[Scene]


# ---------- Reference (synopsis / treatment / ingredients / notes / other) ----------

class CharacterProfile(BaseModel):
    name: str
    description: str = Field(description="캐릭터의 백스토리, 성격, 관계 배경 등 본문 외 설정 정보.")


class UnusedIdea(BaseModel):
    title: str = Field(description="아이디어/재료의 짧은 이름.")
    description: str
    intended_placement: Optional[str] = Field(
        default=None,
        description="작가가 어디에 넣으려 하는지 명시되어 있다면 (예: '2부 어딘가'). 없으면 null.",
    )


class StructureNote(BaseModel):
    section: str = Field(description="어느 부분에 해당하는지 (예: '제 2부', '전체 테마', '톤').")
    content: str


class ReferenceMetadata(BaseModel):
    file_type: Literal["synopsis", "treatment", "ingredients", "notes", "other"] = Field(
        description="문서 유형 추론. 명확하지 않으면 'other'."
    )
    summary: str = Field(description="문서의 목적과 핵심 내용을 한 문단으로 요약.")
    structure_notes: List[StructureNote] = Field(default_factory=list)
    unused_ideas: List[UnusedIdea] = Field(default_factory=list)
    character_profiles: List[CharacterProfile] = Field(default_factory=list)
    other_notes: List[str] = Field(
        default_factory=list,
        description="위 카테고리에 들어가지 않는 작가 메모 / 결정 사항 / 미결정 질문 / todo.",
    )
