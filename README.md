# rag-novel-editor

장편 소설용 RAG 기반 AI 편집자 시스템.

## 구조

```
rag-novel-editor/
├── chapters/         # 장면별 원문 txt (입력)
├── chroma_db/        # ChromaDB 영속 저장소 (자동 생성, gitignored)
├── src/              # 핵심 모듈
│   ├── config.py     # 환경 변수 / 경로 설정
│   └── ...           # ingest / retriever / editor (이후 추가)
├── scripts/          # 진입점 CLI (이후 추가)
├── requirements.txt
└── .env              # API 키 (gitignored)
```

## 파이프라인

1. **메타데이터 추출** — `chapters/*.txt`를 LLM으로 분석해 등장인물, 핵심 사건, 복선, 감정 상태, 관계 변화를 구조화 추출
2. **인덱싱** — 원문 청크 + 메타데이터를 ChromaDB에 임베딩 저장
3. **편집자 응답** — 사용자 질문 → 벡터 검색 → 편집자 페르소나 LLM 응답

## 시작하기

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env  # 그리고 .env에 OPENAI_API_KEY 채우기
```
