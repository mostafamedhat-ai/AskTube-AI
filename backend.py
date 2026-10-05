"""AskTube backend: question answering over YouTube transcripts (RAG).

Flow
----
1. ``POST /LOAD_VIDEO``  -> fetch the transcript, chunk it, summarize it with
   Gemini and index the chunks in FAISS.
2. ``POST /ASK_TUBE``    -> retrieve the most relevant chunks and answer the
   question with Gemini, citing timestamps.

Configuration (environment variables)
-------------------------------------
Required:
    API_KEY            Bearer token clients must send in the Authorization header.
    GOOGLE_API_KEY     Google AI (Gemini) API key.

Optional:
    GEMINI_MODEL       Chat model name            (default: gemini-3.8-flash)
    EMBEDDING_MODEL    Sentence-transformers model (default: paraphrase-multilingual-MiniLM-L12-v2)
    LLM_TIMEOUT        Per-request LLM timeout, seconds (default: 60)
    LLM_MAX_RETRIES    LLM retry count            (default: 2)
    HOST               Bind address               (default: 0.0.0.0)
    PORT               Bind port                  (default: 8000)
    LOG_LEVEL          Logging level              (default: INFO)

Running
-------
    python backend.py
    # or
    uvicorn backend:app --host 0.0.0.0 --port 8000

IMPORTANT: the active video lives in process memory, so run exactly ONE worker
(do not use ``--workers N`` with N > 1).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, AsyncIterator
from urllib.parse import parse_qs, urlparse

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger("asktube")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Application settings, loaded once from environment variables."""

    api_key: str
    google_api_key: str
    gemini_model: str = "gemini-3.8-flash"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    transcript_languages: tuple[str, ...] = ("en", "ar")
    chunk_target_snippets: int = 8
    chunk_overlap: int = 2
    summary_reduce_batch_size: int = 8
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        missing = [name for name in ("API_KEY", "GOOGLE_API_KEY") if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )
        defaults = cls.__dataclass_fields__
        return cls(
            api_key=os.environ["API_KEY"],
            google_api_key=os.environ["GOOGLE_API_KEY"],
            gemini_model=os.getenv("GEMINI_MODEL", defaults["gemini_model"].default),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", defaults["embedding_model"].default
            ),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            llm_max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------- #
# Domain errors
# --------------------------------------------------------------------------- #
class InvalidVideoURLError(ValueError):
    """The URL does not contain a recognizable YouTube video id."""


class EmptyTranscriptError(ValueError):
    """The transcript was fetched but contains no usable text."""


class TranscriptUnavailableError(RuntimeError):
    """No English/Arabic transcript could be fetched for the video."""


class NoActiveVideoError(RuntimeError):
    """A question was asked before any video was loaded."""


# --------------------------------------------------------------------------- #
# YouTube transcript
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TranscriptSnippet:
    text: str
    start: float
    duration: float


def extract_video_id(url: str) -> str:
    """Extract the YouTube video id from a URL.

    Supports ``youtu.be/<id>``, ``youtube.com/watch?v=<id>`` and the
    ``/shorts/``, ``/embed/``, ``/live/`` and ``/v/`` path forms.

    Raises:
        InvalidVideoURLError: if no video id can be found.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        query = parse_qs(parsed.query)
        if query.get("v"):
            return query["v"][0]

        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            return parts[1]

    raise InvalidVideoURLError(f"No video id found in URL: {url}")


def fetch_transcript(
    video_id: str, languages: tuple[str, ...]
) -> list[TranscriptSnippet]:
    """Fetch the timestamped transcript snippets for a video.

    Raises:
        TranscriptUnavailableError: if the transcript cannot be fetched.
        EmptyTranscriptError: if the transcript has no text.
    """
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except Exception as exc:
        logger.warning("Transcript fetch failed for %s: %s", video_id, exc)
        raise TranscriptUnavailableError(
            "Could not fetch an English or Arabic transcript for this video."
        ) from exc

    snippets = [
        TranscriptSnippet(
            text=item.text.replace("\n", " "),
            start=item.start,
            duration=item.duration,
        )
        for item in fetched
    ]
    if not any(s.text.strip() for s in snippets):
        raise EmptyTranscriptError("No transcript text was available for this video.")
    return snippets


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    text: str
    start: float
    end: float


_SENTENCE_END = (".", "?", "!", "؟", "！", "۔")


def _is_sentence_end(text: str) -> bool:
    return text.strip().endswith(_SENTENCE_END)


def create_smart_chunks(
    snippets: list[TranscriptSnippet],
    target_snippets: int = 8,
    overlap: int = 2,
) -> list[Chunk]:
    """Group snippets into overlapping chunks that end on sentence boundaries."""
    chunks: list[Chunk] = []
    i = 0

    while i < len(snippets):
        end_idx = min(i + target_snippets, len(snippets))

        # Extend the chunk until it ends on a sentence boundary.
        while end_idx < len(snippets) and not _is_sentence_end(snippets[end_idx - 1].text):
            end_idx += 1

        group = snippets[i:end_idx]
        if not group:
            break

        chunks.append(
            Chunk(
                text=" ".join(s.text for s in group),
                start=group[0].start,
                end=group[-1].start + group[-1].duration,
            )
        )
        i += max(1, len(group) - overlap)

    return chunks


# --------------------------------------------------------------------------- #
# LLM (Gemini)
# --------------------------------------------------------------------------- #
_SUMMARY_PROMPT = (
    "Summarize only the supplied video transcript. Keep the summary in "
    "the same language as the transcript. Preserve its main ideas and "
    "do not add facts that are not stated in the text."
)
_ANSWER_PROMPT = (
    "Answer the user's question using only the provided YouTube transcript. "
    "Reply in the same language as the question. If the transcript does not "
    "contain the answer, say so. Cite relevant timestamps in [start-end] format."
)


def _content_to_text(content: Any) -> str:
    """Normalize a LangChain message ``content`` (str or list of parts) to text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        text = "\n".join(p for p in parts if p).strip()
        if text:
            return text
    return str(content).strip()


class GeminiClient:
    """Thin wrapper around the chat model for summarization and answering."""

    def __init__(self, settings: Settings) -> None:
        self._reduce_batch_size = settings.summary_reduce_batch_size
        self._model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    def _invoke(self, system: str, human: str) -> str:
        response = self._model.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        return _content_to_text(response.content)

    def summarize_text(self, text: str) -> str:
        return self._invoke(_SUMMARY_PROMPT, text)

    def summarize_chunks(self, chunks: list[Chunk]) -> str:
        """Summarize each chunk, then reduce the summaries into one."""
        summaries = [self.summarize_text(c.text) for c in chunks if c.text.strip()]
        if not summaries:
            return "Not enough text was available to create a summary."

        size = self._reduce_batch_size
        while len(summaries) > 1:
            summaries = [
                batch[0]
                if len(batch) == 1
                else self.summarize_text("\n\n".join(batch))
                for batch in (
                    summaries[i : i + size] for i in range(0, len(summaries), size)
                )
            ]
        return summaries[0]

    def answer(self, question: str, excerpts: list[str]) -> str:
        context = "\n\n".join(excerpts)
        return self._invoke(
            _ANSWER_PROMPT,
            f"Transcript excerpts:\n{context}\n\nQuestion: {question}",
        )


# --------------------------------------------------------------------------- #
# Video service (indexing + retrieval)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActiveVideo:
    """Immutable pair so `video_id` and `store` can never get out of sync."""

    video_id: str
    store: FAISS


@dataclass(frozen=True)
class LoadResult:
    video_id: str
    summary: str
    chunk_count: int


class VideoService:
    """Holds the single active video index and exposes load / ask operations."""

    def __init__(
        self,
        settings: Settings,
        embeddings: HuggingFaceEmbeddings,
        llm: GeminiClient,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._llm = llm
        self._active: ActiveVideo | None = None
        # Loads are serialized because only one video index is kept at a time.
        self._load_lock = asyncio.Lock()

    @property
    def active_video_id(self) -> str | None:
        return self._active.video_id if self._active else None

    # -- loading ---------------------------------------------------------- #
    def _load_blocking(self, video_url: str) -> LoadResult:
        video_id = extract_video_id(video_url)
        snippets = fetch_transcript(video_id, self._settings.transcript_languages)

        chunks = create_smart_chunks(
            snippets,
            target_snippets=self._settings.chunk_target_snippets,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyTranscriptError(
                "Could not create searchable chunks from this transcript."
            )

        summary = self._llm.summarize_chunks(chunks)
        documents = [
            Document(
                page_content=c.text,
                metadata={"start": c.start, "end": c.end, "video_id": video_id},
            )
            for c in chunks
        ]
        store = FAISS.from_documents(documents, self._embeddings)

        # Publish only after every step succeeded, so a failed load keeps the
        # previously loaded video intact.
        self._active = ActiveVideo(video_id=video_id, store=store)
        logger.info("Loaded video %s (%d chunks)", video_id, len(chunks))
        return LoadResult(video_id=video_id, summary=summary, chunk_count=len(chunks))

    async def load(self, video_url: str) -> LoadResult:
        async with self._load_lock:
            return await run_in_threadpool(self._load_blocking, video_url)

    # -- asking ----------------------------------------------------------- #
    def _ask_blocking(self, active: ActiveVideo, question: str, top_k: int) -> str:
        results = active.store.similarity_search(question, k=top_k)
        if not results:
            return "I did not find any suitable clips to answer the question in this video."

        excerpts = [
            f"[{int(doc.metadata['start'])}s-{int(doc.metadata['end'])}s] {doc.page_content}"
            for doc in results
        ]
        return self._llm.answer(question, excerpts)

    async def ask(self, question: str, top_k: int) -> tuple[str, str]:
        """Return ``(answer, video_id)`` for the currently active video."""
        active = self._active  # single read -> consistent snapshot
        if active is None:
            raise NoActiveVideoError("Load a video before asking questions.")
        answer = await run_in_threadpool(self._ask_blocking, active, question, top_k)
        return answer, active.video_id


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LoadVideoRequest(BaseModel):
    video_url: Annotated[NonEmptyStr, StringConstraints(max_length=2048)]


class LoadVideoResponse(BaseModel):
    video_id: str
    summary: str
    chunk_count: int


class AskRequest(BaseModel):
    question: Annotated[NonEmptyStr, StringConstraints(max_length=4000)]
    top_k: Annotated[int, Field(strict=True, ge=1, le=10)] = 4


class AskResponse(BaseModel):
    response: str
    video_id: str


class HealthResponse(BaseModel):
    status: str
    video_id: str | None


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def get_service(request: Request) -> VideoService:
    return request.app.state.service


def require_api_key(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Validate the ``Authorization: Bearer <API_KEY>`` header (constant-time)."""
    expected = f"Bearer {request.app.state.settings.api_key}"
    if not authorization or not secrets.compare_digest(
        authorization.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        embeddings = await run_in_threadpool(
            HuggingFaceEmbeddings, model_name=settings.embedding_model
        )
        app.state.service = VideoService(settings, embeddings, GeminiClient(settings))
        logger.info("Startup complete")
        yield

    app = FastAPI(title="AskTube", lifespan=lifespan)
    app.state.settings = settings

    @app.get("/health", response_model=HealthResponse)
    async def health(service: VideoService = Depends(get_service)) -> HealthResponse:
        return HealthResponse(status="ok", video_id=service.active_video_id)

    @app.post(
        "/LOAD_VIDEO",
        response_model=LoadVideoResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def load_video(
        body: LoadVideoRequest,
        service: VideoService = Depends(get_service),
    ) -> LoadVideoResponse:
        try:
            result = await service.load(body.video_url)
        except (InvalidVideoURLError, EmptyTranscriptError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TranscriptUnavailableError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("/LOAD_VIDEO failed")
            raise HTTPException(
                status_code=500,
                detail=f"Could not process video ({type(exc).__name__}). Check server logs.",
            ) from exc

        return LoadVideoResponse(
            video_id=result.video_id,
            summary=result.summary,
            chunk_count=result.chunk_count,
        )

    @app.post(
        "/ASK_TUBE",
        response_model=AskResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def ask_tube(
        body: AskRequest,
        service: VideoService = Depends(get_service),
    ) -> AskResponse:
        try:
            answer, video_id = await service.ask(body.question, body.top_k)
        except NoActiveVideoError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("/ASK_TUBE failed")
            raise HTTPException(
                status_code=500,
                detail=f"Could not answer the question ({type(exc).__name__}). Check server logs.",
            ) from exc

        return AskResponse(response=answer, video_id=video_id)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=app.state.settings.host,
        port=app.state.settings.port,
        log_level=app.state.settings.log_level.lower(),
    )
