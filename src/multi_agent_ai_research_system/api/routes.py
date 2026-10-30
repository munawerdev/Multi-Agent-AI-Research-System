import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ..agents import build_reader_agent, build_search_agent, critic_chain, writer_chain
from ..tools import scrape_url
from .schemas import (
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
    ScrapeRequest,
    ScrapeResponse,
    SearchRequest,
    SearchResponse,
)

router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _chunk_to_text(chunk) -> str:
    """Normalise a LangChain stream chunk to plain text."""
    text = chunk if isinstance(chunk, str) else getattr(chunk, "content", "")
    if isinstance(text, list):
        text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in text
        )
    return text or ""


async def _run_search(topic: str) -> str:
    agent = build_search_agent()
    result = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    f"Find recent, reliable and detailed information about: {topic}",
                )
            ]
        }
    )
    return result["messages"][-1].content


async def _run_reader(topic: str, search_results: str) -> str:
    agent = build_reader_agent()
    result = await agent.ainvoke(
        {
            "messages": [
                (
                    "user",
                    (
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{search_results[:800]}"
                    ),
                )
            ]
        }
    )
    return result["messages"][-1].content


# ── Routes ───────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model="openai/gpt-oss-20b", version="1.0.0")


@router.post("/search", response_model=SearchResponse)
async def search_only(req: SearchRequest):
    try:
        return SearchResponse(query=req.query, results=await _run_search(req.query))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_only(req: ScrapeRequest):
    try:
        content = await __import__("asyncio").to_thread(scrape_url.invoke, req.url)
        return ScrapeResponse(url=req.url, content=content)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    """Blocking full pipeline — easy for scripts and quick tests."""
    start = time.time()
    try:
        search_results = await _run_search(req.topic)
        scraped_content = await _run_reader(req.topic, search_results)
        combined = (
            f"SEARCH RESULTS:\n{search_results}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{scraped_content}"
        )
        report = await writer_chain.ainvoke({"topic": req.topic, "research": combined})
        feedback = await critic_chain.ainvoke({"report": report})
        return ResearchResponse(
            topic=req.topic,
            search_results=search_results,
            scraped_content=scraped_content,
            report=report,
            feedback=feedback,
            duration_seconds=round(time.time() - start, 2),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


async def _stream_pipeline(topic: str) -> AsyncIterator[dict]:
    """SSE event stream — this is what your mobile clients should consume."""
    start = time.time()
    try:
        yield {
            "event": "step",
            "data": json.dumps({"step": "search", "status": "started"}),
        }
        search_results = await _run_search(topic)
        yield {
            "event": "step",
            "data": json.dumps(
                {"step": "search", "status": "complete", "content": search_results}
            ),
        }

        yield {
            "event": "step",
            "data": json.dumps({"step": "reader", "status": "started"}),
        }
        scraped_content = await _run_reader(topic, search_results)
        yield {
            "event": "step",
            "data": json.dumps(
                {"step": "reader", "status": "complete", "content": scraped_content}
            ),
        }

        combined = (
            f"SEARCH RESULTS:\n{search_results}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{scraped_content}"
        )

        yield {
            "event": "step",
            "data": json.dumps({"step": "writer", "status": "started"}),
        }
        report_parts: list[str] = []
        async for chunk in writer_chain.astream({"topic": topic, "research": combined}):
            text = _chunk_to_text(chunk)
            if text:
                report_parts.append(text)
                yield {"event": "report_chunk", "data": json.dumps({"content": text})}
        report = "".join(report_parts)
        yield {
            "event": "step",
            "data": json.dumps({"step": "writer", "status": "complete"}),
        }

        yield {
            "event": "step",
            "data": json.dumps({"step": "critic", "status": "started"}),
        }
        critic_parts: list[str] = []
        async for chunk in critic_chain.astream({"report": report}):
            text = _chunk_to_text(chunk)
            if text:
                critic_parts.append(text)
                yield {"event": "critic_chunk", "data": json.dumps({"content": text})}
        feedback = "".join(critic_parts)
        yield {
            "event": "step",
            "data": json.dumps({"step": "critic", "status": "complete"}),
        }

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "topic": topic,
                    "search_results": search_results,
                    "scraped_content": scraped_content,
                    "report": report,
                    "feedback": feedback,
                    "duration_seconds": round(time.time() - start, 2),
                }
            ),
        }
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}


@router.post("/research/stream")
async def research_stream(req: ResearchRequest):
    return EventSourceResponse(_stream_pipeline(req.topic))
