from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class ScrapeRequest(BaseModel):
    url: str = Field(..., min_length=10, max_length=2000)


class ResearchResponse(BaseModel):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    duration_seconds: float


class SearchResponse(BaseModel):
    query: str
    results: str


class ScrapeResponse(BaseModel):
    url: str
    content: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
