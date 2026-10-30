from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

app = FastAPI(
    title="ResearchMind API",
    description="Multi-agent AI research system — search, scrape, write, critique.",
    version="1.0.0",
)

# Mobile apps don't enforce CORS, but web clients (Flutter Web, RN Web) do.
# Tighten `allow_origins` for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def main():
    import uvicorn

    uvicorn.run(
        "multi_agent_ai_research_system.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
