# ZunoSync

Multi-agent AI marketing co-pilot for social media content, media generation, scheduling, and knowledge-base assisted responses.

## Features

- Multi-agent orchestration flow for marketing tasks
- Chat API with thread-aware conversation handling
- Human-in-the-loop confirmation flow for ambiguous/media inputs
- Media upload + analysis support (image/video)
- PDF knowledge-base ingestion workflow
- Social account lookup and scheduling endpoints
- Redis-backed user context memory
- PostgreSQL/Supabase-compatible document storage flow

## Tech Stack

- Node.js (ESM)
- Express
- LangGraph / LangChain
- OpenAI models
- PostgreSQL (`pg`)
- Redis (`ioredis`)
- Supabase SDK
- AWS S3 SDK
- Multer for uploads

## Project Structure

```text
source/
  server.js                 # Main HTTP API
  src/
    index.js                # Core orchestration entry
    graph/workflow.js       # Agent workflow graph
    agents/                 # Agent implementations
    lib/                    # DB, Redis, ingestion, media helpers
    tools/                  # Tool-layer helpers
  scripts/
    run-schema.js           # Apply DB schema
    ingest-pdf.js           # CLI PDF ingest