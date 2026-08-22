"""
serve_embed.py
──────────────
Minimal OpenAI-compatible embedding server for Echo-DSRN sentence-transformers
models (e.g. ``ethicalabs/Echo-DSRN-v0.1.3-Embed-Exp``).

Uses the sentence-transformers path (modules.json + 1_Pooling) — the canonical
way to encode through the Echo embedding models; vLLM's pooling runner does not
support the custom Echo architecture.

Endpoints
─────────
    GET  /health
    POST /v1/embeddings   {"model": ..., "input": [...]}  → OpenAI-style response
    POST /embed           {"input": [...]}                → {"embeddings": [...]}
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL", "ethicalabs/Echo-DSRN-v0.1.3-Embed-Exp")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

app = FastAPI(title="echo-embed", version="0.1.0")

_model: SentenceTransformer | None = None


class EmbedRequest(BaseModel):
    model: str | None = None
    input: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_ID, trust_remote_code=True, device=DEVICE)
    return _model


@app.on_event("startup")
def _warmup() -> None:
    m = get_model()
    m.encode(["warmup"], batch_size=1, convert_to_numpy=True)
    print(f"echo-embed ready: {MODEL_ID} on {DEVICE}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest) -> dict:
    vecs = get_model().encode(req.input, batch_size=BATCH_SIZE, convert_to_numpy=True)
    return {
        "object": "list",
        "model": req.model or MODEL_ID,
        "data": [
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(vecs)
        ],
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    vecs = get_model().encode(req.input, batch_size=BATCH_SIZE, convert_to_numpy=True)
    return EmbedResponse(embeddings=[vec.tolist() for vec in vecs])
