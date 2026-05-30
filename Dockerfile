# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile
# Echo-DSRN / Echo-Hybrid — vLLM OpenAI-compatible server on ROCm
#
# Build:
#   docker build -t echo-vllm:rocm .
#   # or pinning a specific vLLM version:
#   docker build \
#     --build-arg VLLM_ROCM_IMAGE=vllm/vllm-openai-rocm:v0.8.3 \
#     -t echo-vllm:rocm .
#
# See docker-compose.yml for serving both models simultaneously.
# ─────────────────────────────────────────────────────────────────────────────

ARG VLLM_ROCM_IMAGE=vllm/vllm-openai-rocm:latest
FROM ${VLLM_ROCM_IMAGE}

LABEL org.opencontainers.image.title="echo-dsrn-vllm-rocm" \
      org.opencontainers.image.description="Echo-DSRN/Hybrid vLLM server (ROCm)" \
      org.opencontainers.image.source="https://github.com/ethicalabs-ai/Echo-DSRN.git"
