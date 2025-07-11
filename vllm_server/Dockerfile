FROM vllm/vllm-openai:latest

# 환경 변수 설정
ENV MODEL_NAME="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
ENV HOST="0.0.0.0"
ENV PORT=8000
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.9
ENV MAX_MODEL_LEN=2048

# vLLM 내장 API 서버 실행
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --host ${HOST} --port ${PORT} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --max-model-len ${MAX_MODEL_LEN}"]
