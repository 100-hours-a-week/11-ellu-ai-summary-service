FROM vllm/vllm-openai:latest

# 환경 변수 설정 (본인의 Hugging Face 모델로 변경)
ENV MODEL_NAME="idealm99/Looper-gemma-3-model"
ENV HOST="0.0.0.0"
ENV PORT=8000

# vLLM이 자동으로 모델을 다운로드하고 로드함
CMD ["sh", "-c", "python -m vllm.entrypoints.openai.api_server --model ${MODEL_NAME} --host ${HOST} --port ${PORT}"]