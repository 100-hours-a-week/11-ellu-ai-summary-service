
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.server.server import Server
from vllm.utils import random_uuid
import argparse
import asyncio
import json

async def main(args):
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    server = Server(
        engine=engine,
        served_model_names=[args.model],
        response_role="assistant",
        lora_modules=None,
        chat_template=None,
    )
    await server.serve(
        host=args.host,
        port=args.port,
        root_path=args.root_path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root-path", type=str, default=None)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(main(args))
