import asyncio
import os
import sys
from typing import List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams


# Request/Response models
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


# Global LLM instance
llm = None


def serve_model():
    """Serve vLLM model as OpenAI-compatible HTTP server in the same process"""
    global llm

    model_name = os.environ["MODEL"] or "facebook/opt-125m"
    host = "0.0.0.0"
    port = 8000

    print(f"Loading model: {model_name}")
    llm = LLM(model=model_name)
    print("Model loaded successfully!")

    app = FastAPI(title="vLLM OpenAI-Compatible API")

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "vllm",
                }
            ],
        }

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest):
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        prompts = (
            [request.prompt] if isinstance(request.prompt, str) else request.prompt
        )
        outputs = llm.generate(prompts, sampling_params)

        choices = [
            CompletionChoice(text=output.outputs[0].text, index=i, finish_reason="stop")
            for i, output in enumerate(outputs)
        ]

        return CompletionResponse(
            id="cmpl-" + str(hash(str(prompts)))[:8],
            created=1677610602,
            model=request.model,
            choices=choices,
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        # Convert messages to prompt
        prompt = ""
        for message in request.messages:
            if message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        prompt += "Assistant:"

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        outputs = llm.generate([prompt], sampling_params)

        return ChatCompletionResponse(
            id="chatcmpl-" + str(hash(prompt))[:8],
            created=1677610602,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant", content=outputs[0].outputs[0].text
                    ),
                    finish_reason="stop",
                )
            ],
        )

    print(f"Starting server at: http://{host}:{port}")
    print("API endpoints:")
    print(f"  - Models: http://{host}:{port}/v1/models")
    print(f"  - Completions: http://{host}:{port}/v1/completions")
    print(f"  - Chat: http://{host}:{port}/v1/chat/completions")

    uvicorn.run(app, host=host, port=port)


def test_offline():
    """Test offline inference (original implementation)"""
    from vllm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if mode == "serve":
        serve_model()
    elif mode == "test":
        test_offline()
    else:
        print("Usage: python main.py [serve|test]")
        print("  serve: Start HTTP server (default)")
        print("  test: Run offline inference test")
