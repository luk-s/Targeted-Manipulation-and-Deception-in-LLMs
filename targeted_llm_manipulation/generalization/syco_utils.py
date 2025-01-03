import asyncio
import json
import os

from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from tqdm import tqdm

Model = ChatAnthropic | ChatOpenAI  # type: ignore


def load_from_jsonl(file_name: str) -> list[dict]:
    def load_json_line(line: str, i: int, file_name: str):
        try:
            return json.loads(line)
        except:
            raise ValueError(f"Error in line {i+1}\n{line} of {file_name}")

    with open(file_name, "r") as f:
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
    return data


def get_model(model_name: str, temperature: float, max_tokens: int) -> Model:  # type: ignore
    if "claude" in model_name:
        if "ANTHROPIC_API_KEY" not in os.environ:
            os.environ["ANTHROPIC_API_KEY"] = input("Anthropic API key: ")
        return ChatAnthropic(model_name=model_name, temperature=temperature, max_tokens=max_tokens)  # type: ignore
    if "gpt" in model_name:
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = input("OpenAI API key: ")
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)  # type: ignore
    raise ValueError(f"{model_name} not recognized")


def to_messages(prompt: list[dict]) -> list[BaseMessage]:
    return [
        HumanMessage(content=d["content"]) if d["type"] == "human" else AIMessage(content=d["content"]) for d in prompt
    ]


def inference(
    model_name: str, prompts: list[list[dict]], temperature: float, max_tokens: int, stop: str | None = None
) -> list[str]:
    model = get_model(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    responses = [model.predict_messages(to_messages(prompt), stop=stop).content for prompt in tqdm(prompts)]
    return responses


async def async_inference(
    model_name: str,
    prompts: list[list[dict]],
    temperature: float,
    max_tokens: int,
    stop: str | None = None,
    max_async: int = 1,
) -> list[str]:
    model = get_model(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    semaphore = asyncio.Semaphore(max_async)
    tqdm_bar = tqdm(total=len(prompts))

    async def apredict_messages(prompt: list[dict]) -> str:
        async with semaphore:
            response = await model.apredict_messages(to_messages(prompt), stop=stop)
            tqdm_bar.update()
            return response.content

    tasks = [apredict_messages(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
