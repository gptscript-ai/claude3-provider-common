import json
import os

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from anthropic._types import NOT_GIVEN
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

debug = os.environ.get("GPTSCRIPT_DEBUG", "false") == "true"


def log(*args):
    if debug:
        print(*args)


async def list_models(client: AsyncAnthropic | AsyncAnthropicBedrock) -> JSONResponse:
    if type(client) == AsyncAnthropic:
        data = [{"id": "claude-3-opus-20240229", "name": "Anthropic Claude 3 Opus"},
                {"id": "claude-3-sonnet-20240229", "name": "Anthropic Claude 3 Sonnet"},
                {"id": "claude-3-haiku-20240307", "name": "Anthropic Claude 3 Haiku"}, ]
    else:
        data = [{"id": "anthropic.claude-3-opus-20240229-v1:0", "name": "AWS Bedrock Anthropic Claude 3 Opus"},
                {"id": "anthropic.claude-3-sonnet-20240229-v1:0", "name": "AWS Bedrock Anthropic Claude 3 Sonnet"},
                {"id": "anthropic.claude-3-haiku-20240307-v1:0", "name": "AWS Bedrock Anthropic Claude 3 Haiku"}, ]
    return JSONResponse(content={"data": data})


def map_tools(tools: list[dict]) -> list[dict]:
    anthropic_tools: list[dict] = []
    for tool in tools:
        anthropic_tool = {
            "name": tool["function"]["name"],
            "description": tool["function"].get("description", ""),
            "input_schema": tool["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools


def map_messages(messages: dict) -> tuple[str, list[dict]]:
    system: str = """
You are task oriented system.
You receive input from a user, process the input from the given instructions, and then output the result.
Your objective is to provide consistent and correct results.
You do not need to explain the steps taken, only provide the result to the given instructions.
You are referred to as a tool.
You don't move to the next step until you have a result.
    """
    mapped_messages: list[dict] = []

    for message in messages:
        if 'role' in message.keys() and (message["role"] in ["system"]):
            # @TODO: figure out if there is a better way to do this - tool-beta claude acts differently than regular claude.
            # If it sees the first user message as not being relevant to the conversation, it will complain.
            # system += message["content"] + "\n"
            mapped_messages.append({
                "role": "user",
                "content": [
                    {
                        "text": message["content"],
                        "type": "text"
                    },
                ]
            })

        if 'role' in message.keys() and message["role"] in "user":
            mapped_messages.append({
                "role": "user",
                "content": [
                    {
                        "text": message["content"],
                        "type": "text"
                    },
                ]
            })

        if 'role' in message.keys() and message["role"] == "tool":
            mapped_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": message["content"],
                    }
                ]
            })

        if 'role' in message.keys() and message["role"] == "assistant":
            if 'tool_calls' in message.keys():
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    tool_calls.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"]),
                    })
                mapped_messages.append({
                    "role": "assistant",
                    "content": tool_calls,
                })
            elif 'content' in message.keys() and message["content"] is not None:
                mapped_messages.append({
                    "role": "assistant",
                    "content": message["content"],
                })

    def prepend_if_unique(lst, new_dict, key, value):
        if not any(d.get(key) == value for d in lst):
            lst.insert(0, new_dict)

    prepend_if_unique(mapped_messages, {"role": "user", "content": "."}, "role", "user")
    if mapped_messages[0]["role"] != "user":
        mapped_messages.insert(0, {"role": "user", "content": "."})

    mapped_messages = merge_consecutive_dicts_with_same_value(mapped_messages, "role")

    return system, mapped_messages


def merge_consecutive_dicts_with_same_value(list_of_dicts, key) -> list[dict]:
    merged_list = []
    index = 0
    while index < len(list_of_dicts):
        current_dict = list_of_dicts[index]
        value_to_match = current_dict.get(key)
        compared_index = index + 1
        while compared_index < len(list_of_dicts) and list_of_dicts[compared_index].get(key) == value_to_match:
            log("CURRENT DICT: ", current_dict)
            log("COMPARED DICT: ", list_of_dicts[compared_index])
            list_of_dicts[compared_index]["content"] = current_dict["content"] + (list_of_dicts[compared_index][
                "content"])
            current_dict.update(list_of_dicts[compared_index])
            compared_index += 1
        merged_list.append(current_dict)
        index = compared_index
    return merged_list


async def completions(client: AsyncAnthropic | AsyncAnthropicBedrock, input: dict):
    log("ORIGINAL REQUEST: ", input)
    tools = input.get("tools", NOT_GIVEN)
    if tools is not NOT_GIVEN:
        tools = map_tools(tools)
        log("MAPPED TOOLS: ", tools)

    if input["model"].startswith("anthropic."):
        log('using bedrock client')
        client = AsyncAnthropicBedrock()
    else:
        client = AsyncAnthropic()

    system, messages = map_messages(input["messages"])

    max_tokens = input.get("max_tokens", 1024)
    if max_tokens is not None:
        max_tokens = int(max_tokens)

    temperature = input.get("temperature", NOT_GIVEN)
    if temperature is not None:
        temperature = float(temperature)

    stream = input.get("stream", False)

    top_k = input.get("top_k", NOT_GIVEN)
    if top_k is not NOT_GIVEN:
        top_k = int(top_k)

    top_p = input.get("top_p", NOT_GIVEN)
    if top_p is not NOT_GIVEN:
        top_p = float(top_p)

    try:
        response = await client.beta.tools.messages.create(
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            model=input["model"],
            temperature=temperature,
            tools=tools,
            top_k=top_k,
            top_p=top_p,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=e.__dict__["status_code"])

    log("RESPONSE FROM CLAUDE")
    log(response.model_dump_json())

    mapped_response = map_resp(response)

    log("MAPPED RESPONSE")
    log(mapped_response.model_dump_json())

    return StreamingResponse("data: " + mapped_response.model_dump_json() + "\n\n", media_type="application/x-ndjson")


def map_resp(response):
    parsed_tool_calls = []
    content: str | None = None

    for item in response.content:
        if response.stop_reason == "tool_use":
            if item.type == "tool_use":
                index = len(parsed_tool_calls)
                parsed_tool_calls.append({
                    "index": index,
                    "id": item.id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": json.dumps(item.input),
                    }
                })
                content = None
        else:
            if item.type == "text":
                content = item.text

    role = response.role
    finish_reason = map_finish_reason(response.stop_reason)

    resp = ChatCompletionChunk(
        id="0",
        choices=[
            Choice(
                delta=ChoiceDelta(
                    content=content,
                    tool_calls=parsed_tool_calls,
                    role=role
                ),
                finish_reason=finish_reason,
                index=0,
            )
        ],
        created=0,
        model="",
        object="chat.completion.chunk",
    )
    return resp


def map_finish_reason(finish_reason: str) -> str:
    if (finish_reason == "end_turn"):
        return "stop"
    elif (finish_reason == "stop_sequence"):
        return "stop"
    elif finish_reason == "max_tokens":
        return "length"
    elif finish_reason == "tool_use":
        return "tool_calls"
    return finish_reason
