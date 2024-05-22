import os
import re

import xmltodict
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from fastapi.responses import JSONResponse, StreamingResponse

from prompt_constructors import *

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


def map_req(req: dict) -> dict:
    system: str | None = ""
    mapped_messages: list[dict] = []

    max_tokens = 4096
    if 'max_tokens' in req.keys():
        max_tokens = req["max_tokens"]

    if "tools" in req.keys():
        system += construct_tool_use_system_prompt(req["tools"])

    messages = req["messages"]

    tool_inputs_xml: list[str] = []
    tool_outputs_xml: list[str] = []
    for message in messages:
        if 'role' in message.keys() and message["role"] == "system":
            system += message["content"] + "\n"

        if 'role' in message.keys() and message["role"] == "user":
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
            content: str = '\n' + construct_tool_outputs_message([message], None)
            mapped_messages.append({
                "role": "user",
                "content": content,
            })

        if 'role' in message.keys() and message["role"] == "assistant":
            if 'tool_calls' in message.keys():
                tool_inputs = []
                for tool_call in message["tool_calls"]:
                    tool_inputs.append({
                        "tool_name": tool_call["function"]["name"],
                        "tool_arguments": tool_call["function"]["arguments"],
                    })
                content: str = '\n' + construct_tool_inputs_message(message.get("content", ""), tool_inputs)
                mapped_messages.append({
                    "role": "assistant",
                    "content": content,
                })
            else:
                mapped_messages.append({
                    "role": "assistant",
                    "content": message["content"]
                })

    for message in mapped_messages:
        if 'role' in message.keys() and message["role"] == "":
            message["role"] = "assistant"

    mapped_messages = merge_consecutive_dicts_with_same_value(mapped_messages, "role")

    mapped_req = {
        "messages": mapped_messages,
        "max_tokens": max_tokens,
        "system": system,
        "model": req["model"],
        "temperature": req["temperature"],
    }
    return mapped_req


def merge_consecutive_dicts_with_same_value(list_of_dicts, key) -> list[dict]:
    merged_list = []
    index = 0
    while index < len(list_of_dicts):
        current_dict = list_of_dicts[index]
        value_to_match = current_dict.get(key)
        compared_index = index + 1
        while compared_index < len(list_of_dicts) and list_of_dicts[compared_index].get(key) == value_to_match:
            list_of_dicts[compared_index]["content"] = current_dict["content"] + (list_of_dicts[compared_index][
                "content"])
            current_dict.update(list_of_dicts[compared_index])
            compared_index += 1
        merged_list.append(current_dict)
        index = compared_index
    return merged_list


def map_resp(response) -> str:
    data = json.loads(response)
    finish_reason = None
    parsed_tool_calls = []

    log("INITIAL RESPONSE DATA: ", data)

    for message in data["content"]:
        if 'text' in message.keys() and "<function_calls>" in message["text"]:
            pattern = re.compile(r'(<function_calls>.*?</invoke>)', re.DOTALL)
            match = pattern.search(message["text"])
            xml_tool_calls = match.group(1) + "</function_calls>"
            tool_calls = xmltodict.parse(xml_tool_calls)
            if tool_calls["function_calls"]["invoke"] is list:
                for key, value in tool_calls["function_calls"]["invoke"].items():
                    parsed_tool_calls.append({
                        "index": 0,
                        "id": value['tool_name'],
                        "type": "function",
                        "function": {
                            "name": value["tool_name"],
                            "arguments": str(value["parameters"]),
                        },
                    })
            else:
                parsed_tool_calls.append({
                    "index": 0,
                    "id": tool_calls["function_calls"]["invoke"]["tool_name"],
                    "type": "function",
                    "function": {
                        "name": tool_calls["function_calls"]["invoke"]["tool_name"],
                        "arguments": json.dumps(tool_calls["function_calls"]["invoke"]["parameters"]),
                    },
                })

            message.pop("text", None)
            message.pop("type", None)
            message["tool_calls"] = parsed_tool_calls
            message["content"] = None
            message["role"] = "assistant"

        if 'text' in message.keys():
            message["content"] = message["text"]
            message.pop("text", None)
            message.pop("type", None)

    if "stop_reason" in data.keys() and data["stop_reason"] == "stop_sequence":
        finish_reason = "tool_calls"

    if "stop_reason" in data.keys() and data["stop_reason"] == "end_turn":
        finish_reason = "stop"

    log("MAPPED RESPONSE DATA: ", data)

    try:
        delta = data["content"][0]
    except:
        delta = []

    translated = {
        "id": data["id"],
        "object": "chat.completion.chunk",
        "created": 0,
        "model": data["model"],
        "system_fingerprint": "TEMP",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            },
        ],
    }

    return json.dumps(translated)


async def completions(client: AsyncAnthropic | AsyncAnthropicBedrock, input: dict) -> StreamingResponse:
    log("ORIGINAL REQUEST: ", input)
    req = map_req(input)
    log("MAPPED REQUEST: ", req)

    try:
        async with client.messages.stream(
                max_tokens=req["max_tokens"],
                system=req["system"],
                messages=req["messages"],
                model=req["model"],
                temperature=req["temperature"],
                stop_sequences=["</function_calls>"],
        ) as stream:
            accumulated = await stream.get_final_message()

    except Exception as e:
        try:
            error_code = e.__dict__["status_code"]
        except:
            error_code = 500

        error_message = {"error from remote": str(e)}
        return StreamingResponse(json.dumps(error_message), media_type="application/x-ndjson", status_code=error_code)

    resp = "data: " + map_resp(accumulated.json()) + "\n\n"
    return StreamingResponse(resp, media_type="application/x-ndjson")
