import json
import os
import re

import xmltodict
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from fastapi.responses import JSONResponse, StreamingResponse

debug = os.environ.get("GPTSCRIPT_DEBUG", "false") == "true"


def log(*args):
    if debug:
        print(*args)


async def list_models(client: AsyncAnthropic | AsyncAnthropicBedrock) -> JSONResponse:
    if type(client) == AsyncAnthropic:
        data = [{"id": "claude-3-opus-20240229", "name": "Anthropic Claude 3 Opus"},
                {"id": "claude-3-sonnet-20240229", "name": "Anthropic Claude 3 Sonnet"},
                {"id": "claude-3-haiku-20240307", "name": "Anthropic Claude 3 Haiku"},
                {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet"} ]
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
            content: str = construct_tool_outputs_message([message], None)
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
                        "tool_arguments": tool_call["function"].get("arguments", None),
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

    def prepend_if_unique(lst, new_dict, key, value):
        if not any(d.get(key) == value for d in lst):
            lst.insert(0, new_dict)

    prepend_if_unique(mapped_messages, {"role": "user", "content": "."}, "role", "user")
    if mapped_messages[0]["role"] != "user":
        mapped_messages.insert(0, {"role": "user", "content": "."})
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
    system = """
You are task oriented system.
You receive input from a user, process the input from the given instructions, and then output the result.
Your objective is to provide consistent and correct results.
You do not need to explain the steps taken, only provide the result to the given instructions.
You are referred to as a tool.
You don't move to the next step until you have a result.
"""
    if req["system"] is not None:
        system = system + req["system"]
    try:
        async with client.messages.stream(
                max_tokens=req["max_tokens"],
                system=system,
                # system=req["system"],
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


# This file contains prompt constructors for various pieces of code. Used primarily to keep other code legible.
def construct_tool_use_system_prompt(tools):
    tool_use_system_prompt = (
            "In this environment you have access to a set of tools you can use to answer the user's question.\n"
            "\n"
            "You may call them like this:\n"
            "<function_calls>\n"
            "<invoke>\n"
            "<tool_name>$TOOL_NAME</tool_name>\n"
            "<parameters>\n"
            "<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>\n"
            "...\n"
            "</parameters>\n"
            "</invoke>\n"
            "</function_calls>\n"
            "\n"
            "Here are the tools available:\n"
            "<tools>\n"
            + '\n'.join(
        [construct_format_tool_for_claude_prompt(tool["function"]["name"], tool["function"].get("description", ""),
                                                 tool["function"].get("parameters", {}).get("properties", {})) for tool
         in
         tools]) +
            "\n</tools>"
    )

    return tool_use_system_prompt


def construct_successful_function_run_injection_prompt(invoke_results_results) -> str:
    constructed_prompt = (
            "<function_results>\n"
            + '\n'.join(
        f"<result>\n<tool_name>{res['tool_call_id']}</tool_name>\n<stdout>\n{res['content']}\n</stdout>\n</result>" for
        res in invoke_results_results) +
            "\n</function_results>"
    )

    return constructed_prompt


def construct_error_function_run_injection_prompt(invoke_results_error_message) -> str:
    constructed_prompt = (
        "<function_results>\n"
        "<system>\n"
        f"{invoke_results_error_message}"
        "\n</system>"
        "\n</function_results>"
    )

    return constructed_prompt


def construct_format_parameters_prompt(parameters) -> str:
    constructed_prompt = "\n".join(
        f"<parameter>\n<name>{key}</name>\n<type>{value['type']}</type>\n<description>{value.get('description', '')}</description>\n</parameter>"
        for key, value in parameters.items())

    return constructed_prompt


def construct_format_tool_for_claude_prompt(name, description, parameters) -> str:
    constructed_prompt = (
        "<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        "<description>\n"
        f"{description}\n"
        "</description>\n"
        "<parameters>\n"
        f"{construct_format_parameters_prompt(parameters)}\n"
        "</parameters>\n"
        "</tool_description>"
    )

    return constructed_prompt


def construct_tool_inputs_message(content, tool_inputs) -> str:
    def format_parameters(tool_arguments):
        log("TOOL ARGS: ", tool_arguments)
        if tool_arguments != "null":
            return '\n'.join([f'<{key}>{value}</{key}>' for key, value in json.loads(tool_arguments).items()])
        else:
            return ""

    single_call_messages = "\n\n".join([
        f"<invoke>\n<tool_name>{tool_input['tool_name']}</tool_name>\n<parameters>\n{format_parameters(tool_input['tool_arguments'])}\n</parameters>\n</invoke>"
        for tool_input in tool_inputs])
    message = (
        f"{content}"
        "\n\n<function_calls>\n"
        f"{single_call_messages}\n"
        "</function_calls>"
    )
    return message


def construct_tool_outputs_message(tool_outputs, tool_error) -> str:
    if tool_error is not None:
        message = construct_error_function_run_injection_prompt(tool_error)
        return f"\n\n{message}"
    elif tool_outputs is not None:
        message = construct_successful_function_run_injection_prompt(tool_outputs)
        return f"\n\n{message}"
    else:
        raise ValueError("At least one of tool_result or tool_error must not be None.")
