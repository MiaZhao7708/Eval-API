import sys
import requests
import json 
import copy
import random 
PARAMETERS = {
    "temperature": 0.3,
    "max_new_tokens": 2048,
    "top_k": 5,
    "top_p": 0.85,
    "repetition_penalty": 1.05,
    "do_sample": True
}
ROLE_MAP = {
    "system": "<B_SYS>",
    "user_system": "<B_USYS>",
    "user": "<H_Q>",
    "assistant": "<H_A>",
    "function": "<B_FUNC>",
    "code": "<B_CODE>",
    "ape": "<B_APE>",
}
class bc(object):
    def __init__(
        self,
        server="10.64.8.61,10.64.8.101",
        parameters=PARAMETERS,
        role_map=ROLE_MAP,
    ):
        self.server = server
        self.parameters = parameters
        self.role_map = role_map
    def _parse_messages(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
        input_str = ""
        for message in messages[:-1]:
            if message["content"] is not None:
                input_str += self.role_map[message["role"]] + message["content"]
        assert messages[-1]["role"] == "user"
        input_str += "<C_Q>" + messages[-1]["content"] + "<C_A>"
        return input_str
    def _parse_parameters(self, input_parameters):
        parameters = copy.deepcopy(self.parameters)
        parameters.update(input_parameters)
        if parameters["temperature"] == 0:
            parameters["temperature"] = 1e-6
        if "max_tokens" in parameters:
            max_tokens = parameters["max_tokens"]
            if "max_new_tokens" not in parameters:
                parameters["max_new_tokens"] = max_tokens
        return parameters
    @property
    def url(self):
        if ',' in self.server:
            server = random.choice(self.server.split(','))
        else:
            server = self.server
        if server.startswith("aliyun:"):
            server = server[7:]
            if "aliyuncs" not in server:
                server = server.replace('_', '-')
                server += ".gw-gqqd25no78ncp72xfw-1151584402193309.cn-wulanchabu.pai-eas.aliyuncs.com"
        if server.startswith("http"):
            url = server
        else:
            url = f"http://{server}"
        return url
    def __call__(self, messages, **parameters):
        input_str = self._parse_messages(messages)
        parameters = self._parse_parameters(parameters)
        request_data = {
            "inputs": input_str,
            "parameters": parameters,
            "stream": True
        }
        response = requests.post(self.url, json=request_data, stream=True)
        result = {
            "response": "",
            "tokens": [],
            "token_ids": [],
            "token_count": 0
        }
        for byte_line in response.iter_lines():
            byte_line = byte_line.rstrip(b'\n')
            if byte_line.startswith(b"data:"):
                data = json.loads(byte_line.decode().lstrip("data:"))
                if data["token"]:
                    result["tokens"].append(data["token"]["text"])
                    result["token_ids"].append(data["token"]["id"])
                    result["token_count"] += 1
        try:
            result["response"] = data["generated_text"]
        except:
            tokens = result["tokens"]
            if tokens and tokens[-1] == "<|endoftext|>":
                tokens = tokens[:-1]
            result["response"] = "".join(tokens)
        return result["response"]
if __name__ == "__main__":
    bc = bc()
    print(bc([{"role": "user", "content": "你是谁"}]))
