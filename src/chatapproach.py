import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from approach import Approach
from text import load_yml_content_from_file

from bs4 import BeautifulSoup
from urllib.parse import urlparse
from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import Anonymize, PromptInjection
from llm_guard.output_scanners import Deanonymize, Toxicity
from llm_guard.vault import Vault


def source_urls_checking(html_string, source_paths):
    soup = BeautifulSoup(html_string, "html.parser")

    allowed_domains = {urlparse(a_tag).netloc for a_tag in source_paths}

    for a_tag in soup.find_all('a', href=True):
        # Extract the domain from the href link
        domain = urlparse(a_tag['href']).netloc
        # If the domain is not in the allowed domains, remove the <a> tag
        if domain not in allowed_domains:
            a_tag.decompose()

    return str(soup)


vault = Vault()
input_scanners = [Anonymize(vault), PromptInjection()]
output_scanners = [Deanonymize(vault), Toxicity()]


class ChatApproach(Approach, ABC):
    query_prompt_few_shots: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "what is the proposal for warba bank?"},
        {"role": "assistant", "content": "explain the details of warba bank proposal"},
        {"role": "user", "content": "so when did warba bank proposal done?"},
        {"role": "assistant", "content": "show the date and time when it done are where it done"},
    ]
    NO_RESPONSE = "0"

    config_data = load_yml_content_from_file("config_prompt_template.yml")

    # Assign to variables
    follow_up_questions_prompt_content = config_data["follow_up_questions_prompt_content"]
    query_prompt = config_data["query_prompt_template"]
    sanitized_prompt, results_valid, _ = scan_prompt(input_scanners, query_prompt)
    if not all(results_valid.values()):
        raise ValueError("blocked_by_llm_guard")
    query_prompt_template = sanitized_prompt

    @property
    @abstractmethod
    def system_message_chat_conversation(self) -> str:
        pass

    @abstractmethod
    async def run_until_final_call(self, messages, overrides, auth_claims, should_stream) -> tuple:
        pass

    def get_system_prompt(self, override_prompt: Optional[str], follow_up_questions_prompt: str) -> str:
        if override_prompt is None:
            return self.system_message_chat_conversation.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
            )
        elif override_prompt.startswith(">>>"):
            return self.system_message_chat_conversation.format(
                injected_prompt=override_prompt[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
            )
        else:
            return override_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt)

    def get_search_query(self, chat_completion: ChatCompletion, user_query: str):
        response_message = chat_completion.choices[0].message

        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                function = tool.function
                if function.name == "search_sources":
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", self.NO_RESPONSE)
                    if search_query != self.NO_RESPONSE:
                        return search_query
        elif query_text := response_message.content:
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query

    def extract_followup_questions(self, content: str):
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    async def run_without_streaming(
            self,
            messages: list[ChatCompletionMessageParam],
            user_role: str,
            overrides: dict[str, Any],
            auth_claims: dict[str, Any],
            session_state: Any = None,
    ) -> dict[str, Any]:
        concatenated_messages = " ".join(item["content"] for item in messages)
        _, results_valid, _ = scan_output(output_scanners, self.query_prompt_template, concatenated_messages)
        if not all(results_valid.values()):
            raise ValueError("blocked_by_llm_guard")
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, user_role, overrides, auth_claims, should_stream=False
        )
        chat_completion_response: ChatCompletion = await chat_coroutine
        chat_resp = chat_completion_response.model_dump()  # Convert to dict to make it JSON serializable
        chat_resp = chat_resp["choices"][0]
        chat_resp["context"] = extra_info
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(chat_resp["message"]["content"])
            chat_resp["message"]["content"] = content
            chat_resp["context"]["followup_questions"] = followup_questions
        chat_resp["session_state"] = session_state
        _, results_valid, _ = scan_output(output_scanners, self.query_prompt_template,
                                          chat_resp.get("message", {}).get("content", None))
        if not all(results_valid.values()):
            raise ValueError("blocked_by_llm_guard")
        return chat_resp

    async def run(
            self,
            messages: list[ChatCompletionMessageParam],
            user_role: str,
            session_state: Any = None,
            context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return await self.run_without_streaming(messages, user_role, overrides, auth_claims, session_state)
