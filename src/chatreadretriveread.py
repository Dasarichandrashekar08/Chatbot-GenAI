import os
import re
from typing import Any, Coroutine, List, Literal, Optional, Union, overload, Tuple

from azure.search.documents.aio import SearchClient
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai_messages_token_helper import build_messages, get_token_limit

from approach import ThoughtStep
from chatapproach import ChatApproach
from authentication import AuthenticationHelper
from text import load_yml_content_from_file
from azure.search.documents.models import VectorizableTextQuery


class ChatReadRetrieveReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then uses Azure AI Search to retrieve relevant documents, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
            self,
            *,
            search_client: SearchClient,
            auth_helper: AuthenticationHelper,
            openai_client: AsyncOpenAI,
            chatgpt_model: str,
            chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
            embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
            embedding_model: str,
            embedding_dimensions: int,
            query_language: str,
            query_speller: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    @property
    def system_message_chat_conversation(self):
        return ""

    @overload
    async def run_until_final_call(
            self,
            messages: list[ChatCompletionMessageParam],
            user_role: str,
            overrides: dict[str, Any],
            auth_claims: dict[str, Any],
            should_stream: Literal[False],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, ChatCompletion]]:
        ...

    @overload
    async def run_until_final_call(
            self,
            messages: list[ChatCompletionMessageParam],
            user_role: str,
            overrides: dict[str, Any],
            auth_claims: dict[str, Any],
            should_stream: Literal[True],
    ) -> tuple[dict[str, Any], Coroutine[Any, Any, AsyncStream[ChatCompletionChunk]]]:
        ...

    async def run_until_final_call(
            self,
            messages: list[ChatCompletionMessageParam],
            user_role: str,
            overrides: dict[str, Any],
            auth_claims: dict[str, Any],
            should_stream: bool = False,
    ) -> tuple[list[str], Coroutine[Any, Any, ChatCompletion | AsyncStream[ChatCompletionChunk]]]:
        seed = overrides.get("seed", None)
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        semantic_config_name = overrides.get("semantic_configuration_name", "default")
        filter = self.build_filter(overrides, auth_claims)

        original_user_query = messages[-1]["content"]
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")
        user_query_request = "Generate search query for: " + original_user_query

        tools: List[ChatCompletionToolParam] = [
            {
                "type": "function",
                "function": {
                    "name": "search_sources",
                    "description": "Retrieve sources from the Azure AI Search index",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Query string to retrieve documents from azure search eg: 'Warba Bank proposal'",
                            }
                        },
                        "required": ["search_query"],
                    },
                },
            }
        ]

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        query_response_token_limit = 200
        query_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=self.query_prompt_template,
            tools=tools,
            few_shots=self.query_prompt_few_shots,
            past_messages=messages[:-1],
            new_user_content=user_query_request,
            max_tokens=self.chatgpt_token_limit - query_response_token_limit,
        )

        chat_completion: ChatCompletion = await self.openai_client.chat.completions.create(
            messages=query_messages,  # type: ignore
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            temperature=0.0,  # Minimize creativity for search query generation
            max_tokens=query_response_token_limit,
            # Setting too low risks malformed JSON, setting too high may affect performance
            n=1,
            tools=tools,
            seed=seed,
        )

        query_text = self.get_search_query(chat_completion, original_user_query)
        vectors_i = VectorizableTextQuery(text=query_text, k_nearest_neighbors=20, fields="vector", exhaustive=True)
        vectors = [vectors_i]

        results = await self.search(
            top,
            query_text,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
            semantic_config_name
        )
        sourcePaths = []
        for doc in results:
            chunk_id = doc.id
            # Extracting the page number using regex
            page_number = re.search(r'_pages_(\d+)', chunk_id).group(1)
            reference = (doc.file_path if doc.file_path else "") + '#page=' + str(int(page_number) + 1)
            # Set the updated reference back to the document's `pageURL` attribute

            doc.file_path = doc.domain if doc.domain else ""+ reference
            sourcePaths.append(doc.file_path)

        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        content = "\n".join(sources_content)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        
        custom_prompt = load_yml_content_from_file("config_prompt_template.yml")
        source_paths_concatenation = " or ".join(sourcePaths) if len(sourcePaths)>0 else ""
        custom_prompt["prompt_template"].replace("{{ info_bank_urls }}", source_paths_concatenation),
        pricing_details_based_on_role = custom_prompt["admin_paragraph"] if user_role=="adminuser"  else custom_prompt["non_admin_paragraph"]
        system_message = self.get_system_prompt(
            custom_prompt["prompt_template"].replace("{{ pricing_details_based_on_role }}", pricing_details_based_on_role),
            self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else "",
        )

        response_token_limit = int(os.getenv("AZURE_LLM_RESPONSE_TOKEN_LIMIT")) or 1024
        messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=system_message,
            past_messages=messages[:-1],
            # Model does not handle lengthy system messages well. Moving sources to latest user conversation to solve
            # follow up questions prompt.
            new_user_content=original_user_query + "\n\nSources:\n" + content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Prompt to generate search query",
                    [str(message) for message in query_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
                ThoughtStep(
                    "Search using generated search query",
                    query_text,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        chat_coroutine = self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=should_stream,
            seed=seed,
        )
        return extra_info, chat_coroutine
