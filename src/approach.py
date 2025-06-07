import os
from abc import ABC
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Optional,
    cast,
)

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import (
    QueryType,
    QueryAnswerType,
    QueryCaptionType,
    VectorQuery,
)
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from authentication import AuthenticationHelper
from text import nonewlines


@dataclass
class Document:
    id: Optional[str]
    content: Optional[str]
    embedding: Optional[List[float]]
    score: Optional[float] = None
    reranker_score: Optional[float] = None
    page_number: Optional[int] = None
    file_path : Optional[str] = None
    area: Optional[str] = None
    file_name: Optional[str] = None
    modified_date: Optional[str] = None
    vertical: Optional[str] = None
    client_name: Optional[str] = None
    program: Optional[str] = None
    technologies_stack: Optional[str] = None
    location: Optional[str] = None
    type: Optional[str] = None
    domain: Optional[str] = None

@dataclass
class ThoughtStep:
    title: str
    description: Optional[Any]
    props: Optional[dict[str, Any]] = None


class Approach(ABC):
    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AsyncOpenAI,
        auth_helper: AuthenticationHelper,
        query_language: Optional[str],
        query_speller: Optional[str],
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        openai_host: str,
    ):
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.query_language = query_language
        self.query_speller = query_speller
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.openai_host = openai_host

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        exclude_category = overrides.get("exclude_category")
        area_filter = overrides.get("area")  # Get the 'area' filter from overrides
        vertical_filter = overrides.get("vertical")
        client_filter = overrides.get("client_name")
            # Add metadata-based filters
        # security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []
        if area_filter:
            filters.append(f"area eq '{area_filter}'")  # Filter by 'area' metadata

        if vertical_filter:
            filters.append(f"vertical eq '{vertical_filter}'")  # Filter by 'vertical' metadata

        if client_filter:
            filters.append(f"client_name eq '{client_filter}'")  # Filter by 'client_name' metadata
        
        if exclude_category:
            filters.append("category ne '{}'".format(exclude_category.replace("'", "''")))
            
        # if security_filter:
        #     filters.append(security_filter)
        return None if len(filters) == 0 else " and ".join(filters)

    async def search(
        self,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: List[VectorQuery],
        use_text_search: bool,
        use_vector_search: bool,
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: Optional[float],
        minimum_reranker_score: Optional[float],
        semantic_config_name: Optional[str]
    ) -> List[Document]:
        search_text = query_text if use_text_search else ""
        search_vectors = vectors if use_vector_search else []
        if use_semantic_ranker:
            results = self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                query_caption=QueryCaptionType.EXTRACTIVE,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name= semantic_config_name,
                semantic_query=query_text,
                query_answer=QueryAnswerType.EXTRACTIVE
            )
        else:
            results = self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                vector_queries=search_vectors,
            )

        documents = [] 
        for document in results:
                documents.append(
                    Document(
                        id=document.get("chunk_id"),
                        content=document.get("chunk"),
                        embedding=document.get("text_vector"),
                        score=document.get("@search.score"),
                        reranker_score=document.get("@search.reranker_score"),
                        page_number=document.get("page_number"),
                        file_path = document.get('fileURL'),
                        file_name = document.get('file_name'),
                        area = document.get('area'),
                        modified_date = document.get('modified_date'),
                        vertical = document.get('vertical'),
                        client_name = document.get('client_name'),
                        technologies_stack = document.get('technologies_stack'),
                        program = document.get('program'),
                        location = document.get('location'),
                        type = document.get("type"),
                        domain = document.get("domain")
                    )
                )

        qualified_documents = [
            doc
            for doc in documents
                if (
                    (doc.score or 0) >= (minimum_search_score or 0)
                    and (doc.reranker_score or 0) >= (minimum_reranker_score or 0)
                )
        ]

        return qualified_documents

    def get_sources_content(
        self, results: List[Document], use_semantic_captions: bool, use_image_citation: bool
    ) -> list[str]:
        if use_semantic_captions:
            return [
                (self.get_citation((doc.file_path or ""), use_image_citation))
                + ": "
                + nonewlines(" . ".join([cast(str, c.text) for c in (doc.captions or [])]))
                for doc in results
            ]
        else:
            return [
                (self.get_citation((doc.file_path or ""), use_image_citation)) + ": " + nonewlines(doc.content or "")
                for doc in results
            ]

    def get_citation(self, sourcepage: str, use_image_citation: bool) -> str:
        if use_image_citation:
            return sourcepage
        else:
            path, ext = os.path.splitext(sourcepage)
            if ext.lower() == ".png":
                page_idx = path.rfind("-")
                page_number = int(path[page_idx + 1 :])
                return f"{path[:page_idx]}.pdf#page={page_number}"

            return sourcepage

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        user_role: str,
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        raise NotImplementedError