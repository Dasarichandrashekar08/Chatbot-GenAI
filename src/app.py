import io
from json import JSONEncoder
import json
import logging
import mimetypes
from typing import Any, AsyncGenerator, Dict, cast

from azure.identity.aio import DefaultAzureCredential
from openai import AsyncAzureOpenAI

from approach import Approach
from chatreadretriveread import ChatReadRetrieveReadApproach
from config import CONFIG_AUTH_CLIENT, CONFIG_CHAT_APPROACH, CONFIG_GPT4V_DEPLOYED, CONFIG_OPENAI_CLIENT, \
    CONFIG_SEARCH_CLIENT, CONFIG_SEMANTIC_RANKER_DEPLOYED, CONFIG_VECTOR_SEARCH_ENABLED
from authentication import AuthenticationHelper
from decorators import authenticated
from error import error_dict, error_response
from text import load_json_from_file
from azure.search.documents.indexes.aio import SearchIndexClient

"""
Quart is an asynchronous web framework for Python, inspired by Flask, 
but designed to be fully compatible with Python's asyncio library. 
This allows you to write asynchronous code, which can be more efficient and scalable, 
especially for I/O-bound tasks like handling multiple web requests or working 
with databases.

"""
from quart import Quart, request, jsonify, current_app
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from quart_cors import cors

# Load environment variables from .env file
load_dotenv()

# Initialize Azure Search client
"""
The SearchClient in Azure Cognitive Search (also known as AI Search) is used to 
interact with a search index, allowing you to query and retrieve documents, 
perform searches, and manage search indexes. 
"""
app = Quart(__name__)


# Fix Windows registry issue with mimetypes
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

@app.errorhandler(403)
async def forbidden_error(e):
    response = {
        "error": "Forbidden, You don't have permission to access this resource. Incase of valid user try logout and login",
        "message": "You don't have permission to access this resource. Incase of valid user try logout and login"
    }
    return jsonify(response), 403

@app.errorhandler(401)
async def forbidden_error(e):
    response = {
        "error": "Unauthorized, You don't have permission to access this resource. Incase of valid user try logout and login",
        "message": "You don't have permission to access this resource. Incase of valid user try logout and login"
    }
    return jsonify(response), 401

@app.route("/api/auth_setup", methods=["GET"])
def auth_setup():
    auth_helper = current_app.config[CONFIG_AUTH_CLIENT]
    return jsonify(auth_helper.get_auth_setup_for_client())

async def format_as_ndjson(r: AsyncGenerator[dict, None]) -> AsyncGenerator[str, None]:
    try:
        async for event in r:
            yield json.dumps(event, ensure_ascii=False, cls=JSONEncoder) + "\n"
    except Exception as error:
        logging.exception("Exception while generating response stream: %s", error)
        yield json.dumps(error_dict(error))


@app.route("/api/chat", methods=["POST"])
@authenticated
async def chat(auth_claims: Dict[str, Any]):
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    context = request_json.get("context", {})

    custom_context_data = load_json_from_file("searchconfig.json")
    context = custom_context_data['context']
    context["auth_claims"] = auth_claims
    try:
        approach: Approach
        approach = cast(Approach, current_app.config[CONFIG_CHAT_APPROACH])

        user_role= auth_claims["roles"][0] if auth_claims and "roles" in auth_claims and len(auth_claims["roles"]) > 0 else ""
        result = await approach.run(
            request_json["messages"],
            user_role,
            context=context,
            session_state=request_json.get("session_state"),
        )
        return jsonify(result)
    except Exception as error:
        return error_response(error, "/chat")   

@app.before_serving
async def setup_clients():
    # Replace these with your own values, either in environment variables or directly here
    AZURE_SEARCH_SERVICE = os.environ["AZURE_SEARCH_SERVICE"]
    AZURE_SEARCH_INDEX = os.environ["AZURE_SEARCH_INDEX"]
    print("Search index name:", AZURE_SEARCH_INDEX)
    # Shared by all OpenAI deployments
    OPENAI_HOST = os.getenv("OPENAI_HOST", "azure")
    OPENAI_CHATGPT_MODEL = os.environ["AZURE_OPENAI_CHATGPT_MODEL"]
    # Used with Azure OpenAI deployments
    AZURE_OPENAI_SERVICE = os.getenv("AZURE_OPENAI_SERVICE")
    AZURE_OPENAI_CHATGPT_DEPLOYMENT = (
        os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT") if OPENAI_HOST.startswith("azure") else None
    )
    OPENAI_EMB_MODEL = os.getenv("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-ada-002")
    OPENAI_EMB_DIMENSIONS = int(os.getenv("AZURE_OPENAI_EMB_DIMENSIONS", 1536))
    AZURE_OPENAI_EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT") if OPENAI_HOST.startswith("azure") else None
    AZURE_OPENAI_CUSTOM_URL = os.getenv("AZURE_OPENAI_CUSTOM_URL")
    AZURE_SEARCH_QUERY_LANGUAGE = os.getenv("AZURE_SEARCH_QUERY_LANGUAGE", "en-us")
    AZURE_SEARCH_QUERY_SPELLER = os.getenv("AZURE_SEARCH_QUERY_SPELLER", "lexicon")
    AZURE_SEARCH_SEMANTIC_RANKER = os.getenv("AZURE_SEARCH_SEMANTIC_RANKER", "free").lower()
    USE_GPT4V = os.getenv("USE_GPT4V", "").lower() == "true"

    # AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT", "")
    AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
    AZURE_USE_AUTHENTICATION = os.getenv("AZURE_USE_AUTHENTICATION", "").lower() == "true"
    AZURE_ENFORCE_ACCESS_CONTROL = os.getenv("AZURE_ENFORCE_ACCESS_CONTROL", "").lower() == "true"
    AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS = os.getenv("AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS", "").lower() == "true"
    AZURE_ENABLE_UNAUTHENTICATED_ACCESS = os.getenv("AZURE_ENABLE_UNAUTHENTICATED_ACCESS", "").lower() == "true"
    AZURE_SERVER_APP_ID = os.getenv("AZURE_SERVER_APP_ID")
    AZURE_SERVER_APP_SECRET = os.getenv("AZURE_SERVER_APP_SECRET")
    AZURE_CLIENT_APP_ID = os.getenv("AZURE_CLIENT_APP_ID")
    AZURE_AUTH_TENANT_ID = os.getenv("AZURE_AUTH_TENANT_ID", AZURE_TENANT_ID)

    # Use the current user identity to authenticate with Azure OpenAI, AI Search and Blob Storage (no secrets needed,
    # just use 'az login' locally, and managed identity when deployed on Azure). If you need to use keys, use separate AzureKeyCredential instances with the
    # keys for each service
    # If you encounter a blocking error during a DefaultAzureCredential resolution, you can exclude the problematic credential by using a parameter (ex. exclude_shared_token_cache_credential=True)
    # azure_credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)

    # Set up clients for AI Search and Storage
    search_client = SearchClient(
        endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
        index_name=AZURE_SEARCH_INDEX,
        # credential=azure_credential,     
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")) if os.getenv(
            "AZURE_SEARCH_API_KEY") else DefaultAzureCredential()
    )

    # Set up authentication helper
    search_index = None
    if AZURE_USE_AUTHENTICATION:
        search_index_client = SearchIndexClient(
            endpoint=f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY")) if os.getenv(
            "AZURE_SEARCH_API_KEY") else DefaultAzureCredential()
        )
        search_index = await search_index_client.get_index(AZURE_SEARCH_INDEX)
        await search_index_client.close()
    auth_helper = AuthenticationHelper(
        search_index=search_index,
        use_authentication=AZURE_USE_AUTHENTICATION,
        server_app_id=AZURE_SERVER_APP_ID,
        server_app_secret=AZURE_SERVER_APP_SECRET,
        client_app_id=AZURE_CLIENT_APP_ID,
        tenant_id=AZURE_AUTH_TENANT_ID,
        require_access_control=AZURE_ENFORCE_ACCESS_CONTROL,
        enable_global_documents=AZURE_ENABLE_GLOBAL_DOCUMENT_ACCESS,
        enable_unauthenticated_access=AZURE_ENABLE_UNAUTHENTICATED_ACCESS,
    )

    # Used by the OpenAI SDK
    openai_client: AsyncAzureOpenAI

    if OPENAI_HOST.startswith("azure"):
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2023-03-15-preview"
        if OPENAI_HOST == "azure_custom":
            if not AZURE_OPENAI_CUSTOM_URL:
                raise ValueError("AZURE_OPENAI_CUSTOM_URL must be set when OPENAI_HOST is azure_custom")
            endpoint = AZURE_OPENAI_CUSTOM_URL
        else:
            if not AZURE_OPENAI_SERVICE:
                raise ValueError("AZURE_OPENAI_SERVICE must be set when OPENAI_HOST is azure")
            endpoint = f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com"
        if api_key := os.getenv("AZURE_OPENAI_API_KEY"):
            openai_client = AsyncAzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)

    current_app.config[CONFIG_OPENAI_CLIENT] = openai_client
    current_app.config[CONFIG_SEARCH_CLIENT] = search_client
    current_app.config[CONFIG_AUTH_CLIENT] = auth_helper

    current_app.config[CONFIG_GPT4V_DEPLOYED] = bool(USE_GPT4V)
    current_app.config[CONFIG_SEMANTIC_RANKER_DEPLOYED] = AZURE_SEARCH_SEMANTIC_RANKER != "disabled"
    current_app.config[CONFIG_VECTOR_SEARCH_ENABLED] = os.getenv("USE_VECTORS", "").lower() != "false"
    current_app.config[CONFIG_CHAT_APPROACH] = ChatReadRetrieveReadApproach(
        search_client=search_client,
        openai_client=openai_client,
        auth_helper=auth_helper,
        chatgpt_model=OPENAI_CHATGPT_MODEL,
        chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        embedding_model=OPENAI_EMB_MODEL,
        embedding_deployment=AZURE_OPENAI_EMB_DEPLOYMENT,
        embedding_dimensions=OPENAI_EMB_DIMENSIONS,
        query_language=AZURE_SEARCH_QUERY_LANGUAGE,
        query_speller=AZURE_SEARCH_QUERY_SPELLER
    )

@app.after_serving
async def close_clients():
    await current_app.config[CONFIG_SEARCH_CLIENT].close()
    os.environ.pop("AZURE_SEARCH_SERVICE", None)
    os.environ.pop("AZURE_SEARCH_INDEX", None)


# Health Check
@app.route("/api/health/", methods=['GET'])
def health_check():
    """
    Basic health check route.
    """
    return {"status": "healthy"}, 200


def create_app():
    # Level should be one of https://docs.python.org/3/library/logging.html#logging-levels
    default_level = "INFO"  # In development, log more verbosely
    if os.getenv("WEBSITE_HOSTNAME"):  # In production, don't log as heavily
        default_level = "WARNING"
    logging.basicConfig(level=os.getenv("APP_LOG_LEVEL", default_level))

    if allowed_origin := os.getenv("ALLOWED_ORIGIN"):
        app.logger.info("CORS enabled for %s", allowed_origin)
        cors(app, allow_origin=allowed_origin, allow_methods=["GET", "POST"])
    return app


# Run the Quart app
if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=8020)
