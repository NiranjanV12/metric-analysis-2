import logging
from langchain_core.documents import Document
import os
from langchain_litellm import ChatLiteLLM
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from src.shared.constants import ADDITIONAL_INSTRUCTIONS
from src.shared.llm_graph_builder_exception import LLMGraphBuilderException
import re
from typing import List
from langchain_core.callbacks.manager import CallbackManager
from src.shared.common_fn import UniversalTokenUsageHandler, get_value_from_env

_OLD_MODEL_MAP = {
    "GPT_4O": "openai/gpt-4o",
    "GPT_4O_MINI": "openai/gpt-4o-mini",
    "GPT_4": "openai/gpt-4",
    "GPT_3_5_TURBO": "openai/gpt-3.5-turbo",
    "GPT_3_5_TURBO_16K": "openai/gpt-3.5-turbo-16k",
    "GPT_5_MINI": "openai/gpt-5-mini",
    "CLAUDE_3_5_SONNET_20241022": "anthropic/claude-3-5-sonnet-20241022",
    "CLAUDE_3_5_SONNET": "anthropic/claude-3-5-sonnet-20240620",
    "CLAUDE_3_OPUS_20240229": "anthropic/claude-3-opus-20240229",
    "CLAUDE_3_HAIKU_20240307": "anthropic/claude-3-haiku-20240307",
    "GEMINI_2_5_FLASH": "vertex_ai/gemini-2.5-flash",
    "GEMINI_2_5_PRO": "vertex_ai/gemini-2.5-pro",
    "GEMINI_1_5_FLASH": "vertex_ai/gemini-1.5-flash",
    "GEMINI_1_5_PRO": "vertex_ai/gemini-1.5-pro",
}


def _resolve_litellm_model(model_key: str, env_value: str) -> str:
    """Resolve a LiteLLM model string from an env var value.

    Supports two formats:
      - New: "openai/gpt-4o" (already a LiteLLM model string)
      - Old: "gpt-4o,sk-xxx" or "model_name,api_key" (comma-separated, possibly with API key)

    For old format, the model name is mapped to a LiteLLM provider-prefixed string
    and the API key is set as an environment variable.
    """
    if "/" in env_value:
        return env_value.strip()

    parts = env_value.split(",")
    raw_model = parts[0].strip()

    api_key = parts[1].strip() if len(parts) > 1 else None
    if api_key:
        if "ANTHROPIC" in model_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
        elif "GROQ" in model_key:
            os.environ.setdefault("GROQ_API_KEY", api_key)
        else:
            os.environ.setdefault("OPENAI_API_KEY", api_key)

    if model_key in _OLD_MODEL_MAP:
        return _OLD_MODEL_MAP[model_key]

    if "GEMINI" in model_key or "VERTEX" in model_key:
        return f"vertex_ai/{raw_model}"
    if "ANTHROPIC" in model_key or "CLAUDE" in model_key:
        return f"anthropic/{raw_model}"
    if "BEDROCK" in model_key:
        return f"bedrock/{raw_model}"
    if "GROQ" in model_key:
        return f"groq/{raw_model}"
    if "OLLAMA" in model_key:
        return f"ollama/{raw_model}"
    if "AZURE" in model_key:
        return f"azure/{raw_model}"

    return f"openai/{raw_model}"


def get_llm(model: str):
    """Retrieve the specified language model based on the model name.

    The LLM_MODEL_CONFIG_<MODEL> env var should contain a LiteLLM model string,
    e.g. "openai/gpt-4o", "vertex_ai/gemini-2.5-flash", "anthropic/claude-3-5-sonnet".

    Legacy format "model_name,api_key" is also supported and auto-converted.

    For Diffbot, the env var format is: "diffbot,<api_key>"
    """
    model = model.upper().replace('.', '_').strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = get_value_from_env(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    callback_handler = UniversalTokenUsageHandler()
    callback_manager = CallbackManager([callback_handler])
    try:
        if "DIFFBOT" in model:
            model_name, api_key = env_value.split(",")
            llm = DiffbotGraphTransformer(
                diffbot_api_key=api_key,
                extract_types=["entities", "facts"],
            )
            callback_handler = None
        else:
            litellm_model = _resolve_litellm_model(model, env_value)
            logging.info(f"Resolved LiteLLM model: {litellm_model}")
            llm = ChatLiteLLM(
                model=litellm_model,
                temperature=0,
                callbacks=callback_manager,
            )
            model_name = litellm_model
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name, callback_handler

def get_llm_model_name(llm):
    """Extract name of llm model from llm object"""
    for attr in ["model_name", "model", "model_id"]:
        model_name = getattr(llm, attr, None)
        if model_name:
            return model_name.lower()
    logging.info("Could not determine model name; defaulting to empty string")
    return ""

def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list

def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
       Document(
           page_content=document["chunk_doc"].page_content,
           metadata={"chunk_id": [document["chunk_id"]]},
       )
       for document in chunkId_chunkDoc_list
   ]
    return combined_chunk_document_list
      

async def get_graph_document_list(
    llm, combined_chunk_document_list, allowedNodes, allowedRelationship,callback_handler, additional_instructions=None
):
    if additional_instructions:
        additional_instructions = sanitize_additional_instruction(additional_instructions)
    graph_document_list = []
    token_usage = 0
    try:
        if "diffbot_api_key" in dir(llm):
            llm_transformer = llm
        else:
            logging.info("Using text-based graph extraction (structured output schema incompatible with OpenAI strict validation)")
            node_properties = False
            relationship_properties = False
            ignore_tool_usage = True
            
            llm_transformer = LLMGraphTransformer(
                llm=llm,
                node_properties=node_properties,
                relationship_properties=relationship_properties,
                allowed_nodes=allowedNodes,
                allowed_relationships=allowedRelationship,
                ignore_tool_usage=ignore_tool_usage,
                additional_instructions=ADDITIONAL_INSTRUCTIONS+ (additional_instructions if additional_instructions else "")
            )
        
        if isinstance(llm,DiffbotGraphTransformer):
            graph_document_list = llm_transformer.convert_to_graph_documents(combined_chunk_document_list)
        else:
            graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
        
        # Debug: Log graph document details
        logging.info(f"=== DEBUG: Graph transformation completed ===")
        for i, gd in enumerate(graph_document_list[:3]):  # Log first 3 documents
            logging.info(f"Document {i}: {len(gd.nodes)} nodes, {len(gd.relationships)} relationships")
            for node in gd.nodes[:3]:  # Log first 3 nodes
                logging.info(f"  Node: type={node.type}, id={node.id}, properties={dict(node.properties)}")
    except Exception as e:
       logging.error(f"Error in graph transformation: {e}", exc_info=True)
       raise LLMGraphBuilderException(f"Graph transformation failed: {str(e)}")
    finally:
        try:
            if callback_handler:
                usage = callback_handler.report()
                token_usage = usage.get("total_tokens", 0)
        except Exception as usage_err:
            logging.error(f"Error while reporting token usage: {usage_err}")

    return graph_document_list, token_usage

async def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship, chunks_to_combine, additional_instructions=None):
   try:
       llm, model_name,callback_handler = get_llm(model)
       logging.info(f"Using model: {model_name}")
    
       combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine)
       logging.info(f"Combined {len(combined_chunk_document_list)} chunks")
    
       if allowedNodes:
           allowed_nodes = [node.strip() for node in allowedNodes.split(',') if node.strip()]
       else:
           allowed_nodes = []
       logging.info(f"Allowed nodes: {allowed_nodes}")
    
       allowed_relationships = []
       if allowedRelationship:
           items = [item.strip() for item in allowedRelationship.split(',') if item.strip()]
           if len(items) % 3 != 0:
               raise LLMGraphBuilderException("allowedRelationship must be a multiple of 3 (source, relationship, target)")
           for i in range(0, len(items), 3):
               source, relation, target = items[i:i + 3]
               if source not in allowed_nodes or target not in allowed_nodes:
                   raise LLMGraphBuilderException(
                       f"Invalid relationship ({source}, {relation}, {target}): "
                       f"source or target not in allowedNodes"
                   )
               allowed_relationships.append((source, relation, target))
           logging.info(f"Allowed relationships: {allowed_relationships}")
       else:
           logging.info("No allowed relationships provided")

       graph_document_list,token_usage = await get_graph_document_list(
           llm,
           combined_chunk_document_list,
           allowed_nodes,
           allowed_relationships,
           callback_handler,
           additional_instructions,
       )
       logging.info(f"Generated {len(graph_document_list)} graph documents")
       return graph_document_list, token_usage
   except Exception as e:
       logging.error(f"Error in get_graph_from_llm: {e}", exc_info=True)
       raise LLMGraphBuilderException(f"Error in getting graph from llm: {e}")

def sanitize_additional_instruction(instruction: str) -> str:
   """
   Sanitizes additional instruction by:
   - Replacing curly braces `{}` with `[]` to prevent variable interpretation.
   - Removing potential injection patterns like `os.getenv()`, `eval()`, `exec()`.
   - Stripping problematic special characters.
   - Normalizing whitespace.
   Args:
       instruction (str): Raw additional instruction input.
   Returns:
       str: Sanitized instruction safe for LLM processing.
   """
   logging.info("Sanitizing additional instructions")
   instruction = instruction.replace("{", "[").replace("}", "]")  # Convert `{}` to `[]` for safety
   # Step 2: Block dangerous function calls
   injection_patterns = [r"os\.getenv\(", r"eval\(", r"exec\(", r"subprocess\.", r"import os", r"import subprocess"]
   for pattern in injection_patterns:
       instruction = re.sub(pattern, "[BLOCKED]", instruction, flags=re.IGNORECASE)
   # Step 4: Normalize spaces
   instruction = re.sub(r'\s+', ' ', instruction).strip()
   return instruction
