import os
import json
import re
import requests
import logging
import sys
import asyncio
import traceback
from io import BytesIO
from dotenv import load_dotenv
from typing_extensions import TypedDict


# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.types import Send
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List
from langchain_neo4j import Neo4jGraph


def execute_neo4j_query(query, params=None):
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database=neo4j_database
    )
    
    print(f"Neo4j Query: {query}")
    print(f"Neo4j Params: {params}")
    
    result = graph.query(query, params or {})
    return result


def parse_source_of_truth_response(neo4j_query_result, type):
    result_dict = {}
    
    for record in neo4j_query_result:

        if(type=="service"):
            service = record.get('service', {})
            result_dict['serviceName'] = service.get('id', '')    
        elif(type=="logs"):
            service = record.get('level', {})
            print("ssssssssssssssssss",service)
            result_dict['log_level'] = service.get('id', '') 
        elif(type=="health"):
            service = record.get('service', {})
            result_dict['serviceName'] = service.get('id', '')             
#{'methodName': 'getLogsErrors', 'message': "Neo4j query result for : [{'level': {'id': 'ERROR'}, 'r1': ({'id': 'ERROR'}, 'HAS_LOG', {'id': 'database exception'}), 'n': {'id': 'database exception'}}, {'level': {'id': 'ERROR'}, 'r1': ({'id': 'ERROR'}, 'HAS_LOG', {'id': 'static-3 : unhealthy'}), 'n': {'id': 'static-3 : unhealthy'}}, {'level': {'id': 'ERROR'}, 'r1': ({'id': 'ERROR'}, 'HAS_LOG', {'id': 'app-service : unhealthy'}), 'n': {'id': 'app-service : unhealthy'}}, {'level': {'id': 'ERROR'}, 'r1': ({'id': 'ERROR'}, 'HAS_LOG', {'id': 'File Not found exception: path /data99/user1/file123.txt'}), 'n': {'id': 'File Not found exception: path /data99/user1/file123.txt'}}]"}
#{'methodName': 'analysisAndSolution', 'message': "Neo4j query result for app-service: [{'service': {'id': 'app-service'}, 'r1': ({'id': 'app-service'}, 'DEPENDS_ON', {'id': 'db-service'}), 'n': {'id': 'db-service'}}, {'service': {'id': 'app-service'}, 'r1': ({'id': 'app-service'}, 'CHECK_QUERY', {'id': 'select * from table1S where value=pending'}), 'n': {'id': 'select * from table1S where value=pending'}}, {'service': {'id': 'app-service'}, 'r1': ({'id': 'app-service'}, 'UPDATE_QUERY', {'id': 'update table table1S set value=init where value=pending'}), 'n': {'id': 'update table table1S set value=init where value=pending'}}, {'service': {'id': 'app-service'}, 'r1': ({'id': 'app-service'}, 'HEALTH_URL', {'id': 'http://app-service/health'}), 'n': {'id': 'http://app-service/health'}}, {'service': {'id': 'app-service'}, 'r1': ({'id': 'app-service'}, 'HEALTH_URL', {'id': 'http://app-service/health'}), 'n': {'id': 'http://app-service/health'}}, {'service': {'id': 'app-service'}, 'r1': ({'id': 'app-service'}, 'START_COMMAND', {'id': 'java -jar app-service.jar'}), 'n': {'id': 'java -jar app-service.jar'}}, {'service': {'id': 'app-service'}, 'r1': ({'id': 'app-service'}, 'CHECK_QUERY', {'id': 'select'}), 'n': {'id': 'select'}}]"}
        r1 = record.get('r1')
        n = record.get('n', {})
        
        if r1 and n:
            rel_type = r1[1]
            value = n.get('id', '')
            
            if rel_type and value:
                if rel_type in result_dict:
                    result_dict[rel_type].append(value)
                else:
                    result_dict[rel_type] = [value]
    
    return json.dumps(result_dict)


class FailedComponent(BaseModel):
    component_name: str = Field(description="Name of the failed component")
    component_type: str = Field(description="Type: 'Service' or 'Functionality'")
    reason_for_failure: str = Field(description="Primary reason for failure")


class ErrorSummaryResponse(BaseModel):
    failed_components: List[FailedComponent] = Field(description="List of failed components")




def validate_error_summary(json_str: str) -> str:
    try:
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        data = json.loads(json_str)
        ErrorSummaryResponse(**data)
        return json.dumps(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logging.error(f"JSON validation failed: {e}")
        return "{}"
    except Exception as e:
        logging.error(f"Error validating JSON: {e}")
        return "{}"


import pprint
import operator
from typing import Annotated


# Load environment variables from .env file (same as other files in project)
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Import chat mode constants
from src.shared.constants import CHAT_DEFAULT_MODE, CHAT_VECTOR_MODE, CHAT_GRAPH_MODE
from src.llm import get_llm

# Upload configuration
CHUNK_NUMBER = os.getenv("CHUNK_NUMBER", "1")
TOTAL_CHUNKS = os.getenv("TOTAL_CHUNKS", "1")
UPLOAD_MODEL = os.getenv("UPLOAD_MODEL", "UPLOAD_MODEL")
CYPHER_MODEL = os.getenv("CYPHER_MODEL", "CYPHER_MODEL")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "ANALYSIS_MODEL")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "SUMMARY_MODEL")

# Schema configuration from env
LOGS_SCHEMA_JSON = os.getenv("LOGS_SCHEMA", "{}")
HEALTH_SCHEMA_JSON = os.getenv("HEALTH_SCHEMA", "{}")
ADDITIONAL_INSTRUCTIONS = os.getenv("ADDITIONAL_INSTRUCTIONS", "")


def merge_error_context(existing, new):
    if existing is None:
        return new
    if new is None:
        return existing
    for key, value in new.items():
        if key in existing:
            existing[key].extend(value)
        else:
            existing[key] = value
    return existing


class AgentState(MessagesState):
    insertedDtls:  Annotated[list, operator.add]
    extractedErrorContext: Annotated[dict, merge_error_context]
    errorSummary: str
    failed_components: list
    chatbot_results: Annotated[list, operator.add]
    analysis_result: Annotated[list, operator.add]
    raw_solution: str
    validated_solution: str
    nodedetails: Annotated[list, operator.add]
    sources: Annotated[list, operator.add]
    entities: Annotated[list, operator.add]
    model: Annotated[str, lambda x, y: y if y else x]
    total_tokens: Annotated[int, lambda x, y: (x or 0) + (y or 0)]
    response_time: Annotated[int, lambda x, y: (x or 0) + (y or 0)]
    display_markdown: str

class Component(TypedDict):
    component_dtls: str
    component_idx: int

# class extractedErrorContextState(MessagesState):
#     extractedErrorContext:  Annotated[list, operator.add]

# class AnalysisAndSolutionResult(MessagesState):
#     analysis_result: str

def parse_logs_schema(logs_schema_json: str) -> tuple:
    """
    Parse LOGS_SCHEMA from env and extract allowedNodes and allowedRelationship.
    
    Expected format:
    LOGS_SCHEMA={"schema": "logjson", "triplet": ["Level-HAS_LOG->Message", "Source-RELATION->Target"]}
    
    Returns:
        tuple: (allowedNodes, allowedRelationship)
    """
    try:
        schema_data = json.loads(logs_schema_json)
        triplets = schema_data.get("triplet", [])
        
        if not triplets:
            log_agent("No triplets found in LOGS_SCHEMA", "parse_logs_schema")
            return "", ""
        
        node_labels = set()
        relationships = []
        
        for triplet in triplets:
            match = re.match(r'(.*?)-([A-Z_]+)->(.*)', triplet)
            if match:
                source, relation, target = match.groups()
                source = source.strip()
                relation = relation.strip()
                target = target.strip()
                
                node_labels.add(source)
                node_labels.add(target)
                relationships.append(f"{source},{relation},{target}")
        
        allowed_nodes = ",".join(sorted(node_labels))
        allowed_relationships = ",".join(relationships)
        
        log_agent(f"Parsed LOGS_SCHEMA: allowedNodes={allowed_nodes}, allowedRelationships={allowed_relationships}", "parse_logs_schema")
        
        return allowed_nodes, allowed_relationships
        
    except json.JSONDecodeError as e:
        log_agent(f"Failed to parse LOGS_SCHEMA JSON: {str(e)}", "parse_logs_schema", "ERROR")
        return "", ""
    except Exception as e:
        log_agent(f"Error parsing LOGS_SCHEMA: {str(e)}", "parse_logs_schema", "ERROR", traceback.format_exc())
        return "", ""

# Simple logger implementation without external dependencies
class CustomLogger:
    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_struct(self, message, severity="INFO"):
        if message is None:
            return
        level = getattr(logging, severity.upper(), logging.INFO)
        msg_str = str(message) if isinstance(message, dict) else str(message)
        self.logger.log(level, msg_str)

logger = CustomLogger()


def log_agent(message: str, method_name: str, severity: str = "INFO", stacktrace: str = None):
    """Helper function to log with methodName included.
    
    Args:
        message: The log message
        method_name: The name of the method function
        severity: Log level (INFO, ERROR, DEBUG, etc.)
        stacktrace: Optional stacktrace string to include
    """
    log_data = {"methodName": method_name, "message": message }
    if stacktrace:
        log_data["stacktrace"] = stacktrace
    logger.log_struct(log_data, severity)


class OverAllState(BaseModel):
    issues: list = Field(default_factory=list)
    total_services: int = 0
    healthy_services: int = 0
    unhealthy_services: int = 0


def check_issues_tool() -> dict:
    """Dummy call
    """
    return {
            "status": "Dummy"
        }


# Initialize tools
tools = [check_issues_tool]

# System message for the agent
sys_msg = SystemMessage(
    content="""You are a helpful assistant that monitors service health. 
You have access to tools to check the health status of services and identify any issues.
Use the check_issues_tool to get a summary of all services and identify any problems.
Use the get_health_status tool to get detailed health information for each service.
Always provide clear, actionable information about service health."""
)

# # Lazy initialization of LLM
# _llm = None
# _llm_with_tools = None

# def _get_llm():
#     """Lazy initialization of LLM to avoid API key requirement at import time."""
#     global _llm, _llm_with_tools
#     if _llm is None:
#         _llm = ChatOpenAI(model="gpt-4o")
#         _llm_with_tools = _llm.bind_tools(tools, parallel_tool_calls=False)
#     return _llm_with_tools

async def insertLogsData(state: AgentState):
    log_agent("insertLogsData node entered", "insertLogsData")
    
    try:
        # Get SERVICES_TO_MONITOR from environment
        services_config = os.getenv("SERVICES_TO_MONITOR", "[]")
        try:
            services = json.loads(services_config)
        except json.JSONDecodeError as err:
            log_agent(f"Invalid JSON in SERVICES_TO_MONITOR: {services_config}", "insertLogsData", "ERROR")
            return {"extractedErrorContext": {}, "messages": [{"role": "system", "content": f"Invalid SERVICES_TO_MONITOR config: {str(err)}"}]}
        
        if not isinstance(services, list):
            return {"extractedErrorContext": {}, "messages": [{"role": "system", "content": "SERVICES_TO_MONITOR must be a JSON array"}]}
        
        log_agent(f"Found {len(services)} services to process", "insertLogsData")
        
        # Get Neo4j credentials from environment
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Parse LOGS_SCHEMA from env to extract allowedNodes and allowedRelationship
        allowed_nodes, allowed_relationships = parse_logs_schema(LOGS_SCHEMA_JSON)
        
        log_agent(f"Using schema - allowedNodes: '{allowed_nodes}', allowedRelationships: '{allowed_relationships}'", "insertLogsData")
        
        all_results = []
        
        # Process each service
        for service in services:
            service_name = service.get("serviceName", "Unknown")
            log_file_path = service.get("logFilePath", "")
            
            if not log_file_path:
                log_agent(f"No logFilePath for service: {service_name}, skipping", "insertLogsData")
                all_results.append({
                    "service": service_name,
                    "status": "skipped",
                    "reason": "No logFilePath specified"
                })
                continue
            
            # Resolve absolute path or relative to myagent folder
            if not os.path.isabs(log_file_path):
                myagent_dir = os.path.dirname(os.path.abspath(__file__))
                log_file_path = os.path.join(myagent_dir, log_file_path)
            
            log_agent(f"Processing file: {log_file_path} for service: {service_name}", "insertLogsData")
            
            try:
                # Read file content
                with open(log_file_path, 'rb') as f:
                    file_content = f.read()
                
                # Get filename from path
                file_name = os.path.basename(log_file_path)
                
                files = {
                    'file': (file_name, BytesIO(file_content), 'text/plain')
                }
                log_agent(f"Using CYPHER_MODEL: {CYPHER_MODEL}", "insertLogsData")

                data = {
                    'chunkNumber': CHUNK_NUMBER,
                    'totalChunks': TOTAL_CHUNKS,
                    'originalname': file_name,
                    'model': CYPHER_MODEL
                }
                
                # Make async call to upload endpoint
                response = await asyncio.to_thread(
                    requests.post,
                    f"{API_BASE_URL}/upload",
                    files=files,
                    data=data,
                    timeout=900
                )
                response.raise_for_status()
                upload_result = response.json()
                
                log_agent(f"upload_large_file_into_chunks response: {upload_result}", "insertLogsData")
                
                # After successful upload, call extract API
                log_agent("Calling extract API after successful upload", "insertLogsData")
                
                # Extract API expects form data (multipart/form-data)
                extract_data = {
                    'uri': neo4j_uri,
                    'userName': neo4j_username,
                    'password': neo4j_password,
                    'database': neo4j_database,
                    'model': CYPHER_MODEL,
                    'source_type': 'local file',
                    'file_name': file_name,
                    'retry_condition': '',
                    'token_chunk_size': 100,
                    'chunk_overlap': 20,
                    'chunks_to_combine': 1,
                    'allowedNodes': allowed_nodes,
                    'allowedRelationship': allowed_relationships,
                    'additional_instructions': ADDITIONAL_INSTRUCTIONS,
                    'embedding_provider': os.getenv("EMBEDDING_PROVIDER", "sentence-transformer"),
                    'embedding_model': os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                }
                
                extract_response = await asyncio.to_thread(
                    requests.post,
                    f"{API_BASE_URL}/extract",
                    data=extract_data,
                    timeout=900
                )
                extract_response.raise_for_status()
                extract_result = extract_response.json()
                
                log_agent(f"extract API response: {extract_result}", "insertLogsData")
                
                all_results.append({
                    "service": service_name,
                    "status": "success",
                    "upload_result": upload_result,
                    "extract_result": extract_result
                })
                
            except FileNotFoundError:
                log_agent(f"File not found: {log_file_path}", "insertLogsData", "ERROR")
                all_results.append({
                    "service": service_name,
                    "status": "failed",
                    "reason": f"File not found: {log_file_path}"
                })
            except Exception as e:
                log_agent(f"Error processing {service_name}: {str(e)}", "insertLogsData", "ERROR", traceback.format_exc())
                all_results.append({
                    "service": service_name,
                    "status": "failed",
                    "reason": str(e)
                })
        
        log_agent(f"insertLogsData node exiting with {len(all_results)} results", "insertLogsData")
        return {
            "insertedDtls": all_results,
            "messages": [
                {"role": "system", "content": f"Processed {len(services)} services. Results: {json.dumps(all_results)}"}
            ]
        }
    except Exception as e:
        log_agent(f"Error in upload/extract process: {str(e)}", "insertLogsData", "ERROR", traceback.format_exc())
        log_agent("insertLogsData node exiting with error", "insertLogsData", "ERROR")
        return {"insertedDtls": [], "messages": [{"role": "system", "content": f"Process failed: {str(e)}"}]}


def insertHealthUrlData(state: AgentState):
    log_agent("insertHealthUrlData node entered", "insertHealthUrlData")
    try:
        # Step 1: Get health data from API
        response = requests.get(f"{API_BASE_URL}/get-service-health", timeout=900)
        response.raise_for_status()
        result = response.json()
        
        log_agent(f"Health status response: {result}", "insertHealthUrlData")
        
        # Step 2: Save result to file
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        health_file_path = os.path.join(logs_dir, 'healthData.txt')
        
        with open(health_file_path, 'w') as f:
            f.write(json.dumps(result, indent=2))
        
        log_agent(f"Saved health data to: {health_file_path}", "insertHealthUrlData")
        
        # Step 3: Parse HEALTH_SCHEMA
        allowed_nodes, allowed_relationships = parse_logs_schema(HEALTH_SCHEMA_JSON)
        
        log_agent(f"Using HEALTH schema - allowedNodes: '{allowed_nodes}', allowedRelationships: '{allowed_relationships}'", "insertHealthUrlData")
        
        # Step 4: Call upload API
        file_name = 'healthData.txt'
        with open(health_file_path, 'rb') as f:
            file_content = f.read()
        
        files = {'file': (file_name, BytesIO(file_content), 'text/plain')}
        upload_data = {
            'chunkNumber': CHUNK_NUMBER,
            'totalChunks': TOTAL_CHUNKS,
            'originalname': file_name,
            'model': CYPHER_MODEL
        }
        
        log_agent(f"Uploading health data with model: {CYPHER_MODEL}", "insertHealthUrlData")
        upload_response = requests.post(f"{API_BASE_URL}/upload", files=files, data=upload_data, timeout=900)
        upload_response.raise_for_status()
        upload_result = upload_response.json()
        
        log_agent(f"Upload response: {upload_result}", "insertHealthUrlData")
        
        # Step 5: Call extract API
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        extract_data = {
            'uri': neo4j_uri,
            'userName': neo4j_username,
            'password': neo4j_password,
            'database': neo4j_database,
            'model': CYPHER_MODEL,
            'source_type': 'local file',
            'file_name': file_name,
            'retry_condition': '',
            'token_chunk_size': 100,
            'chunk_overlap': 20,
            'chunks_to_combine': 1,
            'allowedNodes': allowed_nodes,
            'allowedRelationship': allowed_relationships,
            'additional_instructions': ADDITIONAL_INSTRUCTIONS,
            'embedding_provider': os.getenv("EMBEDDING_PROVIDER", "sentence-transformer"),
            'embedding_model': os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        }
        
        log_agent("Calling extract API for health data", "insertHealthUrlData")
        extract_response = requests.post(f"{API_BASE_URL}/extract", data=extract_data, timeout=900)
        extract_response.raise_for_status()
        extract_result = extract_response.json()
        
        log_agent(f"Extract response: {extract_result}", "insertHealthUrlData")
        
        # Step 6: Return combined results
        log_agent("insertHealthUrlData node exiting successfully", "insertHealthUrlData")
        return {
            "insertedDtls": [{
                "health_data": result,
                "upload_result": upload_result,
                "extract_result": extract_result
            }],
            "messages": [{"role": "system", "content": f"Health data processed successfully"}]
        }
    except Exception as e:
        log_agent(f"Error in getHealthUrlData: {str(e)}", "insertHealthUrlData", "ERROR", traceback.format_exc())
        log_agent("insertHealthUrlData node exiting with error", "insertHealthUrlData", "ERROR")
        return {"insertedDtls": [], "messages": [{"role": "system", "content": f"Error: {str(e)}"}]}


def getLogsErrors(state: AgentState):
    log_agent("getLogsErrors node entered", "getLogsErrors")
    log_agent(f"Using CYPHER_MODEL: {CYPHER_MODEL}", "getLogsErrors")

    try:
        # Get Neo4j credentials from environment
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        query = """
        MATCH (level:Level)-[r1:HAS_LOG]->(n)
        RETURN level,r1,n
        """

        neo4j_query_result = execute_neo4j_query(query)
        log_agent(f"Neo4j query result for : {neo4j_query_result}", "getLogsErrors")
        
        source_of_truth_data = parse_source_of_truth_response(neo4j_query_result, "logs")
        log_agent(f"Parsed source of truth data: {source_of_truth_data}", "getLogsErrors")
        

        log_agent("getLogsErrors node exiting successfully", "getLogsErrors")
        return {
            "extractedErrorContext": {"ErrorLogs": [source_of_truth_data]},
            "messages": [
                {"role": "system", "content": source_of_truth_data}
            ]
        }
        
    except Exception as e:
        log_agent(f"Error in getErrors: {str(e)}", "getLogsErrors", "ERROR", traceback.format_exc())
        log_agent("getLogsErrors node exiting with error", "getLogsErrors", "ERROR")
        return {"messages": [{"role": "system", "content": f"Error analysis failed: {str(e)}"}]}


def getHealthErrors(state: AgentState):
    log_agent("getHealthErrors node entered", "getHealthErrors")
    log_agent(f"Using CYPHER_MODEL: {CYPHER_MODEL}", "getHealthErrors")

    try:
         
        # Get Neo4j credentials from environment
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        
        query = """
        MATCH (service:Service)-[r1:CURRENT_STATUS]->(n:Status {id :"Stopped"})
        RETURN service,r1,n
        """

        neo4j_query_result = execute_neo4j_query(query)
        log_agent(f"Neo4j query result for : {neo4j_query_result}", "getHealthErrors")
        
        source_of_truth_data = parse_source_of_truth_response(neo4j_query_result, "health")
        log_agent(f"Parsed source of truth data: {source_of_truth_data}", "getHealthErrors")
        

        log_agent("getHealthErrors node exiting successfully", "getHealthErrors")
        return {
            "extractedErrorContext": {"StoppedServices": [source_of_truth_data]},
            "messages": [
                {"role": "system", "content": source_of_truth_data}
            ]
        }
        
    except Exception as e:
        log_agent(f"Error in getErrors: {str(e)}", "getHealthErrors", "ERROR", traceback.format_exc())
        log_agent("getHealthErrors node exiting with error", "getHealthErrors", "ERROR")
        return {"messages": [{"role": "system", "content": f"Error analysis failed: {str(e)}"}]}


def errorSummary(state: AgentState):
    log_agent("errorSummary node entered", "errorSummary")
    log_agent(f"Current MessageState: {state}", "errorSummary")

    try:
        services_config = os.getenv("SERVICES_TO_MONITOR", "[]")
        try:
            services = json.loads(services_config)
        except json.JSONDecodeError:
            log_agent(f"Invalid JSON in SERVICES_TO_MONITOR: {services_config}", "errorSummary", "ERROR")
            return {"messages": [{"role": "system", "content": f"Invalid SERVICES_TO_MONITOR config"}]}
        
        if not isinstance(services, list):
            return {"messages": [{"role": "system", "content": "SERVICES_TO_MONITOR must be a JSON array"}]}
        
        document_names = []
        for service in services:
            log_file_path = service.get("logFilePath", "")
            if log_file_path:
                file_name = os.path.basename(log_file_path)
                document_names.append(file_name)
        
        if not document_names:
            return {"messages": [{"role": "system", "content": "No log files found in SERVICES_TO_MONITOR"}]}
        document_names=[]
        log_agent(f"Document names for chat: {document_names}", "errorSummary")


        valid_service_names = []
        for service in services:
            service_name = service.get("serviceName", "")
            if service_name:
                valid_service_names.append(service_name)
        
        if not valid_service_names:
            return {"messages": [{"role": "system", "content": "No services found in SERVICES_TO_MONITOR"}]}
        log_agent(f"valid_service_names from config: {valid_service_names}", "errorSummary")


        try:
            llm, _, _ = get_llm(model=SUMMARY_MODEL)

            goal = """Analyze each service status in stopped StoppedServices and each log ErrorLogs in below Findings to identify failed components with their UNIQUE component_name, component_type and reason_for_failure(s). Respond ONLY with valid JSON in this format without code-blocks, no explanations or surrounding text: {"failed_components": [{"component_name": "...", "component_type": "...", "reason_for_failure": "..."}]}"""

            prompt = (
            f"-----\n## Goal:\n {goal}\n"
            f"-----\n## Fact: \n"
            f"- Extract component names from BOTH 'StoppedServices' table AND 'ErrorLogs' messages.\n"
            f"- If extracted component_name matches ANY name in valid_service_names list below → component_type = 'Service'.\n"
            f"- If NOT found in valid_service_names → component_type = 'Functionality'.\n"
            f"- valid_service_names are: {json.dumps(valid_service_names)}\n"
            f"-----\n## Findings:\n{state['extractedErrorContext']}\n"
            )
            log_agent(f"Prompt: {prompt}", "errorSummary")
            print("prrrrrrrrrrrrrr",prompt)

            

            response = llm.invoke(prompt)
            
            extracted_json = response.content
            log_agent(f"Extracted JSON for failed Component: {extracted_json}", "errorSummary")
            validated_json = validate_error_summary(extracted_json)
        except Exception as e:
            log_agent(f"Failed to extract JSON from prevMessageOut: {str(e)}", "errorSummary", "ERROR", traceback.format_exc())
            validated_json = "{}"

        return {"errorSummary": validated_json}
    except Exception as e:
        log_agent(f"Error in errorSummary: {str(e)}", "errorSummary", "ERROR", traceback.format_exc())
        return {"errorSummary": "{}"}


def component_processor(state: AgentState):
    log_agent("component_processor node entered", "component_processor")
    
    try:
        error_summary_str = state.get("errorSummary", "{}")
        error_summary = json.loads(error_summary_str) if error_summary_str else {}
        failed_components = error_summary.get("failed_components", [])
        
        log_agent(f"Failed components: {failed_components}", "component_processor")
        
        if not failed_components:
            return "empty"
        
        return [
            Send(
                "analysisAndSolution",
                {
                    "component": component,
                    "component_idx": idx,
                }
            )
            for idx, component in enumerate(failed_components)
        ]
    except Exception as e:
        log_agent(f"Error in component_processor: {str(e)}", "component_processor", "ERROR", traceback.format_exc())
        return "empty"


def analysisAndSolution(state: Component):
    log_agent("analysisAndSolution node entered", "analysisAndSolution")
    log_agent(f"Current MessageState: {state}", "analysisAndSolution")

    try:
        component = state["component"]
        
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

        component_name = component.get("component_name", "unknown")
        component_type = component.get("component_type", "unknown")
        reason_for_failure = component.get("reason_for_failure", "unknown")
        
        log_agent(f"Processing component {state["component_idx"]+1}: {component_name}", "analysisAndSolution")
        
        query = """
        MATCH (service:Service {id: $component_name})-[r1]->(n), 
              (chunk:Chunk {fileName: "sourceOfTruth_text.txt"})-[r2]->(n) 
        RETURN service, r1, n
        """

        neo4j_query_result = execute_neo4j_query(query, {"component_name": component_name})
        log_agent(f"Neo4j query result for {component_name}: {neo4j_query_result}", "analysisAndSolution")
        
        source_of_truth_data = parse_source_of_truth_response(neo4j_query_result,"service")
        log_agent(f"Parsed source of truth data: {source_of_truth_data}", "analysisAndSolution")
        
        # Extract fields from chat_bot response
        answer_message = ""
        nodedetails = {}
        sources = []
        entities = []
        model = ""
        total_tokens = 0
        response_time = 0
        
        # if chat_result.get("status") == "Success" and chat_result.get("data"):
        #     data = chat_result["data"]
        #     if data.get("info") and data["info"].get("metric_details") and data["info"]["metric_details"].get("answer"):
        #         answer_message = data["info"]["metric_details"]["answer"]
        #     else:
        #         answer_message = data.get("message", "")
        #     if data.get("info") and data["info"].get("nodedetails"):
        #         nodedetails = data["info"]["nodedetails"]
        #     if data.get("info"):
        #         sources = data["info"].get("sources", [])
        #         entities = data["info"].get("entities", [])
        #         model = data["info"].get("model", "")
        #         total_tokens = data["info"].get("total_tokens", 0)
        #         response_time = data["info"].get("response_time", 0)
        
        # # Accumulate results
        # # all_answer_messages.append(answer_message)
        # # all_nodedetails.append(nodedetails)
        # # all_sources.extend(sources)
        # # all_entities.extend(entities)
        # # if model:
        # #     combined_model = model
        # # combined_total_tokens += total_tokens
        # # combined_response_time += response_time
        
        # # log_agent(f"Processed {component_name}: answer_message={answer_message[:100]}...", "analysisAndSolution")
        
        # # # Combine all results
        # # # answer_message = "\n\n---\n\n".join(all_answer_messages)
        # # # nodedetails = all_nodedetails
        # # # sources = all_sources
        # # # entities = all_entities
        # # model = combined_model
        # # total_tokens = combined_total_tokens
        # # response_time = combined_response_time
        
        # # log_agent(f"analysisAndSolution combined answer_message response: {answer_message[:200]}...", "analysisAndSolution")
        # # log_agent(f"analysisAndSolution combined nodedetails: {nodedetails}", "analysisAndSolution")
        # # log_agent(f"analysisAndSolution combined sources: {sources}", "analysisAndSolution")
        
        # # log_agent("analysisAndSolution node exiting successfully", "analysisAndSolution")
        
        # Add component details to source_of_truth_data
        source_of_truth_dict = json.loads(source_of_truth_data) if source_of_truth_data else {}
        source_of_truth_dict['component_name'] = component_name
        source_of_truth_dict['component_type'] = component_type
        source_of_truth_dict['reason_for_failure'] = reason_for_failure
        source_of_truth_data = json.dumps(source_of_truth_dict)
        
        return {
            "analysis_result": [source_of_truth_data],
            "nodedetails": [nodedetails],
            "sources": [sources],
            "entities": [entities],
            "model": model,
            "total_tokens": total_tokens,
            "response_time": response_time,
            "messages": [{"role": "system", "content": source_of_truth_data}]
        }
    # return {"extractedErrorContext": []}       
    except Exception as e:
        log_agent(f"Error in getErrors: {str(e)}", "analysisAndSolution", "ERROR", traceback.format_exc())
        log_agent("analysisAndSolution node exiting with error", "analysisAndSolution", "ERROR")
        return {
            "extractedErrorContext": {},
            "nodedetails": [],
            "sources": [],
            "entities": [],
            "model": "",
            "total_tokens": 0,
            "response_time": 0,
            "messages": [{"role": "system", "content": f"Error analysis failed: {str(e)}"}]
        }

def convertToMarkdown(state: AgentState):
    log_agent("convertToMarkdown node entered", "convertToMarkdown")
    
    try:
        analysis_result_list = state.get("analysis_result", [])
        
        if not analysis_result_list:
            return {"display_markdown": "No analysis results found."}
        
        # Check if already markdown format
        if len(analysis_result_list) == 1 and (analysis_result_list[0].startswith("##") or analysis_result_list[0].startswith("---")):
            return {"display_markdown": analysis_result_list[0]}
        
        # Parse each result individually and merge failed_component_dtls
        all_failed_services = []
        markdown_results = []
        
        for result_str in analysis_result_list:
            if result_str.startswith("##") or result_str.startswith("---"):
                markdown_results.append(result_str)
                continue
            
            try:
                data = json.loads(result_str)
                services = data.get("failed_component_dtls", [])
                if services:
                    all_failed_services.extend(services)
                    
                # Check for source_of_truth format with component_name, etc.
                if "serviceName" in data or "health_status" in data or "log_level" in data or "component_name" in data:
                    md = "## Component Analysis\n\n"
                    
                    # Add component details if present
                    if "component_name" in data:
                        md += f"**Component Name:** {data['component_name']}\n\n"
                    if "component_type" in data:
                        md += f"**Component Type:** {data['component_type']}\n\n"
                    if "reason_for_failure" in data:
                        md += f"**Reason for Failure:** {data['reason_for_failure']}\n\n"
                    
                    # Add other key-value pairs
                    for key, value in data.items():
                        if key in ['component_name', 'component_type', 'reason_for_failure']:
                            continue
                        title = key.replace('_', ' ').title()
                        md += f"**{title}:**\n"
                        if isinstance(value, list):
                            for item in value:
                                md += f"- {item}\n"
                        elif value:
                            md += f"- {value}\n"
                        else:
                            md += "- None\n"
                        md += "\n"
                    markdown_results.append(md)
                    continue
            except (json.JSONDecodeError, Exception) as e:
                log_agent(f"Error parsing individual result: {str(e)}", "convertToMarkdown", "ERROR")
                markdown_results.append(result_str)
        
        # If we have merged services, create markdown
        if all_failed_services:
            md = "## Failed Services Details\n\n"
            for idx, service in enumerate(all_failed_services, 1):
                md += f"### {idx}. {service.get('component_name', 'N/A')}\n"
                for key, value in service.items():
                    if key in ['name', 'component_name']:
                        continue
                    title = key.replace('_', ' ').title()
                    md += f"**{title}:**\n"
                    if isinstance(value, list):
                        for item in value:
                            md += f"- `{item}`\n"
                    elif value:
                        md += f"- {value}\n"
                    else:
                        md += "- None\n"
                    md += "\n"
            log_agent(f"convertToMarkdown output: {md}", "convertToMarkdown")
            return {"display_markdown": md}
        
        # If no services found but have markdown results, join them
        if markdown_results:
            combined = "\n\n---\n\n".join(markdown_results)
            return {"display_markdown": combined}
        
        # Fallback: join all results
        combined_result = "\n\n---\n\n".join(analysis_result_list)
        return {"display_markdown": combined_result}
        
    except Exception as e:
        log_agent(f"Error in convertToMarkdown: {str(e)}", "convertToMarkdown", "ERROR", traceback.format_exc())
        return {"display_markdown": ""}


# Build the state graph
builder = StateGraph(AgentState, output_schema=AgentState)
   
builder.add_node("insertLogsData", insertLogsData)
builder.add_node("insertHealthUrlData", insertHealthUrlData)
builder.add_node("getLogsErrors", getLogsErrors)
builder.add_node("getHealthErrors", getHealthErrors)

builder.add_node("analysisAndSolution", analysisAndSolution)
builder.add_node("errorSummary", errorSummary)
#builder.add_node("tools", ToolNode(tools))
#builder.add_edge(START, "getHealthUrlData")
#builder.add_edge(START, "getLogsData")
#builder.add_edge("getLogsData", "getErrors")
#builder.add_edge("getHealthUrlData", "getErrors")


# builder.add_conditional_edges(
#     "analysisAndSolution",
#     tools_condition,
# )
# builder.add_edge("tools", "analysisAndSolution")

# builder.add_edge("getErrors", "analysisAndSolution")
# builder.add_edge("analysisAndSolution", "outputDiagnosisAndSolution")
# builder.add_edge("outputDiagnosisAndSolution", "getExecutionApproval")
# builder.add_edge("outputDiagnosisAndSolution", END)

# builder.add_edge("getExecutionApproval", END)
#builder.add_edge(START, "getErrors")



# Add entry point and edges
# Run getLogsData first to upload and extract log files
##builder.add_edge(START, "getLogsData")
##builder.add_edge(START, "getHealthUrlData")

# After logs are processed, run error analysis
##builder.add_edge("getLogsData", "getErrors")
##builder.add_edge("getHealthUrlData", "insertHealthUrlData")

#builder.add_edge(START, "insertLogsData")
#builder.add_edge(START, "insertHealthUrlData")

#builder.add_edge("insertLogsData", "getLogsErrors")
#builder.add_edge("insertHealthUrlData", "getHealthErrors")

builder.add_edge(START, "getLogsErrors")
builder.add_edge(START, "getHealthErrors")
builder.add_edge("getLogsErrors", "errorSummary")
builder.add_edge("getHealthErrors", "errorSummary")

##builder.add_node("component_processor", component_processor)
builder.add_node("convertToMarkdown", convertToMarkdown)

builder.add_conditional_edges(
    "errorSummary",
    component_processor,
    {
        "empty": END,
        "analysisAndSolution": "analysisAndSolution"
    }
)

builder.add_edge("analysisAndSolution", "convertToMarkdown")
builder.add_edge("convertToMarkdown", END)
##builder.add_edge("getErrors", END)


# Compile the graph
react_graph = builder.compile()


def generate_graph_diagram(output_path: str = "./agent_graph_diagram.png") -> None:
    """Generate and save the Mermaid diagram of the agent graph.
    This function can be called independently without initializing the LLM.
    
    Args:
        output_path: Path to save the PNG file. Defaults to ./agent_graph_diagram.png
    """
    log_agent("Generating graph Mermaid diagram...", "generate_graph_diagram")
    
    # Get Mermaid code
    try:
        mermaid_code = react_graph.get_graph(xray=True).draw_mermaid()
        log_agent(f"MERMAID CODE:\n{mermaid_code}", "generate_graph_diagram")
    except Exception as e:
        log_agent(f"Error generating Mermaid code: {str(e)}", "generate_graph_diagram", "ERROR", traceback.format_exc())
        mermaid_code = ""
    
    # Generate PNG
    try:
        png_bytes = react_graph.get_graph(xray=True).draw_mermaid_png()
        with open(output_path, 'wb') as f:
            f.write(png_bytes)
        log_agent(f"Graph diagram saved to: {output_path}, File size: {len(png_bytes)} bytes", "generate_graph_diagram")
    except Exception as e:
        log_agent(f"Error generating PNG: {str(e)}", "generate_graph_diagram", "ERROR", traceback.format_exc())


def get_graph_mermaid_diagram(output_path: str = "") -> bytes:
    """Generate a Mermaid diagram of the agent graph.

    Args:
        output_path: Optional path to save the PNG file. If empty, returns the bytes.

    Returns:
        PNG image bytes of the Mermaid diagram.
    """
    try:
        # Get the graph visualization as Mermaid PNG
        mermaid_png = react_graph.get_graph(xray=True).draw_mermaid_png()

        if output_path:
            with open(output_path, 'wb') as f:
                f.write(mermaid_png)
            log_agent(f"Mermaid diagram saved to: {output_path}", "get_graph_mermaid_diagram")

        return mermaid_png
    except Exception as e:
        log_agent(f"Error generating Mermaid diagram: {str(e)}", "get_graph_mermaid_diagram", "ERROR", traceback.format_exc())
        return b""


def get_graph_mermaid_code() -> str:
    """Generate Mermaid code (text format) of the agent graph.
    
    Returns:
        Mermaid code string representing the graph structure.
    """
    try:
        mermaid_code = react_graph.get_graph(xray=True).draw_mermaid()
        return mermaid_code
    except Exception as e:
        log_agent(f"Error generating Mermaid code: {str(e)}", "get_graph_mermaid_code", "ERROR", traceback.format_exc())
        return ""


async def run_agent(query: str = "Check for issues with all services") -> dict:
    """Run the agent with a query.
    
    Args:
        query: The user's query about service health.
        
    Returns:
        The final response from the agent including messages and diagnosis.
    """
    
    # Save mermaid diagram after agent execution
    generate_graph_diagram(output_path="agent_graph_diagram.png")

    log_agent(f"In run_agent with query: {query}", "run_agent")
    messages = [HumanMessage(content=query)]
    state: MessagesState = {"messages": messages}
    result = await react_graph.ainvoke(state)
    log_agent(f"query result: {result}", "run_agent")
        
    # Format the response
    response = {
        "status": "Success",
        "query": query,
        "messages": []
    }
    
    # Convert messages to serializable format
    for msg in result.get("messages", []):
        if hasattr(msg, 'content'):
            response["messages"].append({
                "type": type(msg).__name__,
                "content": msg.content
            })
    
    # Add diagnosis if available
    if "diagnosis" in result:
        response["diagnosis"] = result["diagnosis"]
    
    # Add tool calls result if available
    if "services" in result:
        response["services"] = result["services"]
        response["total_services"] = result.get("total_services", 0)
        response["healthy_services"] = result.get("healthy_services", 0)
        response["unhealthy_services"] = result.get("unhealthy_services", 0)
    
    # Add analysis_result if available (maps to diagnosis for frontend)
    if "analysis_result" in result:
        response["diagnosis"] = result["analysis_result"]
    
    # Combine/merge nodedetails from list to object (for Details modal)
    if "nodedetails" in result and result["nodedetails"]:
        combined_nodedetails = {"chunkdetails": [], "entitydetails": {}, "communitydetails": []}
        for nd in result["nodedetails"]:
            if isinstance(nd, dict):
                if nd.get("chunkdetails"):
                    combined_nodedetails["chunkdetails"].extend(nd["chunkdetails"])
                if nd.get("entitydetails"):
                    combined_nodedetails["entitydetails"].update(nd["entitydetails"])
                if nd.get("communitydetails"):
                    combined_nodedetails["communitydetails"].extend(nd["communitydetails"])
        response["nodedetails"] = combined_nodedetails
    
    # Combine/merge sources from list (for Details modal - Sources tab)
    if "sources" in result and result["sources"]:
        combined_sources = []
        for s in result["sources"]:
            if isinstance(s, list):
                combined_sources.extend(s)
            elif s:
                combined_sources.append(s)
        response["sources"] = list(set(combined_sources))
    
    # Combine/merge entities from list (for Details modal - Entities tab)
    if "entities" in result and result["entities"]:
        combined_entities = {"entityids": [], "relationshipids": []}
        for e in result["entities"]:
            if isinstance(e, dict):
                if e.get("entityids"):
                    combined_entities["entityids"].extend(e["entityids"])
                if e.get("relationshipids"):
                    combined_entities["relationshipids"].extend(e["relationshipids"])
        response["entities"] = combined_entities
    
    # Add model if available (for Details modal display)
    if "model" in result:
        response["model"] = result["model"]
    
    # Add total_tokens if available (for Details modal display)
    if "total_tokens" in result:
        response["total_tokens"] = result["total_tokens"]
    
    # Add response_time if available (for Details modal display)
    if "response_time" in result:
        response["response_time"] = result["response_time"]
    
    # Add display_markdown if available (for frontend display)
    if "display_markdown" in result:
        response["display_markdown"] = result["display_markdown"]
    
    return response


if __name__ == "__main__":
    import sys
    
    # Generate Mermaid diagram first (before running agent that needs API key)
    log_agent("Generating Graph Visualization", "__main__")
    
    # Get Mermaid code
    mermaid_code = get_graph_mermaid_code()
    if mermaid_code:
        log_agent(f"Mermaid Code:\n{mermaid_code}", "__main__")
    
    # Save PNG diagram
    output_file = "agent_graph_diagram.png"
    png_bytes = get_graph_mermaid_diagram(output_path=output_file)
    if png_bytes:
        log_agent(f"Graph diagram saved as: {output_file}, File size: {len(png_bytes)} bytes", "__main__")
    
    log_agent("Running Agent (requires OpenAI API key)", "__main__")
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Check for issues with all services"
    
    log_agent(f"Running agent with query: {query}", "__main__")
    
    result = asyncio.run(run_agent(query))
    
    log_agent(f"Final Response:\n{json.dumps(result, indent=2)}", "__main__")



##MATCH (service:Service)-[r1:CURRENT_STATUS]->(n:Status {id :"Stopped"}), (service)-[r2:DEPENDS_ON]->(n2)
#RETURN service,r1,n,r2,n2

#MATCH (service:Service)-[r1:CURRENT_STATUS]->(n), (service2:Service)-[r2:DEPENDS_ON]->(n2)
#RETURN service,r1,n,r2,service2,n2

#MATCH (service:Service { id:"static-webapp"})-[r1]->(n), (chunk:Chunk {fileName:"sourceOfTruth_text.txt"})-[r2]->(n) 
#RETURN service,r1,n


#MATCH (level:Level)-[r1:HAS_LOG]->(n), (chunk:Chunk)-[r2:HAS_ENTITY]->(level)
#RETURN level,r1,n,chunk,r2

#MATCH (level:Level)-[r1:HAS_LOG]->(n)
#RETURN level,r1,n
