import os
from neo4j_graphrag.experimental.components.schema import GraphSchema


def transformers_and_llms_schema() -> GraphSchema:
    path = os.path.join("schema", "transformers_and_LLMs.yaml")
    return GraphSchema.from_file(path)
    
