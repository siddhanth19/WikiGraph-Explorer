import dotenv
import streamlit as st
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import RunnablePassthrough

dotenv.load_dotenv()

# Initialize LLMs
groq_llm = ChatGroq(model='gemma2-9b-it')
llm = ChatOllama(model='llama3.1', temperature=0)
encoder = HuggingFaceEmbeddings()

class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(..., description="Extracted entity names from the text.")

# Prompt for extracting entities
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}")
])

chat_model = prompt | groq_llm.with_structured_output(Entities)

def load_data(entity):
    """Loads Wikipedia data for the given entity."""
    loader = WikipediaLoader(entity, load_max_docs=1)
    data = loader.load()

    for doc in data:
        del doc.metadata['summary']

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    docs = splitter.split_documents(data)
    return docs

def create_graph(docs):
    """Creates a Neo4j graph from processed documents."""
    graph = Neo4jGraph()
    llm_transformer = LLMGraphTransformer(llm=llm)

    try:
        graph_docs = llm_transformer.convert_to_graph_documents(docs)

        if not graph_docs or not any(doc.nodes for doc in graph_docs):
            raise ValueError("No valid nodes were generated. Check LLM output.")

        graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)
        return graph

    except Exception as e:
        print(f"Error during graph creation: {e}")
        return None

def create_vector_index():
    """Creates vector index for document retrieval."""
    graph = Neo4jGraph()
    vector_index = Neo4jVector.from_existing_graph(
        encoder, search_type='hybrid',
        node_label='Document', text_node_properties=['text'],
        embedding_node_property='embedding'
    )
    return vector_index

def generate_full_text_query(input: str) -> str:
    """Generates a full-text search query for better entity matching."""
    words = [el for el in remove_lucene_chars(input).split() if el]
    return " AND ".join([f"{word}~2" for word in words])

def structured_retriever(question: str) -> str:
    """Retrieves structured knowledge from Neo4j graph."""
    graph = Neo4jGraph()
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    result = ""
    entities = chat_model.invoke({"question": question})

    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50""",
            {"query": generate_full_text_query(entity)}
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str, vector_index):
    """Retrieves structured (graph) and unstructured (vector) knowledge."""
    if vector_index is None:
        raise ValueError("Vector index is not initialized. Create the graph first.")

    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]

    return f"Structured Data: {structured_data} \n Unstructured Data: {'#Document '.join(unstructured_data)}"

def answer_query(question: str, vector_index):
    """Answers queries using structured and unstructured retrieval and returns context."""
    if vector_index is None:
        raise ValueError("Vector index is not initialized. Please create the graph first.")

    template = """Answer the question based only on the given context:
    context: {context}
    question: {question}"""

    prompt = PromptTemplate.from_template(template)

    # ✅ Retrieve context separately
    retrieved_context = retriever(question, vector_index)

    # ✅ Ensure context is passed inside a dictionary to match LangChain expectations
    chain = (
        {
            "context": lambda x: retrieved_context,  # ✅ Wrap context retrieval in a callable
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)  # ✅ Pass only the question as input

    return answer, retrieved_context  # ✅ Return both answer and retrieved context


