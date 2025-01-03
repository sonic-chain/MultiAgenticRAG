### Build Index

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT


load_dotenv()

### from langchain_cohere import CohereEmbeddings

# Set embeddings
embd = OpenAIEmbeddings()

vectorstore = Chroma(
    collection_name="rag-chroma-google",
    embedding_function=embd,
    persist_directory='vector_db'
)

all_data = vectorstore.get(include=["documents", "metadatas"])

from langchain_core.documents import Document

# Assuming all_data contains your documents and metadatas
documents = []
for content, meta in zip(all_data["documents"], all_data["metadatas"]):
    # Ensure metadata is a dictionary; replace None with an empty dict
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise ValueError(f"Expected metadata to be a dict, but got {type(meta)}")
    documents.append(Document(page_content=content, metadata=meta))


retriever_BM25 = None

retriever_BM25 = BM25Retriever.from_documents(documents, search_kwargs={"k": 3})
retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.3, 0.3, 0.4]
)




from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import Send


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere



async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the question (a step in the research plan).

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(TypedDict):
        queries: list[str]

    print("GENERATE QUERIES")
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response["queries"]
    queries.append(state.question)
    print(f"Queries: {queries}")
    return {"queries": response["queries"]}


async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration with the retriever used to fetch documents.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    print("---RETRIEVING DOCUMENTS---")
    #https://www.kaggle.com/code/marcinrutecki/rag-ensemble-retriever-in-langchain
    #response = await ensemble_retriever.ainvoke(state.query)

    compressor = CohereRerank(top_n=2, model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    print(f"Query for the retrieval process: {state.query}")
    response = compression_retriever.invoke(state.query)
    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_and_rerank_documents", QueryState(query=query)) for query in state.queries
    ]



builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()
