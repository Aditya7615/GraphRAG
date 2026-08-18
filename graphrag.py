"""GraphRAG pipeline — pure Python, no Streamlit dependency.

Usage:
    from graphrag import GraphRAG
    engine = GraphRAG()
    result = engine.ask("What is hybrid retrieval?")
    print(result["generation"])
"""

import os
import warnings
from collections import defaultdict
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# --------------- Config ---------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
CHROMA_WEIGHT = float(os.getenv("CHROMA_WEIGHT", "0.6"))

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


# --------------- Dotenv loader ---------------
def load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# --------------- Pydantic schemas ---------------
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' otherwise")


class GradeHallucination(BaseModel):
    binary_score: str = Field(description="'yes' if grounded, 'no' otherwise")


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search_needed: str


# --------------- Lightweight EnsembleRetriever ---------------
class EnsembleRetriever:
    """Combine multiple retrievers using Reciprocal Rank Fusion."""

    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    def invoke(self, query, **kwargs):
        scores = defaultdict(float)
        doc_map = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                docs = retriever.invoke(query, **kwargs)
            except Exception:
                docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                key = doc.page_content[:200]
                scores[key] += weight / (rank + 60)
                doc_map[key] = doc
        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in ranked]


# --------------- Helpers ---------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def get_llm():
    return ChatGroq(model="qwen/qwen3.6-27b", temperature=0)


def load_documents():
    """Load all saved PDFs and text files from data/."""
    docs = []
    if os.path.isdir(PDF_DIR):
        from langchain_community.document_loaders import PyPDFLoader
        for f in sorted(os.listdir(PDF_DIR)):
            if f.lower().endswith(".pdf"):
                try:
                    docs.extend(PyPDFLoader(os.path.join(PDF_DIR, f)).load())
                except Exception:
                    pass
    if os.path.isdir(TEXT_DIR):
        from langchain_community.document_loaders import TextLoader, DirectoryLoader
        for ext in ["*.txt", "*.md"]:
            try:
                docs.extend(
                    DirectoryLoader(
                        TEXT_DIR, glob=ext, loader_cls=TextLoader, show_progress=False
                    ).load()
                )
            except Exception:
                pass
    return docs


def build_retrievers(raw_docs):
    """Split docs and build hybrid retrievers."""
    if not raw_docs:
        return None, []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)
    if not chunks:
        return None, []

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    chroma_ret = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = RETRIEVAL_K

    hybrid = EnsembleRetriever(
        retrievers=[bm25_ret, chroma_ret], weights=[BM25_WEIGHT, CHROMA_WEIGHT]
    )
    return hybrid, chunks


def build_graph(hybrid_retriever):
    """Build and compile the LangGraph workflow."""
    llm = get_llm()
    structured_grader = llm.with_structured_output(GradeDocuments)
    structured_hallucination = llm.with_structured_output(GradeHallucination)

    def retrieve_node(state: AgentState):
        question = state["question"]
        docs = hybrid_retriever.invoke(question)
        return {"documents": docs, "question": question}

    def grade_documents_node(state: AgentState):
        question = state["question"]
        documents = state["documents"]
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a grader assessing relevance of a retrieved document to a user query. "
             "Give a binary score 'yes' or 'no' indicating relevance."),
            ("human", "Retrieved document:\n\n{context}\n\nUser query: {question}"),
        ])
        chain = prompt | structured_grader
        filtered = []
        for doc in documents:
            try:
                score = chain.invoke({"context": doc.page_content[:1500], "question": question})
                if score.binary_score == "yes":
                    filtered.append(doc)
            except Exception:
                filtered.append(doc)
        web_search_needed = "no" if filtered else "yes"
        if not filtered:
            filtered = documents
        return {"documents": filtered, "web_search_needed": web_search_needed}

    def generate_node(state: AgentState):
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join([d.page_content[:1000] for d in documents[:5]])
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert AI assistant. Answer the user's query using *only* the provided context. "
             "If you cannot find the answer in the context, state that clearly. Do not hallucinate."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ])
        response = (prompt | llm).invoke({"context": context, "question": question})
        return {
            "generation": response.content,
            "documents": documents,
            "question": question,
            "web_search_needed": state.get("web_search_needed", "no"),
        }

    def hallucination_grader_node(state: AgentState):
        generation = state["generation"]
        documents = state["documents"]
        context = "\n\n".join([d.page_content[:800] for d in documents[:3]])
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a grader checking if an answer is grounded in the provided context. "
             "Respond with 'yes' if fully supported, 'no' otherwise."),
            ("human", "Context:\n{context}\n\nAnswer:\n{generation}\n\nIs this answer grounded?"),
        ])
        try:
            result = (prompt | structured_hallucination).invoke(
                {"context": context, "generation": generation[:1500]}
            )
            if result.binary_score == "yes":
                return {"generation": generation, "documents": documents,
                        "question": state["question"],
                        "web_search_needed": state.get("web_search_needed", "no")}
            return {"generation": "", "documents": documents,
                    "question": state["question"],
                    "web_search_needed": state.get("web_search_needed", "no")}
        except Exception:
            return {"generation": generation, "documents": documents,
                    "question": state["question"],
                    "web_search_needed": state.get("web_search_needed", "no")}

    def decide_after_grading(state):
        return "web_search" if state.get("web_search_needed") == "yes" else "generate"

    def decide_after_hallucination(state):
        return "generate" if state.get("generation") == "" else END

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_grader", hallucination_grader_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", decide_after_grading,
        {"web_search": "generate", "generate": "generate"},
    )
    workflow.add_edge("generate", "hallucination_grader")
    workflow.add_conditional_edges(
        "hallucination_grader", decide_after_hallucination,
        {"generate": "generate", END: END},
    )
    return workflow.compile()


# --------------- High-level API ---------------
class GraphRAG:
    """Drop-in RAG engine. Builds the full pipeline on init."""

    def __init__(self):
        self.raw_docs = load_documents()
        if not self.raw_docs:
            raise ValueError("No documents found in data/pdfs/ or data/texts/")
        self.hybrid, self.chunks = build_retrievers(self.raw_docs)
        if self.hybrid is None:
            raise ValueError("Could not build retrievers from loaded documents")
        self.graph = build_graph(self.hybrid)

    def ask(self, question: str) -> dict:
        """Run a question through the pipeline. Returns dict with 'generation' and 'documents'."""
        return self.graph.invoke({"question": question})

    @property
    def page_count(self):
        return len(self.raw_docs)

    @property
    def chunk_count(self):
        return len(self.chunks)
