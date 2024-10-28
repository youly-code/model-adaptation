from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Optional
import ollama
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Define model constants
PROMPT_ONLY = False
USE_OLLAMA = True
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OLLAMA_CHAT_MODEL = "llama3.2:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# Initialize clients
if USE_OLLAMA:
    client = ollama.Client()
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class Document:
    """Represents a document with content and metadata."""

    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None


def get_embedding(text: str) -> List[float]:
    """Get embeddings using either Ollama or OpenAI."""
    if USE_OLLAMA:
        response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    else:
        response = client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
        return response.data[0].embedding


class RAGSystem:
    """RAG implementation supporting both Ollama and OpenAI."""

    def __init__(self):
        """Initialize the RAG system."""
        self.documents: List[Document] = []
        if not USE_OLLAMA:
            if api_key := os.getenv("OPENAI_API_KEY"):
                self.client = OpenAI(api_key=api_key)
            else:
                raise ValueError("OPENAI_API_KEY environment variable not set")

    def add_document(self, content: str, metadata: Dict = None) -> None:
        """Add a document to the knowledge base."""
        if metadata is None:
            metadata = {}

        # Get embedding for the document
        embedding = get_embedding(content)

        # Store document with its embedding
        doc = Document(content=content, metadata=metadata, embedding=embedding)
        self.documents.append(doc)

    def find_relevant_docs(self, query: str, top_k: int = 3) -> List[Document]:
        """Find the most relevant documents using similarity search."""
        # Get query embedding
        query_embedding = get_embedding(query)

        # Calculate similarities
        similarities = []
        for doc in self.documents:
            similarity = np.dot(query_embedding, doc.embedding)
            similarities.append(similarity)

        # Get top k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate_response(self, query: str) -> Dict:
        """Generate a response using retrieved documents."""
        # Find relevant documents
        relevant_docs = self.find_relevant_docs(query)

        # Construct context from relevant documents
        context = "\n\n".join(doc.content for doc in relevant_docs)

        # Construct the prompt
        prompt = f"""Use the following information to answer the question. 
        If the information doesn't contain the answer, say so.
        
        Information:
        {context}
        
        Question: {query}
        
        Answer based only on the provided information. If uncertain, say so.
        """

        # Generate response using selected model
        if USE_OLLAMA:
            response = ollama.chat(
                model=OLLAMA_CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided information.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response["message"]["content"]
        else:
            response = client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer = response.choices[0].message.content

        return {"answer": answer, "sources": [doc.metadata for doc in relevant_docs]}


def demo_rag():
    """Demonstrate RAG with example company documents."""
    # Create RAG instance
    rag = RAGSystem()

    # Example company documents
    documents = [
        {
            "content": """Company Dress Code Policy:
            1. Business casual attire is required Monday through Thursday
            2. Casual Friday allows jeans and sneakers
            3. All clothing must be clean and professional
            4. No excessive jewelry or visible tattoos
            5. Client meetings require formal business attire""",
            "metadata": {"type": "policy", "department": "HR", "id": "POL-001"},
        },
        {
            "content": """Employee Benefits Summary:
            - Health insurance coverage starts after 30 days
            - 401k matching up to 5% of salary
            - 20 days paid vacation annually
            - Hybrid work schedule available
            - Annual learning budget of $2000""",
            "metadata": {"type": "benefits", "department": "HR", "id": "BEN-001"},
        },
        {
            "content": """Office Security Protocol:
            1. Wear ID badge at all times
            2. Visitors must sign in at reception
            3. Lock computers when away from desk
            4. Report suspicious activity to security
            5. No tailgating through security doors""",
            "metadata": {
                "type": "security",
                "department": "Facilities",
                "id": "SEC-001",
            },
        },
    ]

    # Add documents to RAG
    for doc in documents:
        rag.add_document(doc["content"], doc["metadata"])
        print(f"Added document: {doc['metadata']['id']}")

    # Example queries
    queries = [
        "What is the dress code for Fridays?",
        "How many vacation days do employees get?",
        "What is the lunch policy?",  # Not in documents
        "What are the security requirements?",
    ]

    # Run queries and show responses
    for query in queries:
        print(f"\nQuestion: {query}")
        result = rag.generate_response(query)
        print(f"Answer: {result['answer']}")
        print("Sources used:")
        for source in result["sources"]:
            print(f"- {source['type']} ({source['id']})")
        print("-" * 80)


if __name__ == "__main__":
    demo_rag()
