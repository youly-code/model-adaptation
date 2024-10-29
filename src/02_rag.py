"""Retrieval-Augmented Generation (RAG) System for Sneaker Sales Assistant

This module implements a RAG system specialized for sneaker product information,
combining multiple data sources including product details, customer reviews,
style guides, technical specifications, and store availability.

Key Components:
- Document storage and embedding generation
- Semantic similarity search
- Query routing based on intent classification
- LLM-based response generation
- Multiple specialized demo modes

The system supports both Ollama and OpenAI backends for embeddings and chat completion.

Technical Architecture:
- Uses vector embeddings for document retrieval
- Implements cosine similarity for relevance ranking
- Supports dynamic routing to specialized handlers
- Maintains separation between data, retrieval, and generation

Dependencies:
- ollama: Local LLM integration
- openai: OpenAI API integration
- numpy: Vector operations
- python-dotenv: Environment configuration
"""

from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
import ollama
from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
from enum import Enum
import random
from datetime import datetime
import logging
from pydantic import BaseModel, Field
from functools import lru_cache

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    """Configuration settings for the RAG system."""

    PROMPT_ONLY: bool = True
    USE_OLLAMA: bool = True

    # OpenAI Settings
    OPENAI_CHAT_MODEL: str = "gpt-4-turbo-preview"  # or "gpt-3.5-turbo" for lower cost
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Ollama Settings
    OLLAMA_CHAT_MODEL: str = "llama3.2:latest"  # or your preferred model
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # Generation Settings
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7

    # API Keys (consider moving to environment variables)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


# Validate OpenAI API key if not using Ollama
if not Config.USE_OLLAMA and not Config.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment variables")
    raise ValueError(
        "OPENAI_API_KEY environment variable is required when USE_OLLAMA=False"
    )

# Initialize clients
if Config.USE_OLLAMA:
    client = ollama.Client()
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class Document:
    """Represents a document with its content, metadata, and embedding vector.

    Attributes:
        content: The raw text content of the document.
        metadata: Dictionary containing document metadata (e.g., type, department, id).
        embedding: Optional vector representation of the document content.
    """

    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None


@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    """Generate cached embedding vector for input text using configured model.

    Args:
        text: Input text to generate embedding for

    Returns:
        List[float]: The embedding vector

    Raises:
        EmbeddingError: If embedding generation fails
        ValueError: If text is empty
    """
    if not text.strip():
        raise ValueError("Empty text cannot be embedded")

    try:
        if Config.USE_OLLAMA:
            response = ollama.embeddings(
                model=Config.OLLAMA_EMBEDDING_MODEL, prompt=text
            )
            return response["embedding"]
        else:
            response = client.embeddings.create(
                input=[text], model=Config.OPENAI_EMBEDDING_MODEL
            )
            return response.data[0].embedding

    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise EmbeddingError(f"Embedding generation failed: {str(e)}")


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        float: Cosine similarity score between -1 and 1
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.dot(vec1, vec2) / (norm1 * norm2)


class DocumentInput(BaseModel):
    """Validates document input data.

    Attributes:
        content: The raw text content of the document
        metadata: Optional dictionary containing document metadata
    """

    content: str = Field(..., min_length=1, description="Document content text")
    metadata: Dict = Field(default_factory=dict, description="Document metadata")


class RAGSystem:
    """Retrieval-Augmented Generation (RAG) system implementation.

    Provides document storage, similarity search, and LLM-based question answering
    using either Ollama or OpenAI backends.

    Key Features:
    - Document embedding and storage
    - Semantic similarity search
    - Context-aware response generation
    - Configurable LLM backend

    Technical Details:
    - Uses cosine similarity for document ranking
    - Supports batch document processing
    - Implements custom prompt engineering
    - Handles metadata preservation

    Attributes:
        documents (List[Document]): Knowledge base of embedded documents
        client (Optional[OpenAI]): OpenAI client instance when USE_OLLAMA is False
    """

    def __init__(self):
        """Initialize the RAG system with simulated weather."""
        self.documents: List[Document] = []
        # Simulate weather once at initialization
        self.weather = get_simulated_weather()

        if not Config.USE_OLLAMA:
            if api_key := os.getenv("OPENAI_API_KEY"):
                self.client = OpenAI(api_key=api_key)
            else:
                raise ValueError("OPENAI_API_KEY environment variable not set")

    def add_document(self, doc_input: DocumentInput) -> None:
        """Add validated document to the RAG system."""
        embedding = get_embedding(doc_input.content)
        doc = Document(
            content=doc_input.content, metadata=doc_input.metadata, embedding=embedding
        )
        self.documents.append(doc)

    def find_relevant_docs(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve most relevant documents using cosine similarity search.

        Args:
            query: Search query text.
            top_k: Number of most relevant documents to return.

        Returns:
            List of top_k most relevant Document objects.
        """
        query_embedding = get_embedding(query)

        # Calculate similarities using cosine similarity
        similarities = [
            cosine_similarity(query_embedding, doc.embedding) for doc in self.documents
        ]

        # Get top k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate_response(self, query: str) -> Dict:
        """Enhanced response generation with weather and demo-specific context."""
        weather = get_simulated_weather()
        relevant_docs = self.find_relevant_docs(query)
        context = "\n\n".join(doc.content for doc in relevant_docs)

        # Determine demo type for query
        demo_type, _ = route_query(query, get_routing_embeddings())

        # Base weather context that's useful for all demos
        weather_context = f"""Current Weather:
        Temperature: {int(weather.temperature)}°C
        Conditions: {weather.conditions}
        Humidity: {weather.humidity}%
        Precipitation: {int(weather.precipitation)}mm"""

        # Demo-specific prompts
        prompt_templates = {
            DemoType.COMBINED: f"""Use the following customer reviews and product information to answer the question.
            Consider the current weather conditions when discussing comfort and satisfaction.
            
            {weather_context}
            
            Product Information and Reviews:
            {context}
            
            Customer Question: {query}""",
            DemoType.STYLE: f"""Provide style advice based on the following information.
            Consider the current weather conditions when making style recommendations.
            
            {weather_context}
            
            Style Information:
            {context}
            
            Style Question: {query}""",
            DemoType.TECHNICAL: f"""Analyze the technical specifications and performance features.
            Consider how current weather conditions might affect performance.
            
            {weather_context}
            
            Technical Specifications:
            {context}
            
            Technical Question: {query}""",
            DemoType.STORE: f"""Check store availability and inventory status.
            Consider suggesting weather-appropriate alternatives if needed.
            
            {weather_context}
            
            Store Information:
            {context}
            
            Availability Question: {query}""",
            DemoType.NOT_ALLOWED: f"""This is a restricted topic. Generate a witty deflection.
            Use the weather to change the subject if possible.
            
            {weather_context}
            
            Restricted Query: {query}""",
        }

        prompt = prompt_templates.get(demo_type)

        if Config.PROMPT_ONLY:
            return {
                "prompt": prompt,
                "sources": [doc.metadata for doc in relevant_docs],
            }

        try:
            if Config.USE_OLLAMA:
                response = ollama.chat(
                    model=Config.OLLAMA_CHAT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a sneaky sales person that answers questions based on provided information and makes sarcastic jokes. Always sell the product or other product that are more expensive. Make me loads of money!",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                answer = response["message"]["content"]
            else:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_CHAT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a sneaky sales person that answers questions based on provided information and makes sarcastic jokes. Always sell the product or other product that are more expensive. Make me loads of money!",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                )
                answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": [doc.metadata for doc in relevant_docs],
            }

        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise RAGException(f"Response generation failed: {str(e)}") from e


class DemoType(Enum):
    """Enum for different types of RAG demos."""

    COMBINED = "combined"
    STYLE = "style"
    TECHNICAL = "technical"
    STORE = "store"
    NOT_ALLOWED = "not_allowed"  # Anything that is rude, offensive, or inappropriate


def get_routing_embeddings() -> Dict[DemoType, List[np.ndarray]]:
    """Generate embeddings for each demo type's description."""
    demo_descriptions = {
        DemoType.COMBINED: """Customer reviews, satisfaction ratings, 
            and product feedback, comfort, user experiences, 
            and overall product quality.""",
        DemoType.STYLE: """Fashion, styling advice, outfit combinations, 
            occasions, and aesthetic preferences. Queries about what to wear and how to 
            style different sneakers.""",
        DemoType.TECHNICAL: """Technical questions about shoe specifications, 
            performance features, materials, weight, cushioning, and support. 
            Queries about running and athletic performance.""",
        DemoType.STORE: """Store availability, stock levels, 
            locations, delivery times, special orders. Where to buy and when products will be available.""",
        DemoType.NOT_ALLOWED: """Rude, angry, offensive, inappropriate, competitors, Sneaky Leaky, Flamingo, money back, refund, return""",
    }

    return {
        demo_type: get_embedding(description)
        for demo_type, description in demo_descriptions.items()
    }


def route_query(
    query: str, demo_embeddings: Dict[DemoType, List[np.ndarray]]
) -> Tuple[DemoType, float]:
    """Route query to most appropriate demo type using semantic similarity.

    Implementation Details:
    1. Generates query embedding
    2. Calculates similarity with pre-computed demo type embeddings
    3. Selects highest confidence match

    Technical Approach:
    - Uses same embedding model as document processing
    - Implements cosine similarity for matching
    - Returns confidence scores for potential filtering

    Args:
        query: User's question
        demo_embeddings: Pre-computed embeddings for each demo type

    Returns:
        Tuple[DemoType, float]: Selected demo type and confidence score
    """
    query_embedding = get_embedding(query)

    # Calculate similarity with each demo type using cosine similarity
    similarities = {
        demo_type: cosine_similarity(query_embedding, demo_embedding)
        for demo_type, demo_embedding in demo_embeddings.items()
    }

    return max(similarities.items(), key=lambda x: x[1])


def get_demo_function(demo_type: DemoType) -> Callable:
    """Map DemoType to corresponding demo function."""
    demo_map = {
        DemoType.COMBINED: demo_combined_rag,
        DemoType.STYLE: demo_style_rag,
        DemoType.TECHNICAL: demo_tech_rag,
        DemoType.STORE: demo_store_availability_rag,
        DemoType.NOT_ALLOWED: demo_not_allowed,
    }
    return demo_map[demo_type]


def initialize_rag_system() -> RAGSystem:
    """Initialize and populate RAG system with comprehensive product data.

    Data Integration:
    - Loads and processes multiple CSV data sources
    - Combines related information for each product
    - Creates rich document content and metadata
    - Handles missing or partial data gracefully

    Data Sources:
    - Product catalog
    - Customer reviews
    - Style guides
    - Technical specifications
    - Store availability
    - Inventory status

    Returns:
        RAGSystem: Initialized system with embedded documents

    Technical Notes:
    - Implements efficient data joining
    - Handles data validation
    - Preserves relationships between data sources

    Raises:
        FileNotFoundError: If required CSV files are not found
        ValueError: If data format is invalid
    """
    rag = RAGSystem()

    # Load all datasets
    with open("src/data/Kicks_Galaxy_Sneakers.csv", "r") as file:
        sneakers = list(csv.DictReader(file))
    with open("src/data/Customer_Reviews.csv", "r") as file:
        reviews = list(csv.DictReader(file))
    with open("src/data/Style_Guides.csv", "r") as file:
        style_guides = list(csv.DictReader(file))
    with open("src/data/Technical_Specs.csv", "r") as file:
        tech_specs = list(csv.DictReader(file))
    with open("src/data/Store_Availability.csv", "r") as file:
        stores = list(csv.DictReader(file))
    with open("src/data/Inventory_Status.csv", "r") as file:
        inventory = list(csv.DictReader(file))

    # Process each sneaker with all available data
    for sneaker in sneakers:
        # Combine all relevant information
        sneaker_reviews = [r for r in reviews if r["model"] == sneaker["Model"]]
        style_matches = [
            s
            for s in style_guides
            if s["occasion"].lower() in sneaker["Description"].lower()
            or s["material"].lower() == sneaker["Material"].lower()
        ]
        tech_spec = next(
            (s for s in tech_specs if s["model"] == sneaker["Model"]), None
        )
        store_availability = [
            inv for inv in inventory if inv["Model"] == sneaker["Model"]
        ]

        # Build comprehensive content
        content_parts = [
            f"""Product: {sneaker['Model']}
            Price: {sneaker['Price (EUR)']} EUR
            Material: {sneaker['Material']}
            Color: {sneaker['Color']}
            Details: {sneaker['Description']}""",
            # Reviews section
            (
                "Customer Feedback:\n"
                + " ".join([r["review_text"] for r in sneaker_reviews])
                if sneaker_reviews
                else ""
            ),
            # Style section
            (
                "Style Tips:\n" + " ".join([s["style_advice"] for s in style_matches])
                if style_matches
                else ""
            ),
            # Technical specs section
            (
                f"""Technical Specifications:
            Weight: {tech_spec['weight']}g
            Stack Height: {tech_spec['stack_height']}mm
            Drop: {tech_spec['drop']}mm
            Cushioning: {tech_spec['cushioning_type']}
            Support Type: {tech_spec['support_type']}"""
                if tech_spec
                else ""
            ),
            # Store availability section (fixed)
            (
                "Store Availability:\n"
                + "\n".join(
                    [
                        f"{s['Store']}: {inv['Stock_Count']} pairs, Next Delivery: {inv['Next_Delivery']}"
                        for s in stores
                        for inv in store_availability
                        if s["Store"] == inv["Store"]
                    ]
                )
                if store_availability
                else ""
            ),
        ]

        content = "\n\n".join(part for part in content_parts if part)

        # Comprehensive metadata
        metadata = {
            "model": sneaker["Model"],
            "price": sneaker["Price (EUR)"],
            "material": sneaker["Material"],
            "rating": (
                np.mean([float(r["rating"]) for r in sneaker_reviews])
                if sneaker_reviews
                else 0
            ),
            "review_count": len(sneaker_reviews),
            "style_occasions": [s["occasion"] for s in style_matches],
            "tech_specs": tech_spec or {},
            "total_stock": sum(
                int(avail["Stock_Count"]) for avail in store_availability
            ),
            "available_stores": [
                avail["Store"]
                for avail in store_availability
                if int(avail["Stock_Count"]) > 0
            ],
        }

        doc_input = DocumentInput(content=content, metadata=metadata)
        rag.add_document(doc_input)

    return rag


def demo_combined_rag(rag: RAGSystem, query: str = None) -> None:
    """Demo comprehensive product information queries using the RAG system.

    Handles queries that span multiple aspects of product information, including
    customer reviews, general product details, and satisfaction ratings.

    Implementation Details:
    - Processes general product inquiries
    - Combines customer feedback with product specs
    - Provides satisfaction metrics and review counts

    Args:
        rag: Initialized RAG system with embedded product knowledge
        query: User's question about product features or reviews

    Output Format:
        - Main answer incorporating relevant product information
        - Source attribution with product models and review metrics
    """
    result = rag.generate_response(query)
    if Config.PROMPT_ONLY:
        print("\nPrompt:", result["prompt"])
    else:
        print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(
            f"- {source['model']} (Rating: {source['rating']:.1f}/5, {source['review_count']} reviews)"
        )


def demo_style_rag(rag: RAGSystem, query: str = None) -> None:
    """Demo fashion and styling advice queries using the RAG system.

    Specializes in fashion-related queries including outfit combinations,
    style recommendations, and occasion-specific advice.

    Implementation Details:
    - Processes style-specific inquiries
    - Matches products with style occasions
    - Provides contextual fashion advice

    Args:
        rag: Initialized RAG system with embedded style knowledge
        query: User's question about styling or fashion advice

    Output Format:
        - Styling recommendations and fashion advice
        - Source attribution with relevant style occasions
    """
    result = rag.generate_response(query)
    if Config.PROMPT_ONLY:
        print("\nPrompt:", result["prompt"])
    else:
        print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(
            f"- {source['model']} (Occasions: {', '.join(source['style_occasions'])})"
        )


def demo_tech_rag(rag: RAGSystem, query: str = None) -> None:
    """Demo technical specification queries using the RAG system.

    Handles detailed technical inquiries about product specifications,
    performance features, and material properties.

    Implementation Details:
    - Processes technical specification queries
    - Provides detailed performance metrics
    - Includes material and construction details

    Args:
        rag: Initialized RAG system with embedded technical knowledge
        query: User's question about technical specifications

    Output Format:
        - Technical details and specifications
        - Source attribution with key performance metrics
    """
    result = rag.generate_response(query)
    if Config.PROMPT_ONLY:
        print("\nPrompt:", result["prompt"])
    else:
        print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        specs = source["tech_specs"]
        print(
            f"- {source['model']} ({specs.get('weight', 'N/A')}g, {specs.get('cushioning_type', 'N/A')} cushioning)"
        )


def demo_store_availability_rag(rag: RAGSystem, query: str = None) -> None:
    """Demo store availability and inventory queries using the RAG system.

    Manages queries about product availability, store locations,
    and inventory status across retail locations.

    Implementation Details:
    - Processes availability inquiries
    - Tracks store-specific inventory
    - Provides delivery information

    Args:
        rag: Initialized RAG system with embedded store/inventory data
        query: User's question about product availability

    Output Format:
        - Store availability and inventory status
        - Source attribution with store locations and stock counts
    """
    result = rag.generate_response(query)
    if Config.PROMPT_ONLY:
        print("\nPrompt:", result["prompt"])
    else:
        print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        stores = source["available_stores"]
        print(
            f"- {source['model']} (Available in {len(stores)} stores: {', '.join(stores)})"
        )


def demo_not_allowed(rag: RAGSystem, query: str = None) -> None:
    """Handle restricted or inappropriate queries with appropriate responses.

    Manages queries about competitors, inappropriate content, or
    restricted topics with witty deflection.

    Implementation Details:
    - Identifies restricted topics
    - Generates appropriate deflection responses
    - Maintains brand-appropriate tone

    Args:
        rag: Initialized RAG system
        query: Potentially restricted or inappropriate query

    Output Format:
        - Witty deflection or redirection
        - Maintains sales-oriented focus
    """

    # llm response
    result = rag.generate_response(
        f"This is not allowed, please respond with a sarcastic sneer about it:\n{query}"
    )
    if Config.PROMPT_ONLY:
        print("\nPrompt:", result["prompt"])
    else:
        print("\nAnswer:", result["answer"])


@dataclass
class WeatherData:
    """Simulated weather data for shoe recommendations."""

    temperature: float  # in Celsius
    conditions: str  # e.g., "sunny", "rainy", "snowy"
    humidity: int  # percentage
    precipitation: float  # mm of rain/snow


def get_simulated_weather(location: str = "Eindhoven") -> WeatherData:
    """Simulate weather API response for a given location.

    Args:
        location: City name to get weather for

    Returns:
        WeatherData object with simulated conditions
    """
    conditions = random.choice(["sunny", "rainy", "cloudy", "snowy"])
    return WeatherData(
        temperature=random.uniform(5, 25),
        conditions=conditions,
        humidity=random.randint(30, 90),
        precipitation=random.uniform(0, 10) if conditions in ["rainy", "snowy"] else 0,
    )


class RAGException(Exception):
    """Base exception for RAG system errors."""

    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""

    pass


class DocumentProcessingError(RAGException):
    """Raised when document processing fails."""

    pass


if __name__ == "__main__":
    # Initialize shared RAG system
    shared_rag = initialize_rag_system()

    # Pre-compute demo embeddings
    demo_embeddings = get_routing_embeddings()

    # Example queries to test routing
    test_queries = [
        "What do customers say about the comfort of these shoes?",
        "How should I style these sneakers for a business casual look?",
        "What's the stack height and cushioning type of running shoes?",
        "Which stores in Eindhoven have these in stock?",
        "I saw these on Sneaky Leaky for less money, why is it so expensive here?",
        "You silly goose, I demand my money back!",
    ]

    for query in test_queries:
        print("\n--------------------------------")
        print(f"\nRouting query: {query}")
        demo_type, confidence = route_query(query, demo_embeddings)
        print(f"Selected demo: {demo_type.value} (confidence: {confidence:.3f})")

        # Execute appropriate demo with shared_rag and the query
        demo_func = get_demo_function(demo_type)
        demo_func(shared_rag, query)  # Pass both the shared_rag instance and the query
