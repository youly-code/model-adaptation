from typing import List, Callable, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class Route:
    """Represents a semantic routing path with its handler function."""
    description: str
    handler: Callable
    embedding: np.ndarray = None

class SemanticRouter:
    """Routes queries to handlers based on semantic similarity."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the router with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.routes: List[Route] = []

    def add_route(self, description: str, handler: Callable) -> None:
        """Add a new route with its description and handler."""
        embedding = self.model.encode(description, convert_to_tensor=False)
        self.routes.append(Route(description, handler, embedding))

    def route(self, query: str, threshold: float = 0.7) -> Callable:
        """Route the query to the most semantically similar handler."""
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # Calculate similarities with all routes
        similarities = [
            np.dot(query_embedding, route.embedding) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(route.embedding))
            for route in self.routes
        ]
        
        # Print similarities for each route
        for i, route in enumerate(self.routes):
            print(f"Route '{route.description[:30]}...': {similarities[i]:.3f}")
        
        # Find the most similar route
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]
        
        if max_similarity < threshold:
            raise ValueError(f"No suitable route found for query: {query}")
            
        print(f"\nSelected route: '{self.routes[max_similarity_idx].description[:30]}...' with similarity: {max_similarity:.3f}\n")
        return self.routes[max_similarity_idx].handler

# Example usage
def handle_greeting(query: str) -> str:
    return "Hello! How can I help you today?"

def handle_weather(query: str) -> str:
    return "The weather is currently sunny and warm."

def handle_time(query: str) -> str:
    return "The current time is 12:00 PM."

if __name__ == "__main__":
    # Initialize the router
    router = SemanticRouter()
    
    # Add routes with more detailed descriptions
    router.add_route(
        "Greetings, hello, hi, hey, good morning, good afternoon, welcome", 
        handle_greeting
    )
    router.add_route(
        "Weather information, what's the weather like, temperature, forecast, is it raining", 
        handle_weather
    )
    router.add_route(
        "Time queries, what time is it, current time, clock, what's the time", 
        handle_time
    )
    
    # Test the router with a lower threshold
    test_queries = [
        "What do you want my dog to eat?",
        "What's the weather like?",
        "Can you tell me the time?",
    ]
    
    for query in test_queries:
        try:
            # Lower the threshold to 0.3 for more lenient matching
            handler = router.route(query, threshold=0.3)
            response = handler(query)
            print(f"Query: {query}")
            print(f"Response: {response}\n")
        except ValueError as e:
            print(f"Error: {e}\n")
