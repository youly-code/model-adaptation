from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import asyncio
from ollama import AsyncClient
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import networkx as nx
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the multi-agent conversation.

    Attributes:
        content (str): The text content of the message
        sender (str): Name of the agent who sent the message
        vector (np.ndarray, optional): Embedded vector representation of the message
    """

    content: str
    sender: str
    vector: np.ndarray = None

    def __eq__(self, other: object) -> bool:
        """Checks equality between two Message objects.
        
        Args:
            other (object): Another object to compare with
            
        Returns:
            bool: True if content and sender match, False otherwise
        """
        if not isinstance(other, Message):
            return NotImplemented
        return (self.content == other.content and 
                self.sender == other.sender)


@dataclass
class ChatHistory:
    """Maintains the conversation history between agents.

    Attributes:
        messages (List[Message]): Chronological list of all messages
    """

    messages: List[Message] = field(default_factory=list)

    def add_message(self, message: Message):
        """Adds a new message to the conversation history."""
        self.messages.append(message)

    def get_recent_context(self, n: int = 5) -> str:
        """Retrieves the n most recent messages formatted as a conversation.

        Args:
            n (int): Number of recent messages to include

        Returns:
            str: Formatted string of recent messages
        """
        return "\n".join([f"{msg.sender}: {msg.content}" for msg in self.messages[-n:]])


@dataclass
class ResearchAgent:
    """An AI agent that participates in research discussions.

    Attributes:
        name (str): Agent's identifier
        personality (str): Description of agent's personality traits
        expertise (str): Agent's domain of expertise
        client (AsyncClient): Ollama client for LLM interactions
        chat_history (ChatHistory): Shared conversation history
        memory (Dict[str, Any]): Agent's memory/state for tracking information
    """

    name: str
    personality: str
    expertise: str
    client: AsyncClient
    chat_history: ChatHistory
    memory: Dict[str, Any] = field(default_factory=dict)

    async def get_embedding(self, text: str) -> np.ndarray:
        """Generates vector embedding for input text using Ollama.

        Args:
            text (str): Input text to embed

        Returns:
            np.ndarray: Vector embedding of the text
            
        Raises:
            Exception: If embedding generation fails with Ollama API
        """
        try:
            # Call Ollama API to generate embeddings using hermes3 model
            response = await self.client.embeddings(model="hermes3", prompt=text)
            # Convert embedding list to numpy array for vector operations
            return np.array(response["embedding"])
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Consider fallback strategy or re-raising custom exception
            raise

    async def analyze_sentiment(self, message: Message) -> float:
        """Analyzes sentiment of messages to inform responses.
        
        Args:
            message (Message): The message to analyze
            
        Returns:
            float: Sentiment score of the message
        """
        # Add sentiment analysis logic
        prompt = f"Analyze the sentiment of this message and return a score from -1 to 1: {message.content}"
        response = await self.client.chat(model="hermes3", messages=[{"role": "user", "content": prompt}])
        try:
            sentiment = float(response["message"]["content"])
            return max(min(sentiment, 1.0), -1.0)  # Ensure value is between -1 and 1
        except ValueError:
            return 0.0  # Default neutral sentiment if parsing fails
        
    async def respond(self, topic: str) -> str:
        """Generates a response to the current topic considering conversation context.

        Args:
            topic (str): The current discussion topic

        Returns:
            str: Agent's response based on personality and expertise
        """
        context = self.chat_history.get_recent_context()
        
        # Consider message sentiment in response
        recent_messages = self.chat_history.messages[-3:]
        sentiments = [await self.analyze_sentiment(msg) for msg in recent_messages]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        sentiment_context = "positive" if avg_sentiment > 0.3 else "negative" if avg_sentiment < -0.3 else "neutral"
        
        prompt = f"""You are {self.name}, a researcher with {self.expertise}. 
        Your personality is: {self.personality}
        
        Topic: {topic}
        Recent conversation:
        {context}
        
        The current conversation tone is {sentiment_context}.
        
        Provide your perspective on the topic, considering the conversation history and your unique viewpoint.
        Keep your response concise (2-3 sentences).
        """

        response = await self.client.chat(
            model="hermes3", messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    async def update_memory(self, key: str, value: Any):
        """Updates agent's memory with new information.
        
        Args:
            key (str): The key to store the information under
            value (Any): The value to store in memory
        """
        self.memory[key] = value

    async def reflect_on_conversation(self) -> Dict[str, Any]:
        """Analyzes conversation patterns and updates internal model"""
        recent_context = self.chat_history.get_recent_context()
        reflection_prompt = f"""
        Analyze the conversation so far and identify:
        1. Key themes and patterns
        2. Points of agreement/disagreement
        3. Potential knowledge gaps
        Based on: {recent_context}
        """
        # Process reflection and update memory
        
    async def form_opinion(self, topic: str) -> Dict[str, float]:
        """Develops weighted stance on discussion topics"""
        return {
            "agreement_level": 0.7,
            "confidence": 0.8,
            "expertise_relevance": 0.9
        }


class SemanticRouter:
    """Routes messages to the most relevant agent based on semantic similarity.

    Attributes:
        agents (List[ResearchAgent]): List of available agents
    """

    def __init__(self, agents: List[ResearchAgent]):
        self.agents = agents
        self.routing_history = []
        self.concept_graph = nx.DiGraph()
        self.embedding_cache = {}  # Cache for expertise embeddings

    async def get_cached_embedding(self, text: str) -> np.ndarray:
        """Gets embedding from cache or generates new one."""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = await self.agents[0].get_embedding(text)
        return self.embedding_cache[text]

    async def route_message(self, message: Message) -> ResearchAgent:
        """Selects the most appropriate agent to respond based on message content.

        Args:
            message (Message): The message to route

        Returns:
            ResearchAgent: An agent selected based on semantic similarity with
                         controlled randomization
        """
        # Calculate semantic similarity between message and each agent's expertise
        message_embedding = await self.get_cached_embedding(message.content)
        similarities = []

        for agent in self.agents:
            expertise_embedding = await self.get_cached_embedding(agent.expertise)
            similarity = cosine_similarity(
                message_embedding.reshape(1, -1), expertise_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)

        # Convert similarities to probabilities using softmax
        similarities = np.array(similarities)
        probabilities = np.exp(similarities) / np.sum(np.exp(similarities))

        # Add randomization factor (temperature)
        temperature = (
            0.8  # Adjust this value to control randomness (higher = more random)
        )
        probabilities = np.power(probabilities, 1 / temperature)
        probabilities /= np.sum(probabilities)

        # Select agent based on probability distribution
        return np.random.choice(self.agents, p=probabilities)


async def run_discussion(topic: str, num_turns: int = 5, 
                        min_agents: int = 2, max_consecutive: int = 2):
    """Orchestrates a multi-turn discussion between agents on a given topic.
    Adds constraints to ensure conversation variety and prevent agent domination.

    Args:
        topic (str): The main discussion topic
        num_turns (int): Number of conversation turns to simulate
        min_agents (int): Minimum number of unique agents that should participate
        max_consecutive (int): Maximum number of consecutive turns by same agent
    """
    client = AsyncClient()
    chat_history = ChatHistory()

    agents = [
        ResearchAgent(
            "Dr. Optimistify",
            "Enthusiastic and forward-thinking",
            "technological innovation and future possibilities, think about the best case scenario",
            client,
            chat_history,
        ),
        ResearchAgent(
            "Dr. Skepticalculus",
            "Critical and analytical",
            "risk assessment and potential downsides, think about the worst case scenario",
            client,
            chat_history,
        ),
        ResearchAgent(
            "Dr. Pragmatium",
            "Balanced and solution-oriented",
            "practical implementation and real-world applications, which are the most important things",
            client,
            chat_history,
        ),
        ResearchAgent(
            "Dr. Outthereium",
            "Outspoken and unconventional",
            "unconventional ideas and alternative perspectives, weird and weirdly specific knowledge",
            client,
            chat_history,
        ),
        ResearchAgent(
            "Dr. Curiousmos",
            "Curious and open-minded",
            "new and emerging technologies, and the potential for unexpected discoveries",
            client,
            chat_history,
        ),
        ResearchAgent(
            "Dr. Woohwoorium",
            "Strangely excited and mentally unstable",
            "idiosyncratic ideas from the fringes of reality, think about the weirdest things",
            client,
            chat_history,
        ),
    ]

    router = SemanticRouter(agents)
    consecutive_turns = 0
    last_agent = None

    try:
        # Initial message
        initial_message = Message(topic, "System")
        current_agent = await router.route_message(initial_message)

        for turn in range(num_turns):
            logger.info(f"\nTurn {turn + 1}:")
            response = await current_agent.respond(topic)
            message = Message(response, current_agent.name)
            chat_history.add_message(message)
            logger.info(f"{current_agent.name}: {response}")

            # Route to next agent
            current_agent = await router.route_message(message)
            await asyncio.sleep(1)  # Rate limiting

            # Prevent same agent from dominating
            while (current_agent == last_agent and 
                   consecutive_turns >= max_consecutive):
                current_agent = await router.route_message(message)
                
            if current_agent == last_agent:
                consecutive_turns += 1
            else:
                consecutive_turns = 1
                
            last_agent = current_agent
    finally:
        # Cleanup resources
        await client.aclose()
        logger.info("Discussion completed, resources cleaned up")


class Config:
    """Central configuration for the multi-agent system."""
    MODEL_NAME: str = "hermes3"
    TEMPERATURE: float = 0.8
    MAX_RETRIES: int = 3
    RATE_LIMIT_DELAY: float = 1.0
    CONTEXT_WINDOW: int = 5


@dataclass
class Metrics:
    """Tracks system performance metrics."""
    response_times: List[float] = field(default_factory=list)
    agent_selections: Dict[str, int] = field(default_factory=dict)
    embedding_cache_hits: int = 0


class ConversationManager:
    """Manages conversation flow and quality"""
    
    async def evaluate_discussion_quality(self) -> float:
        """Scores conversation based on:
        - Diversity of viewpoints
        - Depth of insights
        - Coherence
        - Agent participation balance
        """
        
    async def suggest_direction(self) -> str:
        """Suggests new angles or topics to explore"""
        
    async def detect_convergence(self) -> bool:
        """Determines if discussion has reached natural conclusion"""


class SemanticProcessor:
    """Advanced semantic processing capabilities"""
    
    async def extract_key_concepts(self, text: str) -> List[str]:
        """Identifies main concepts from text"""
        
    async def track_concept_evolution(self) -> Dict[str, List[float]]:
        """Tracks how concepts develop through conversation"""
        
    async def measure_semantic_coherence(self) -> float:
        """Evaluates semantic consistency of discussion"""


class DiscussionPool:
    """Manages multiple parallel discussions"""
    
    async def schedule_discussion(self, topic: str) -> str:
        """Schedules new discussion with available agents"""
        
    async def balance_load(self):
        """Optimizes resource utilization across discussions"""
        
    async def monitor_health(self) -> Dict[str, Any]:
        """Tracks system performance metrics"""


class KnowledgeBase:
    """Manages shared knowledge across discussions"""
    
    async def store_insight(self, insight: Dict[str, Any]):
        """Stores valuable insights for future reference"""
        
    async def query_relevant_knowledge(self, context: str) -> List[Dict]:
        """Retrieves relevant past insights"""
        
    async def update_agent_knowledge(self, agent: ResearchAgent):
        """Updates agent's knowledge based on discussion outcomes"""


class DiscussionAnalytics:
    """Advanced analytics for discussions"""
    
    def analyze_agent_influence(self) -> Dict[str, float]:
        """Measures each agent's impact on discussion"""
        
    def identify_breakthrough_moments(self) -> List[int]:
        """Identifies key turning points in discussion"""
        
    def generate_discussion_graph(self) -> nx.Graph:
        """Creates network graph of concept relationships"""


class ExternalIntegration:
    """Handles external system integration"""
    
    async def export_insights(self, format: str = "json") -> str:
        """Exports discussion insights in specified format"""
        
    async def import_external_knowledge(self, source: str):
        """Imports knowledge from external sources"""
        
    async def webhook_notifications(self, event: str):
        """Sends notifications about significant events"""


@dataclass
class AdaptiveAgent(ResearchAgent):
    """Agent capable of learning and adapting from discussions."""
    
    learning_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    belief_system: Dict[str, float] = field(default_factory=dict)
    
    async def learn_from_interaction(self, message: Message, outcome: Dict[str, Any]):
        """Updates agent's knowledge and behavior based on interaction outcomes.
        
        Args:
            message (Message): The interaction message
            outcome (Dict[str, Any]): Interaction results (agreement, impact, etc.)
        """
        # Extract concepts and update belief strengths
        concepts = await self.extract_concepts(message.content)
        for concept in concepts:
            current_belief = self.belief_system.get(concept, 0.5)
            impact = outcome.get('impact', 0)
            
            # Update belief using Bayesian-inspired approach
            new_belief = (current_belief * 0.8 + impact * 0.2)
            self.belief_system[concept] = max(0, min(1, new_belief))
            
        # Track learning progress
        self.learning_history['belief_updates'].append({
            'timestamp': time.time(),
            'concepts': concepts,
            'outcome': outcome
        })

    async def adapt_personality(self, discussion_metrics: Dict[str, float]):
        """Adjusts personality traits based on discussion effectiveness.
        
        Args:
            discussion_metrics: Metrics about discussion success
        """
        effectiveness = discussion_metrics.get('effectiveness', 0.5)
        engagement = discussion_metrics.get('engagement', 0.5)
        
        # Adjust personality parameters
        if effectiveness < 0.3:
            self.personality = self.personality.replace('assertive', 'collaborative')
        elif engagement < 0.3:
            self.personality = self.personality.replace('reserved', 'engaging')


class AdvancedSemanticRouter:
    """Enhanced router with sophisticated message routing capabilities."""
    
    def __init__(self, agents: List[ResearchAgent]):
        self.agents = agents
        self.routing_history = []
        self.concept_graph = nx.DiGraph()
        self.embedding_cache = {}
        
    async def route_message(self, message: Message) -> Tuple[ResearchAgent, float]:
        """Routes message using multiple semantic factors.
        
        Args:
            message (Message): Message to route
            
        Returns:
            Tuple[ResearchAgent, float]: Selected agent and confidence score
        """
        # Multi-factor scoring
        scores = await asyncio.gather(*[
            self.compute_semantic_score(agent, message),
            self.compute_expertise_match(agent, message),
            self.compute_conversation_flow(agent, message),
            self.compute_diversity_score(agent)
        ])
        
        # Weight different factors
        weights = [0.4, 0.3, 0.2, 0.1]
        final_scores = []
        
        for agent_idx, agent in enumerate(self.agents):
            agent_scores = [s[agent_idx] for s in scores]
            weighted_score = sum(s * w for s, w in zip(agent_scores, weights))
            final_scores.append(weighted_score)
            
        # Apply temperature-based sampling
        temperature = self.compute_dynamic_temperature(final_scores)
        probabilities = self.apply_temperature(final_scores, temperature)
        
        selected_agent = np.random.choice(self.agents, p=probabilities)
        confidence = max(final_scores)
        
        # Update routing history
        self.update_routing_history(message, selected_agent, confidence)
        
        return selected_agent, confidence
        
    async def compute_semantic_score(self, agent: ResearchAgent, 
                                   message: Message) -> float:
        """Computes semantic similarity with advanced techniques."""
        # Use multiple embedding models for robustness
        embeddings = await asyncio.gather(
            self.get_embedding("hermes3", message.content),
            self.get_embedding("llama2", message.content)
        )
        
        # Compute weighted similarity across models
        similarities = []
        for emb in embeddings:
            sim = cosine_similarity(emb, agent.expertise_embedding)
            similarities.append(sim)
            
        return np.mean(similarities)


@dataclass
class PersonalityModel:
    """Sophisticated personality model for agents."""
    
    traits: Dict[str, float] = field(default_factory=lambda: {
        'openness': 0.0,
        'conscientiousness': 0.0,
        'extraversion': 0.0,
        'agreeableness': 0.0,
        'neuroticism': 0.0
    })
    
    interaction_style: Dict[str, float] = field(default_factory=lambda: {
        'assertiveness': 0.0,
        'cooperation': 0.0,
        'creativity': 0.0,
        'analytical': 0.0
    })
    
    expertise_confidence: Dict[str, float] = field(default_factory=dict)
    
    def generate_response_style(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Generates response parameters based on personality and context.
        
        Args:
            context: Current discussion context
            
        Returns:
            Dict[str, float]: Response style parameters
        """
        # Compute base style from traits
        style = {
            'temperature': self.traits['openness'] * 0.5 + 0.5,
            'creativity_weight': self.traits['openness'] * 0.7,
            'formality': self.traits['conscientiousness'] * 0.6,
            'engagement': self.traits['extraversion'] * 0.8
        }
        
        # Adjust for context
        if context.get('disagreement_level', 0) > 0.7:
            style['temperature'] *= self.traits['agreeableness']
            
        return style

@dataclass
class PersonalityDrivenAgent(ResearchAgent):
    """Agent with sophisticated personality-driven behavior."""
    
    personality_model: PersonalityModel = field(default_factory=PersonalityModel)
    
    async def generate_response(self, 
                              message: Message, 
                              context: Dict[str, Any]) -> str:
        """Generates response influenced by personality traits.
        
        Args:
            message: Input message
            context: Current discussion context
            
        Returns:
            str: Personality-influenced response
        """
        # Get personality-driven response style
        style = self.personality_model.generate_response_style(context)
        
        # Adjust prompt based on personality
        prompt = self.create_personality_aware_prompt(message, style)
        
        # Generate response with personality-adjusted parameters
        response = await self.client.chat(
            model="hermes3",
            messages=[{"role": "user", "content": prompt}],
            temperature=style['temperature'],
            presence_penalty=style['creativity_weight']
        )
        
        return self.apply_personality_filters(response, style)


if __name__ == "__main__":
    topic = "The impact of artificial general intelligence on human society"
    asyncio.run(run_discussion(topic))
