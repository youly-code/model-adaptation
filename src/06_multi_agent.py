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
import uuid
import statistics
from typing import Set
import json
import re
import aiohttp

# Update logging configuration to suppress httpx logs
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
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
        return self.content == other.content and self.sender == other.sender


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
class PersonalityModel:
    """Sophisticated personality model using semantic embeddings for trait mapping."""

    traits: Dict[str, float] = field(
        default_factory=lambda: {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
        }
    )
    core_trait_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    trait_embedding_cache: Dict[str, np.ndarray] = field(default_factory=dict)

    async def initialize_trait_embeddings(self, client: AsyncClient):
        """Initialize semantic embeddings for core traits and descriptors."""
        try:
            # Core trait embeddings
            for trait in self.traits.keys():
                response = await client.embeddings(model="hermes3", prompt=trait)
                self.core_trait_embeddings[trait] = np.array(response["embedding"])

            # Initialize with random variations to create unique personalities
            for trait in self.traits:
                self.traits[trait] = np.random.uniform(0.3, 0.8)

            logger.debug(f"Initialized personality traits: {self.traits}")

        except Exception as e:
            logger.error(f"Failed to initialize trait embeddings: {e}")
            # Set default values if initialization fails
            for trait in self.traits:
                self.traits[trait] = 0.5

    def generate_response_style(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Generates response style parameters based on personality traits and context."""
        try:
            # Base parameters
            style = {
                "temperature": 0.7,
                "creativity_weight": 0.5,
                "formality": 0.5,
                "assertiveness": 0.5,
            }

            # Adjust based on personality traits
            if self.traits["openness"] > 0.6:
                style["temperature"] += 0.2
                style["creativity_weight"] += 0.2

            if self.traits["conscientiousness"] > 0.6:
                style["formality"] += 0.2
                style["temperature"] -= 0.1

            if self.traits["extraversion"] > 0.6:
                style["assertiveness"] += 0.2
                style["creativity_weight"] += 0.1

            if self.traits["agreeableness"] > 0.6:
                style["assertiveness"] -= 0.1

            if self.traits["neuroticism"] > 0.6:
                style["temperature"] -= 0.1

            # Normalize values to valid ranges
            for key in style:
                style[key] = max(0.1, min(1.0, style[key]))

            return style

        except Exception as e:
            logger.error(f"Error generating response style: {e}")
            # Return default style if there's an error
            return {
                "temperature": 0.7,
                "creativity_weight": 0.5,
                "formality": 0.5,
                "assertiveness": 0.5,
            }


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
        personality_model (PersonalityModel): Agent's personality model
        belief_system (Dict[str, float]): Agent's belief system
    """

    name: str
    personality: str
    expertise: str
    client: AsyncClient
    chat_history: ChatHistory
    memory: Dict[str, Any] = field(default_factory=dict)
    personality_model: PersonalityModel = field(default_factory=PersonalityModel)
    belief_system: Dict[str, float] = field(default_factory=dict)

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
        try:
            prompt = f"Analyze the sentiment of this message and return a single number between -1 and 1: {message.content}"
            response = await self.client.chat(
                model="hermes3", 
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the first number from the response
            content = response["message"]["content"]
            # Use regex to find the first number in the response
            match = re.search(r'-?\d*\.?\d+', content)
            if match:
                sentiment = float(match.group())
                return max(min(sentiment, 1.0), -1.0)  # Ensure value is between -1 and 1
            return 0.0  # Default neutral sentiment if no number found
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0  # Return neutral sentiment on error

    async def respond(self, topic: str) -> str:
        """Generates a response influenced by personality traits and learning history."""
        context = self.chat_history.get_recent_context()

        # Get personality-driven response style
        style = self.personality_model.generate_response_style(
            {
                "topic": topic,
                "context": context,
                "recent_messages": len(self.chat_history.messages),
            }
        )

        prompt = f"""You are {self.name}, a researcher with {self.expertise}. 
        Your personality is: {self.personality}
        Your current traits are: {self.personality_model.traits}
        
        Topic: {topic}
        Recent conversation:
        {context}
        
        Provide your perspective on the topic, considering:
        1. Your personality traits and expertise
        2. The conversation history
        3. Your current belief system: {self.belief_system}
        
        Keep your response concise (2-3 sentences). End with a question or statement 
        that invites further discussion.
        """

        response = await self.client.chat(
            model="hermes3",
            messages=[{"role": "user", "content": prompt}],
            temperature=style["temperature"],
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
        return {"agreement_level": 0.7, "confidence": 0.8, "expertise_relevance": 0.9}

    async def initialize(self):
        """Initialize agent's personality model and belief system."""
        try:
            # Initialize personality model
            await self.personality_model.initialize_trait_embeddings(self.client)

            # Initialize belief system
            self.belief_system = await self._initialize_belief_system()

            logger.debug(
                f"Agent {self.name} initialized with traits: {self.personality_model.traits}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.name}: {e}")
            # Set default values if initialization fails
            self.personality_model = PersonalityModel()
            self.belief_system = {}

    async def _initialize_belief_system(self) -> Dict[str, float]:
        """Initialize belief system based on expertise."""
        try:
            prompt = f"""Given the expertise in {self.expertise}, identify 3-5 key concepts 
            and their confidence levels (0-1). Return as a simple JSON object with concept:confidence pairs.
            Example format: {{"concept1": 0.8, "concept2": 0.6}}"""

            response = await self.client.chat(
                model="hermes3", 
                messages=[{"role": "user", "content": prompt}]
            )

            content = response["message"]["content"]
            
            # Clean the response to ensure valid JSON
            content = content.strip()
            if not content.startswith('{'): # Find the first {
                content = content[content.find('{'):]
            if not content.endswith('}'): # Find the last }
                content = content[:content.rfind('}')+1]

            try:
                beliefs = json.loads(content)
                # Ensure all values are floats
                return {k: float(v) for k, v in beliefs.items() if isinstance(v, (int, float, str))}
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse belief system JSON: {e}")
                return {"default_belief": 0.5}  # Return default belief if parsing fails
                
        except Exception as e:
            logger.error(f"Failed to initialize belief system: {e}")
            return {"default_belief": 0.5}  # Return default belief on any error


@dataclass
class AdaptiveAgent(ResearchAgent):
    """Agent capable of learning and adapting from discussions."""

    personality_model: PersonalityModel = field(default_factory=PersonalityModel)
    learning_history: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    belief_system: Dict[str, float] = field(default_factory=dict)

    async def respond(self, topic: str) -> str:
        """Generates a response influenced by personality traits and learning history."""
        context = self.chat_history.get_recent_context()

        # Get personality-driven response style
        style = self.personality_model.generate_response_style(
            {
                "topic": topic,
                "context": context,
                "recent_messages": len(self.chat_history.messages),
            }
        )

        prompt = f"""You are {self.name}, a researcher with {self.expertise}. 
        Your personality is: {self.personality}
        Your current traits are: {self.personality_model.traits}
        
        Topic: {topic}
        Recent conversation:
        {context}
        
        Provide your perspective on the topic, considering:
        1. Your personality traits and expertise
        2. The conversation history
        3. Your current belief system: {self.belief_system}
        
        Keep your response concise (2-3 sentences). End with a question or statement 
        that invites further discussion.
        """

        response = await self.client.chat(
            model="hermes3",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    async def learn_from_interaction(self, message: Message, outcome: Dict[str, Any]):
        """Updates agent's knowledge and behavior based on interaction outcomes."""
        # Extract concepts and update belief strengths
        concepts = await self.extract_concepts(message.content)
        for concept in concepts:
            current_belief = self.belief_system.get(concept, 0.5)
            impact = outcome.get("impact", 0)
            new_belief = current_belief * 0.8 + impact * 0.2
            self.belief_system[concept] = max(0, min(1, new_belief))

        # Track learning progress
        self.learning_history["belief_updates"].append(
            {"timestamp": time.time(), "concepts": concepts, "outcome": outcome}
        )

    async def extract_concepts(self, text: str) -> List[str]:
        """Extracts key concepts from text."""
        prompt = f"""Extract key concepts from this text as a comma-separated list:
        Text: {text}
        Concepts:"""

        response = await self.client.chat(
            model="hermes3", messages=[{"role": "user", "content": prompt}]
        )

        return [
            concept.strip() for concept in response["message"]["content"].split(",")
        ]

    async def adapt_personality(self, discussion_metrics: Dict[str, float]):
        """Adjusts personality traits based on discussion effectiveness."""
        effectiveness = discussion_metrics.get("effectiveness", 0.5)
        engagement = discussion_metrics.get("engagement", 0.5)

        if effectiveness < 0.3:
            self.personality_model.traits["agreeableness"] += 0.1
            self.personality_model.traits["openness"] += 0.1
        elif engagement < 0.3:
            self.personality_model.traits["extraversion"] += 0.1

        # Normalize traits to [0,1] range
        for trait in self.personality_model.traits:
            self.personality_model.traits[trait] = max(
                0, min(1, self.personality_model.traits[trait])
            )


class SemanticRouter:
    """Routes messages to the most relevant agent based on semantic similarity.

    Attributes:
        agents (List[ResearchAgent]): List of available agents
    """

    def __init__(self, agents: List[AdaptiveAgent]):
        self.agents = agents
        self.routing_history = []
        self.concept_graph = nx.DiGraph()
        self.embedding_cache = {}  # Cache for expertise embeddings

    async def get_cached_embedding(self, text: str) -> np.ndarray:
        """Gets embedding from cache or generates new one."""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = await self.agents[0].get_embedding(text)
        return self.embedding_cache[text]

    async def route_message(self, message: Message) -> Tuple[AdaptiveAgent, float]:
        """Selects the most appropriate agent to respond based on message content."""
        message_embedding = await self.get_cached_embedding(message.content)
        similarities = []
        agent_list = []  # Add this list to maintain order
        
        # Filter available agents and store in a list
        available_agents = [
            agent for agent in self.agents 
            if agent.name != message.sender
        ]

        if not available_agents:
            available_agents = [
                agent for agent in self.agents 
                if agent.name != message.sender
            ]
        
        # Calculate similarities
        for agent in available_agents:
            expertise_embedding = await self.get_cached_embedding(agent.expertise)
            similarity = cosine_similarity(
                message_embedding.reshape(1, -1), 
                expertise_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)
            agent_list.append(agent)  # Keep track of agents in same order

        # Convert to numpy array for calculations
        similarities = np.array(similarities)
        
        if len(similarities) == 0:
            # Fallback if no similarities calculated
            return available_agents[0], 0.0
        
        # Calculate probabilities
        probabilities = np.exp(similarities) / np.sum(np.exp(similarities))
        temperature = 0.8
        probabilities = np.power(probabilities, 1 / temperature)
        probabilities /= np.sum(probabilities)

        # Select agent using index to maintain correct mapping
        selected_idx = np.random.choice(len(agent_list), p=probabilities)
        selected_agent = agent_list[selected_idx]
        confidence = similarities[selected_idx]

        return selected_agent, confidence


class TeamComposer:
    """Dynamically composes research teams based on research questions."""

    def __init__(self, client: AsyncClient):
        self.client = client
        self.expertise_cache = {}

    async def compose_team(self, research_question: str) -> List[Dict[str, Any]]:
        """Analyzes research question and generates optimal team composition."""
        try:
            logger.debug(
                f"\n=== Starting Team Composition for Research Question ===\n{research_question}"
            )

            prompt = f"""Analyze this research question and identify:
            1. Key knowledge domains required (list 4-6)
            2. Critical perspectives needed
            3. Potential cognitive biases to address
            4. Methodological requirements

            Research Question: {research_question}

            Format your response as JSON with these keys: domains, perspectives, biases, methods"""

            logger.debug(f"Sending analysis prompt to LLM:\n{prompt}")

            response = await self.client.chat(
                model="hermes3", messages=[{"role": "user", "content": prompt}]
            )

            logger.debug(
                f"Received analysis response:\n{response['message']['content']}"
            )

            analysis = json.loads(response["message"]["content"])
            logger.debug(f"Parsed analysis:\n{json.dumps(analysis, indent=2)}")

            team_specs = []

            # Generate domain experts
            logger.info("\n=== Generating Domain Experts ===")
            for domain in analysis["domains"]:
                logger.debug(f"\nGenerating expert for domain: {domain}")
                expert_spec = await self._generate_expert_spec(
                    domain, analysis["perspectives"], analysis["biases"]
                )
                logger.debug(
                    f"Generated expert spec:\n{json.dumps(expert_spec, indent=2)}"
                )
                team_specs.append(expert_spec)

            # Add methodology specialist
            if analysis["methods"]:
                logger.info("\n=== Generating Methodology Specialist ===")
                method_spec = await self._generate_methodologist_spec(
                    analysis["methods"]
                )
                logger.debug(
                    f"Generated methodologist spec:\n{json.dumps(method_spec, indent=2)}"
                )
                team_specs.append(method_spec)

            # Add integrator/synthesizer
            logger.info("\n=== Generating Synthesizer ===")
            synth_spec = await self._generate_synthesizer_spec(
                analysis["domains"], analysis["perspectives"]
            )
            logger.debug(
                f"Generated synthesizer spec:\n{json.dumps(synth_spec, indent=2)}"
            )
            team_specs.append(synth_spec)

            logger.info(
                f"\n=== Team Composition Complete ===\nTotal agents: {len(team_specs)}"
            )
            return team_specs

        except Exception as e:
            logger.error(f"Error composing team: {str(e)}")
            raise

    async def _generate_expert_spec(
        self, domain: str, perspectives: List[str], biases: List[str]
    ) -> Dict[str, Any]:
        """Generates specification for a domain expert."""
        prompt = f"""Create a research agent specialized in {domain}.
        Consider these perspectives: {perspectives}
        And potential biases: {biases}

        Generate:  
        1. A realistic academic full name
        2. Detailed expertise description
        3. Clear personality description
        4. Personality traits that would be valuable
        5. Potential cognitive biases to be aware of
        6. Unique perspective they bring

        Format as JSON with keys: name, expertise, personality, traits, biases, perspective"""

        logger.debug(f"Sending expert generation prompt:\n{prompt}")

        response = await self.client.chat(
            model="hermes3", messages=[{"role": "user", "content": prompt}]
        )

        try:
            spec = json.loads(response["message"]["content"])
            # Ensure personality is present
            if "personality" not in spec:
                spec["personality"] = f"Expert researcher in {domain}"
            return spec
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse expert spec: {e}")
            # Return a default spec
            return {
                "name": f"Dr. {domain.title().replace(' ', '')}",
                "expertise": domain,
                "personality": f"Expert researcher in {domain}",
                "traits": ["analytical", "thorough", "objective"],
                "biases": ["confirmation bias"],
                "perspective": f"Specialized knowledge in {domain}",
            }

    async def _generate_methodologist_spec(self, methods: List[str]) -> Dict[str, Any]:
        """Generates specification for methodology specialist."""
        prompt = f"""Create a research agent specialized in research methodology.
        Key methods to cover: {methods}

        Generate a specification with these exact fields:
        {{
            "name": "a fitting name for the methodology specialist",
            "expertise": "detailed description of their methodological expertise",
            "personality": "description of their personality traits",
            "traits": ["list", "of", "key", "personality", "traits"],
            "biases": ["list", "of", "potential", "biases"],
            "perspective": ["list", "of", "unique", "perspectives"]
        }}

        Return ONLY valid JSON, no other text.
        """

        logger.debug(f"Sending methodologist prompt:\n{prompt}")

        try:
            response = await self.client.chat(
                model="hermes3", messages=[{"role": "user", "content": prompt}]
            )

            logger.debug(
                f"Received methodologist response:\n{response['message']['content']}"
            )

            # Try to clean and parse the JSON
            content = response["message"]["content"].strip()
            if not content.startswith("{"):
                # Try to find JSON block
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]
                else:
                    raise ValueError("No JSON object found in response")

            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Attempted to parse: {content}")
                # Return a default methodologist spec as fallback
                return {
                    "name": "Dr. Methodicus",
                    "expertise": "Research methodology and experimental design",
                    "personality": "Rigorous, systematic, and detail-oriented",
                    "traits": {
                        "analytical": 0.9,
                        "thoroughness": 0.8,
                        "objectivity": 0.9,
                        "precision": 0.85,
                        "skepticism": 0.7,
                    },
                    "biases": ["confirmation bias", "methodological purism"],
                    "perspective": "Ensures scientific rigor and methodological soundness",
                }

        except Exception as e:
            logger.error(f"Error generating methodologist spec: {e}")
            raise

    async def _generate_synthesizer_spec(
        self, domains: List[str], perspectives: List[str]
    ) -> Dict[str, Any]:
        """Generates specification for integration/synthesis specialist."""
        prompt = f"""Create a research agent specialized in integrating multiple perspectives.
        Domains to integrate: {domains}
        Perspectives to consider: {perspectives}

        Generate a specification with these exact fields:
        {{
            "name": "a fitting name for the synthesis specialist",
            "expertise": "detailed description of their integrative expertise",
            "personality": "description of their personality traits",
            "traits": ["list", "of", "key", "personality", "traits"],
            "biases": ["list", "of", "potential", "biases"],
            "perspective": ["list", "of", "unique", "perspectives"]
        }}

        Return ONLY valid JSON, no other text.
        """

        logger.debug(f"Sending synthesizer prompt:\n{prompt}")

        try:
            response = await self.client.chat(
                model="hermes3", messages=[{"role": "user", "content": prompt}]
            )

            logger.debug(
                f"Received synthesizer response:\n{response['message']['content']}"
            )

            # Clean and parse the JSON
            content = response["message"]["content"].strip()
            if not content.startswith("{"):
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    content = content[start:end]
                else:
                    raise ValueError("No JSON object found in response")

            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                logger.error(f"Attempted to parse: {content}")
                # Return a default synthesizer spec as fallback
                return {
                    "name": "Dr. Synthesis",
                    "expertise": "Integration of multidisciplinary perspectives and knowledge synthesis",
                    "personality": "Balanced, inclusive, and holistic thinker",
                    "traits": {
                        "integrative_thinking": 0.9,
                        "open_mindedness": 0.8,
                        "communication": 0.9,
                        "diplomacy": 0.85,
                        "systems_thinking": 0.9,
                    },
                    "biases": ["holistic bias", "consensus seeking bias"],
                    "perspective": "Specializes in finding connections and synthesizing insights across different domains",
                }

        except Exception as e:
            logger.error(f"Error generating synthesizer spec: {e}")
            raise


async def initialize_agents(
    client: AsyncClient, chat_history: ChatHistory, research_question: str = None
) -> List[AdaptiveAgent]:
    """Initialize the agent pool with dynamically generated team."""

    try:
        logger.debug(
            f"\n=== Initializing Dynamic Team for Research Question ===\n{research_question}"
        )
        composer = TeamComposer(client)

        try:
            team_specs = await composer.compose_team(research_question)
        except Exception as e:
            logger.error(f"Failed to compose team: {str(e)}")
            # Provide minimal emergency fallback
            return [
                AdaptiveAgent(
                    name="Emergency Backup Agent",
                    personality="Balanced and adaptable researcher",
                    expertise="General discussion and problem-solving",
                    client=client,
                    chat_history=chat_history,
                )
            ]

        logger.debug("\n=== Creating Agent Instances ===")
        agents = []
        for spec in team_specs:
            logger.debug(f"\nCreating agent from spec:\n{json.dumps(spec, indent=2)}")

            agent = AdaptiveAgent(
                name=spec["name"],
                personality=spec.get("personality", "Balanced researcher"),
                expertise=spec["expertise"],
                client=client,
                chat_history=chat_history,
            )

            # Initialize the agent
            await agent.initialize()
            agents.append(agent)

            logger.debug(f"Created agent: {agent.name}")
            logger.debug(f"Personality: {agent.personality}")
            logger.debug(f"Expertise: {agent.expertise}")

        return agents

    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}")
        raise


async def run_discussion(
    topic: str,
    num_turns: int = 5,
    min_agents: int = 2,
    max_consecutive: int = 2,
    research_question: str = None,
):
    """Orchestrates a multi-turn discussion between agents on a given topic."""
    client = AsyncClient()
    chat_history = ChatHistory()
    conversation_manager = ConversationManager(chat_history)
    semantic_processor = SemanticProcessor(client)  # Create instance

    try:
        # Initialize agents with research question
        research_question = research_question or topic
        agents: List[AdaptiveAgent] = await initialize_agents(
            client, chat_history, research_question
        )

        # Initialize each agent
        for agent in agents:
            await agent.initialize()

        router = SemanticRouter(agents)

        # Log the team composition
        logger.info("\n=== Research Team ===")
        for agent in agents:
            logger.info(f"\nAgent: {agent.name}")
            logger.info(f"Expertise: {agent.expertise}")
            logger.info(f"Personality: {agent.personality}")

        # Initial message to start discussion
        initial_message = Message(topic, "System")
        chat_history.add_message(initial_message)

        current_agent, confidence = await router.route_message(initial_message)
        logger.info(
            f"Selected initial agent {current_agent.name} with confidence {confidence:.2f}"
        )

        last_speaker = None
        consecutive_turns = 0

        # Main discussion loop
        for turn in range(num_turns):
            logger.info(f"\nTurn {turn + 1}:")

            # Generate response
            response = await current_agent.respond(topic)
            message = Message(response, current_agent.name)
            chat_history.add_message(message)
            logger.info(f"{current_agent.name}: {response}")

            # Route to next agent
            next_agent, next_confidence = await router.route_message(message)

            # Prevent same agent from speaking too many times in a row
            while next_agent.name == current_agent.name:
                logger.info("Avoiding repeat speaker, selecting new agent...")
                next_agent, next_confidence = await router.route_message(message)

            logger.info(
                f"Selected next agent {next_agent.name} with confidence {next_confidence:.2f}"
            )
            current_agent = next_agent
            await asyncio.sleep(1)  # Rate limiting

        # After discussion completes, generate summary and reflections
        logger.info("\n=== Discussion Summary ===")

        # Get discussion quality metrics
        final_quality = await conversation_manager.evaluate_discussion_quality()
        logger.info(f"Overall Discussion Quality: {final_quality:.2f}")

        # Track concept evolution using instance
        concept_trajectories = await semantic_processor.track_concept_evolution(
            chat_history.messages
        )
        logger.info("\nKey Concept Evolution:")
        for concept, trajectory in concept_trajectories.items():
            logger.info(f"{concept}: {np.mean(trajectory):.2f}")

        # Get agent reflections
        logger.info("\n=== Agent Reflections ===")
        for agent in agents:
            reflection_prompt = f"""
            As {agent.name}, reflect on the discussion about '{topic}':
            1. What were your key insights?
            2. How did your perspective evolve?
            3. What surprised you most?
            
            Keep your reflection concise (2-3 sentences).
            
            Previous discussion:
            {chat_history.get_recent_context()}
            """

            reflection = await agent.client.chat(
                model="hermes3",
                messages=[{"role": "user", "content": reflection_prompt}],
            )
            logger.info(
                f"\n{agent.name}'s Reflection:\n{reflection['message']['content']}"
            )

            # Log learning outcomes
            logger.info(f"\nLearning Metrics for {agent.name}:")
            for concept, belief in agent.belief_system.items():
                logger.info(f"- {concept}: {belief:.2f}")

        logger.info("\nDiscussion completed")

    except KeyboardInterrupt:
        logger.info("\nDiscussion terminated by user.")
    except Exception as e:
        logger.error(f"Error running discussion: {str(e)}")
        logger.info("Try starting the Ollama service with: ollama serve")


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

    def __init__(self, chat_history: ChatHistory):
        self.chat_history = chat_history
        self.quality_metrics = defaultdict(list)

    async def evaluate_discussion_quality(self) -> float:
        """Scores conversation based on quality metrics."""
        if len(self.chat_history.messages) < 2:
            return 0.0

        scores = {
            "diversity": await self._measure_viewpoint_diversity(),
            "coherence": await self._measure_coherence(),
            "depth": await self._measure_insight_depth(),
            "balance": self._measure_participation_balance(),
        }

        weights = {"diversity": 0.3, "coherence": 0.3, "depth": 0.2, "balance": 0.2}
        return sum(score * weights[metric] for metric, score in scores.items())

    async def _measure_insight_depth(self) -> float:
        """Measures the depth and sophistication of insights in the discussion."""
        if not self.chat_history.messages:
            return 0.0

        # Analyze the last few messages for depth indicators
        recent_messages = self.chat_history.messages[-5:]  # Look at last 5 messages
        depth_indicators = {
            "references_concepts": 0.0,  # References to established concepts
            "builds_on_previous": 0.0,  # Builds on previous points
            "novel_insights": 0.0,  # Introduces new perspectives
            "complexity": 0.0,  # Complexity of language and ideas
        }

        for msg in recent_messages:
            # Count concept references (simplified)
            concept_count = len(
                re.findall(
                    r"\b(consciousness|quantum|cognition|intelligence|emergence)\b",
                    msg.content.lower(),
                )
            )
            depth_indicators["references_concepts"] += min(1.0, concept_count / 3)

            # Check if message builds on previous points
            if any(
                prev.content.lower() in msg.content.lower()
                for prev in self.chat_history.messages[:-1]
            ):
                depth_indicators["builds_on_previous"] += 0.2

            # Estimate complexity through sentence length and structure
            sentences = msg.content.split(".")
            avg_length = statistics.mean(
                [len(s.split()) for s in sentences if s.strip()]
            )
            depth_indicators["complexity"] += min(1.0, avg_length / 20)

        # Average the indicators
        depth_score = statistics.mean(depth_indicators.values())
        return min(1.0, depth_score)

    async def _measure_viewpoint_diversity(self) -> float:
        """Measures diversity of perspectives in discussion."""
        unique_senders = len(set(msg.sender for msg in self.chat_history.messages))
        return min(1.0, unique_senders / 3)  # Normalize to [0,1]

    async def _measure_coherence(self) -> float:
        """Measures semantic coherence between consecutive messages."""
        if len(self.chat_history.messages) < 2:
            return 1.0

        coherence_scores = []
        for i in range(1, len(self.chat_history.messages)):
            prev_msg = self.chat_history.messages[i - 1]
            curr_msg = self.chat_history.messages[i]
            if prev_msg.vector is not None and curr_msg.vector is not None:
                similarity = cosine_similarity(
                    prev_msg.vector.reshape(1, -1), curr_msg.vector.reshape(1, -1)
                )[0][0]
                coherence_scores.append(similarity)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    async def suggest_direction(self) -> str:
        """Suggests new angles or topics to explore"""

    async def detect_convergence(self) -> bool:
        """Determines if discussion has reached natural conclusion"""

    def _measure_participation_balance(self) -> float:
        """Measures how evenly distributed participation is among agents.

        Returns:
            float: Score between 0-1, where 1 indicates perfectly balanced participation
        """
        if not self.chat_history.messages:
            return 1.0

        # Count messages per agent
        participation = defaultdict(int)
        for message in self.chat_history.messages:
            if message.sender != "System":  # Exclude system messages
                participation[message.sender] += 1

        if not participation:  # If no agent messages
            return 1.0

        # Calculate standard deviation of participation
        counts = list(participation.values())
        mean = statistics.mean(counts)
        if mean == 0:
            return 1.0

        try:
            std_dev = statistics.stdev(counts)
            # Normalize to 0-1 range (0 = high imbalance, 1 = perfect balance)
            # Using coefficient of variation (CV) = std_dev / mean
            cv = std_dev / mean
            balance_score = 1 / (1 + cv)  # Convert to 0-1 scale where 1 is better
            return max(0.0, min(1.0, balance_score))  # Ensure bounds
        except statistics.StatisticsError:  # If only one agent
            return 1.0


class SemanticProcessor:
    """Advanced semantic processing capabilities"""

    def __init__(self, client: AsyncClient):
        self.client = client
        self.concept_cache = {}

    async def track_concept_evolution(
        self, messages: List[Message]
    ) -> Dict[str, List[float]]:
        """Tracks concept development through conversation."""
        concept_trajectories = defaultdict(list)

        try:
            for msg in messages:
                concepts = await self.extract_key_concepts(msg.content)
                for concept in concepts:
                    if concept not in self.concept_cache:
                        self.concept_cache[concept] = await self._get_concept_embedding(
                            concept
                        )

                    if msg.vector is not None:
                        similarity = cosine_similarity(
                            self.concept_cache[concept].reshape(1, -1),
                            msg.vector.reshape(1, -1),
                        )[0][0]
                        concept_trajectories[concept].append(similarity)

            return dict(concept_trajectories)
        except Exception as e:
            logger.error(f"Error tracking concept evolution: {e}")
            return {}

    async def extract_key_concepts(self, text: str) -> List[str]:
        """Identifies main concepts from text using LLM."""
        try:
            prompt = f"""Extract key concepts from this text as a comma-separated list:
            Text: {text}
            Concepts:"""

            response = await self.client.chat(
                model="hermes3", messages=[{"role": "user", "content": prompt}]
            )

            concepts = [
                concept.strip() for concept in response["message"]["content"].split(",")
            ]
            return concepts[:5]  # Limit to top 5 concepts
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []

    async def _get_concept_embedding(self, concept: str) -> np.ndarray:
        """Get embedding for a concept."""
        try:
            response = await self.client.embeddings(model="hermes3", prompt=concept)
            return np.array(response["embedding"])
        except Exception as e:
            logger.error(f"Error getting concept embedding: {e}")
            return np.zeros(1536)  # Return zero vector as fallback


class DiscussionPool:
    """Manages multiple parallel discussions with resource optimization."""

    def __init__(self, max_concurrent: int = 5):
        self.discussions: Dict[str, Dict[str, Any]] = {}
        self.max_concurrent = max_concurrent
        self.metrics = Metrics()
        self.active_agents: Dict[str, Set[str]] = defaultdict(set)
        self.health_stats: Dict[str, List[float]] = defaultdict(list)

    async def schedule_discussion(
        self, topic: str, agents: List[ResearchAgent], config: Dict[str, Any] = None
    ) -> str:
        """Schedules new discussion with available agents.

        Args:
            topic: Discussion topic
            agents: List of available agents
            config: Optional configuration parameters

        Returns:
            str: Discussion ID

        Raises:
            ResourceError: If system capacity exceeded
        """
        if len(self.discussions) >= self.max_concurrent:
            await self._wait_for_slot()

        discussion_id = str(uuid.uuid4())

        # Default configuration
        default_config = {
            "max_turns": 20,
            "min_agents": 2,
            "max_consecutive": 2,
            "timeout": 3600,  # 1 hour
        }
        config = {**default_config, **(config or {})}

        # Initialize discussion state
        self.discussions[discussion_id] = {
            "topic": topic,
            "agents": agents,
            "status": "scheduled",
            "start_time": None,
            "messages": [],
            "config": config,
            "metrics": defaultdict(list),
        }

        # Update agent availability
        for agent in agents:
            self.active_agents[agent.name].add(discussion_id)

        return discussion_id

    async def balance_load(self):
        """Optimizes resource utilization across discussions."""
        # Calculate current load metrics
        agent_loads = self._calculate_agent_loads()
        discussion_priorities = self._calculate_discussion_priorities()

        # Identify overloaded and underutilized agents
        overloaded = {
            agent: load for agent, load in agent_loads.items() if load > 0.8
        }  # 80% threshold
        underutilized = {
            agent: load for agent, load in agent_loads.items() if load < 0.3
        }  # 30% threshold

        if overloaded:
            await self._rebalance_agents(
                overloaded, underutilized, discussion_priorities
            )

        # Adjust discussion parameters based on load
        await self._adjust_discussion_parameters()

    async def monitor_health(self) -> Dict[str, Any]:
        """Tracks system performance metrics and health indicators.

        Returns:
            Dict containing health metrics and status indicators
        """
        current_health = {
            "active_discussions": len(self.discussions),
            "agent_utilization": self._calculate_agent_utilization(),
            "response_times": self._calculate_response_metrics(),
            "error_rates": self._calculate_error_rates(),
            "memory_usage": self._get_memory_usage(),
            "discussion_quality": await self._evaluate_discussion_quality(),
        }

        # Update historical stats
        for metric, value in current_health.items():
            self.health_stats[metric].append(value)

        # Add trend analysis
        current_health["trends"] = self._analyze_trends()

        return current_health

    async def _wait_for_slot(self, timeout: float = 60.0):
        """Waits for a discussion slot to become available."""
        start_time = time.time()
        while len(self.discussions) >= self.max_concurrent:
            if time.time() - start_time > timeout:
                raise ResourceError("Timeout waiting for available discussion slot")
            await asyncio.sleep(1)
            await self._cleanup_completed_discussions()

    def _calculate_agent_loads(self) -> Dict[str, float]:
        """Calculates current load for each agent."""
        loads = defaultdict(float)
        for discussion in self.discussions.values():
            if discussion["status"] != "completed":
                for agent in discussion["agents"]:
                    loads[agent.name] += 1 / len(discussion["agents"])
        return {name: load / self.max_concurrent for name, load in loads.items()}

    def _calculate_discussion_priorities(self) -> Dict[str, float]:
        """Assigns priority scores to active discussions."""
        priorities = {}
        for id, discussion in self.discussions.items():
            if discussion["status"] != "completed":
                # Calculate priority based on multiple factors
                priority = 0.0
                # Age factor (older discussions get higher priority)
                if discussion["start_time"]:
                    age = time.time() - discussion["start_time"]
                    priority += min(1.0, age / 3600)  # Max age factor of 1.0
                # Progress factor
                messages = len(discussion["messages"])
                target = discussion["config"]["max_turns"]
                progress = messages / target
                priority += (1 - progress) * 0.5  # Prioritize completing discussions
                # Quality factor
                quality_metrics = discussion["metrics"].get("quality", [])
                if quality_metrics:
                    priority += statistics.mean(quality_metrics) * 0.3

                priorities[id] = priority

        return priorities

    async def _rebalance_agents(
        self,
        overloaded: Dict[str, float],
        underutilized: Dict[str, float],
        priorities: Dict[str, float],
    ):
        """Rebalances agent assignments based on load and priorities."""
        for agent, load in overloaded.items():
            # Find discussions where agent can be replaced
            agent_discussions = self.active_agents[agent]
            for discussion_id in sorted(
                agent_discussions, key=lambda x: priorities.get(x, 0)
            ):
                if underutilized:
                    # Replace with underutilized agent
                    new_agent = min(underutilized.items(), key=lambda x: x[1])[0]
                    await self._transfer_agent(discussion_id, agent, new_agent)
                    underutilized[new_agent] += 1 / self.max_concurrent
                    if underutilized[new_agent] > 0.3:
                        del underutilized[new_agent]

    async def _adjust_discussion_parameters(self):
        """Adjusts discussion parameters based on system load."""
        system_load = len(self.discussions) / self.max_concurrent
        for discussion in self.discussions.values():
            if discussion["status"] == "active":
                if system_load > 0.9:  # High load
                    # Reduce max turns and increase rate limiting
                    discussion["config"]["max_turns"] = max(
                        5, discussion["config"]["max_turns"] - 5
                    )
                    discussion["config"]["rate_limit"] *= 1.5
                elif system_load < 0.5:  # Low load
                    # Allow more turns and faster responses
                    discussion["config"]["max_turns"] += 2
                    discussion["config"]["rate_limit"] = max(
                        1.0, discussion["config"]["rate_limit"] * 0.8
                    )

    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculates error rates for different components."""
        error_counts = defaultdict(int)
        total_operations = defaultdict(int)

        for discussion in self.discussions.values():
            for metric in discussion["metrics"]:
                if metric.startswith("error_"):
                    component = metric.split("error_")[1]
                    error_counts[component] += sum(
                        1 for x in discussion["metrics"][metric] if x
                    )
                    total_operations[component] += len(discussion["metrics"][metric])

        return {
            component: count / total_operations[component]
            for component, count in error_counts.items()
            if total_operations[component] > 0
        }

    async def _evaluate_discussion_quality(self) -> float:
        """Evaluates overall quality of active discussions."""
        if not self.discussions:
            return 1.0

        quality_scores = []
        for discussion in self.discussions.values():
            if discussion["status"] == "active":
                metrics = discussion["metrics"]
                # Combine multiple quality indicators
                engagement = statistics.mean(metrics.get("engagement", [1.0]))
                coherence = statistics.mean(metrics.get("coherence", [1.0]))
                diversity = statistics.mean(metrics.get("diversity", [1.0]))

                quality = engagement * 0.4 + coherence * 0.4 + diversity * 0.2
                quality_scores.append(quality)

        return statistics.mean(quality_scores) if quality_scores else 1.0

    def _analyze_trends(self) -> Dict[str, str]:
        """Analyzes trends in health metrics."""
        trends = {}
        for metric, values in self.health_stats.items():
            if len(values) >= 3:
                recent = values[-3:]
                if all(b > a for a, b in zip(recent, recent[1:])):
                    trends[metric] = "increasing"
                elif all(b < a for a, b in zip(recent, recent[1:])):
                    trends[metric] = "decreasing"
                else:
                    trends[metric] = "stable"
            else:
                trends[metric] = "insufficient_data"
        return trends


class KnowledgeBase:
    """Manages shared knowledge across discussions"""

    def __init__(self):
        self.insights = []
        self.concept_graph = nx.DiGraph()

    async def store_insight(self, insight: Dict[str, Any]):
        """Stores valuable insights with metadata."""
        insight["timestamp"] = time.time()
        insight["references"] = []

        # Link related insights
        for existing in self.insights:
            if await self._are_insights_related(insight, existing):
                insight["references"].append(existing["id"])

        self.insights.append(insight)
        await self._update_concept_graph(insight)

    async def query_relevant_knowledge(self, context: str) -> List[Dict]:
        """Retrieves relevant past insights using semantic search."""
        if not self.insights:
            return []

        context_embedding = await self._get_embedding(context)
        scores = []

        for insight in self.insights:
            if "embedding" in insight:
                similarity = cosine_similarity(
                    context_embedding.reshape(1, -1),
                    insight["embedding"].reshape(1, -1),
                )[0][0]
                scores.append((similarity, insight))

        scores.sort(reverse=True)
        return [insight for _, insight in scores[:3]]

    async def _are_insights_related(
        self, insight1: Dict[str, Any], insight2: Dict[str, Any]
    ) -> bool:
        """Determines if two insights are semantically related."""
        # Get embeddings if not already present
        if "embedding" not in insight1:
            insight1["embedding"] = await self._get_embedding(insight1["content"])
        if "embedding" not in insight2:
            insight2["embedding"] = await self._get_embedding(insight2["content"])

        similarity = cosine_similarity(
            insight1["embedding"].reshape(1, -1), insight2["embedding"].reshape(1, -1)
        )[0][0]

        return similarity > 0.7  # Threshold for relatedness

    async def _update_concept_graph(self, insight: Dict[str, Any]):
        """Updates concept graph with new insight."""
        # Extract concepts from insight
        concepts = await self._extract_concepts(insight["content"])

        # Add nodes and edges
        for concept in concepts:
            if concept not in self.concept_graph:
                self.concept_graph.add_node(concept, weight=1)
            else:
                self.concept_graph.nodes[concept]["weight"] += 1

        # Connect related concepts
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1 :]:
                if self.concept_graph.has_edge(concept1, concept2):
                    self.concept_graph[concept1][concept2]["weight"] += 1
                else:
                    self.concept_graph.add_edge(concept1, concept2, weight=1)

    async def _extract_concepts(self, text: str) -> List[str]:
        """Extracts key concepts from text."""
        prompt = f"""Extract key concepts from this text as a comma-separated list:
        Text: {text}
        Concepts:"""

        response = await self.client.chat(
            model="hermes3", messages=[{"role": "user", "content": prompt}]
        )

        return [c.strip() for c in response["message"]["content"].split(",")]


class DiscussionAnalytics:
    """Advanced analytics for discussions"""

    def analyze_agent_influence(self) -> Dict[str, float]:
        """Measures each agent's impact on discussion.

        Returns:
            Dict mapping agent names to influence scores
        """
        influence_scores = {}

        # Calculate influence based on:
        # 1. How often others reference their points
        # 2. Sentiment impact of their messages
        # 3. Topic steering capability

        return influence_scores

    def identify_breakthrough_moments(self) -> List[int]:
        """Identifies key turning points in discussion.

        Returns:
            List of message indices representing breakthroughs
        """
        breakthroughs = []

        # Identify moments where:
        # 1. Multiple agents converge on new understanding
        # 2. Novel concepts are introduced
        # 3. Significant perspective shifts occur

        return breakthroughs

    def generate_discussion_graph(self) -> nx.Graph:
        """Creates network graph of concept relationships.

        Returns:
            NetworkX graph representing concept connections
        """
        graph = nx.Graph()

        # Build graph with:
        # 1. Nodes as key concepts
        # 2. Edges as semantic relationships
        # 3. Weights based on relationship strength

        return graph


class ExternalIntegration:
    """Handles external system integration"""

    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.webhook_urls = {}

    async def export_insights(
        self, insights: List[Dict[str, Any]], format: str = "json"
    ) -> str:
        """Exports discussion insights in specified format."""
        if format == "json":
            return json.dumps(insights, indent=2)
        elif format == "csv":
            # Convert to CSV format
            headers = ["timestamp", "content", "agent", "references"]
            rows = [
                [
                    insight["timestamp"],
                    insight["content"],
                    insight["agent"],
                    ",".join(insight["references"]),
                ]
                for insight in insights
            ]

            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(headers)
            writer.writerows(rows)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def import_external_knowledge(
        self, source: str, content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Imports knowledge from external sources."""
        if source == "api":
            # Make API request to external system
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/knowledge",
                    json=content,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ) as response:
                    return await response.json()
        elif source == "file":
            # Import from file
            with open(content["path"], "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported knowledge source: {source}")

    async def webhook_notifications(self, event: str, payload: Dict[str, Any]):
        """Sends notifications about significant events."""
        if event not in self.webhook_urls:
            logger.warning(f"No webhook URL configured for event: {event}")
            return

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.webhook_urls[event],
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook notification failed: {response.status}")
                    return await response.json()
            except Exception as e:
                logger.error(f"Failed to send webhook notification: {str(e)}")


@dataclass
class PersonalityDrivenAgent(ResearchAgent):
    """Agent with sophisticated personality-driven behavior."""

    personality_model: PersonalityModel = field(default_factory=PersonalityModel)

    async def generate_response(self, message: Message, context: Dict[str, Any]) -> str:
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
            temperature=style["temperature"],
            presence_penalty=style["creativity_weight"],
        )

        return self.apply_personality_filters(response, style)

    def create_personality_aware_prompt(
        self, message: Message, style: Dict[str, float]
    ) -> str:
        """Creates a prompt that incorporates personality traits and style.

        Args:
            message: Input message to respond to
            style: Response style parameters

        Returns:
            str: Personality-aware prompt
        """
        formality_level = "formal" if style["formality"] > 0.7 else "casual"
        assertiveness = (
            "confidently" if style["assertiveness"] > 0.7 else "thoughtfully"
        )

        prompt = f"""As {self.name}, a researcher with expertise in {self.expertise},
        respond to this message: "{message.content}"
        
        Your personality traits are: {self.personality_model.traits}
        Respond in a {formality_level} tone, speaking {assertiveness}.
        
        Consider:
        1. Your expertise and background
        2. Your current personality traits
        3. The discussion context
        
        Keep your response focused and engaging, maintaining your unique perspective.
        """
        return prompt

    def apply_personality_filters(
        self, response: Dict[str, str], style: Dict[str, float]
    ) -> str:
        """Applies personality-based filtering to the response.

        Args:
            response: Raw LLM response
            style: Response style parameters

        Returns:
            str: Filtered and adjusted response
        """
        content = response["message"]["content"]

        # Adjust response based on personality traits
        if style["formality"] > 0.7:
            # Replace casual language with more formal alternatives
            content = content.replace("yeah", "yes")
            content = content.replace("gonna", "going to")

        if style["assertiveness"] > 0.7:
            # Strengthen statements
            content = content.replace("might", "will")
            content = content.replace("could", "should")

        return content


if __name__ == "__main__":
    research_question = "What are the key opportunities for generative AI in education?"

    try:
        asyncio.run(
            run_discussion(
                topic="Generative AI in Education",
                research_question=research_question,
                num_turns=20,
                min_agents=3,  # Min agents in the discussion
                max_consecutive=2,  # Max consecutive turns for the same agent
            )
        )
    except KeyboardInterrupt:
        logger.info("\nDiscussion terminated by user.")
    except Exception as e:
        logger.error(f"Error running discussion: {str(e)}")
        logger.info("Try starting the Ollama service with: ollama serve")
