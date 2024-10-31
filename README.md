# Model Adaptation Examples

This repository contains example code demonstrating various model adaptation techniques covered in the Model Adaptation lectures at Fontys University of Applied Sciences (FICT), minor [AI for Society](https://www.fontys.nl/en/Study-at-Fontys/Exchange-programmes/Artificial-Intelligence-For-Society.htm) by [Leon van Bokhorst](https://github.com/leonvanbokhorst).

## Overview

The codebase explores several key areas of model adaptation:

1. **Prompting Techniques** (`01_prompting.py`)
   - Basic prompting
   - Structured prompts
   - Chain-of-thought
   - Few-shot learning
   - Role playing
   - Task decomposition
   - Zero-shot learning
   - Self-consistency
   - Constrained generation
   - Socratic method
   - Reflective prompting
   - Guided feedback
   - Persona-based prompting
   - Template-based prompting
   - Comparative analysis
   - Iterative refinement
   - Scenario-based prompting
   - Self-verification
   - Logical verification

2. **Retrieval-Augmented Generation (RAG)** (`02_rag.py`)
   - Document storage and embedding generation
   - Semantic similarity search with cosine similarity
   - Query routing based on intent classification
   - LLM-based response generation
   - Multiple specialized demo modes:
     - Combined product information
     - Style and fashion advice
     - Technical specifications
     - Store availability
   - Support for both Ollama and OpenAI backends
   - Comprehensive product knowledge base
   - Dynamic wind event simulation

3. **Semantic Space Visualization** (`03_semantic_space.py`)
   - 2D and 3D visualization of semantic embeddings
   - Multiple dimensionality reduction techniques:
     - t-SNE
     - PCA
     - UMAP
     - MDS
   - Interactive 3D visualizations using Plotly
   - Semantic similarity heatmaps
   - Hierarchical clustering dendrograms
   - Word analogy visualization

4. **Model Fine-tuning** (`04_fine-tuning.py`)
   - LoRA-based fine-tuning of LLMs
   - Optimized training parameters
   - Efficient memory management
   - Key features:
     - Custom tokenizer configuration
     - Dataset preparation and formatting
     - Gradient checkpointing
     - Configurable LoRA parameters
     - Inference optimization
     - Training monitoring and logging
   - Support for:
     - Hugging Face models
     - Custom datasets
     - Instruction fine-tuning
     - Performance optimization

5. **Synthetic Data Generation** (`05_synthetic_data.py`)
   - Instruction-based data generation
   - Quality assessment with complexity scoring
   - Dataset versioning and management
   - Balanced dataset creation
   - NLTK-based linguistic analysis

6. **Multi-Agent Research Simulation** (`06_multi_agent.py`)
   - Dynamic team composition based on research questions
   - Adaptive agents with personality models
   - Semantic routing of discussions
   - Real-time discussion quality monitoring
   - Advanced analytics and visualization
   - Key features:
     - Personality-driven agent behavior
     - Belief system evolution
     - Concept tracking and evolution
     - Discussion quality metrics
     - Knowledge base management
     - External system integration

7. **Swarm Intelligence** (`07_swarm.py`)
   - Boid-based flocking simulation
   - Dynamic wind events and turbulence
   - Adaptive behavior patterns
   - Complex environmental interactions

## Installation

```bash
pip install -r requirements.txt
```

## Testing

Tests are written using pytest and can be run with:

```bash
pytest
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
