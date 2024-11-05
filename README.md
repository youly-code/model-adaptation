# Model Adaptation Examples

[![CI Testing](https://github.com/leonvanbokhorst/model-adaptation/actions/workflows/ci.yml/badge.svg)](https://github.com/leonvanbokhorst/model-adaptation/actions/workflows/ci.yml)

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
   - Dynamic content aggregation
   - Inventory management integration
   - Real-time availability tracking
   - Customer review analysis
   - Style recommendation engine

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
   - Temporal embedding analysis
   - Cross-model embedding comparison

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
     - Quantization techniques
     - Model pruning

5. **Synthetic Data Generation** (`05_synthetic_data.py`)
   - Instruction-based data generation
   - Quality assessment with complexity scoring
   - Dataset versioning and management
   - Balanced dataset creation
   - NLTK-based linguistic analysis
   - Data augmentation techniques
   - Quality validation pipelines
   - Automated labeling
   - Domain-specific generation
   - Cross-validation support

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
     - Consensus building algorithms
     - Debate simulation
     - Research methodology adaptation
  
7. **Sentiment Analysis and Research Question Generation** (`07_sentiment_analysis.py`)
   - LLM-based sentiment analysis using Ollama
   - Research question generation for academic topics
   - Asynchronous API endpoints with FastAPI
   - Key features:
     - Sentiment classification (positive/negative/neutral)
     - Detailed sentiment explanations
     - Structured research question generation
     - Main and sub-question formulation
     - Robust error handling and logging
     - Resource management with context managers
     - Response parsing and validation
     - Cached client connections
     - RESTful API endpoints
   - Support for:
     - Custom text analysis
     - Academic research planning
     - Scalable API deployment
     - Performance optimization
     - Structured response formats

## Installation

```bash
pip install -r requirements.txt
```

## Testing

Tests are written using pytest and can be run with:

```bash
pytest
```

For integration tests only:

```bash
pytest -m integration
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
