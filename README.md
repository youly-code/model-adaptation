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
  
7. **Sentiment Analysis** (`07_sentiment_analysis.py`)
   - LLM-based sentiment analysis using Ollama
   - Asynchronous API endpoints with FastAPI
   - Key features:
     - Sentiment classification (positive/negative/neutral)
     - Detailed sentiment explanations
     - Robust error handling and logging
     - Resource management with context managers
     - Response parsing and validation
     - Cached client connections
     - RESTful API endpoints
   - Support for:
     - Custom text analysis
     - Scalable API deployment
     - Performance optimization
     - Structured response formats

8. **Research Question Generation** (`08_research_questions.py`)
   - LLM-based research question generation using Ollama
   - FastAPI-based REST endpoints
   - Key features:
     - Main question generation
     - Sub-question derivation
     - Academic rigor validation
     - Structured response format
   - Support for:
     - Asynchronous processing
     - Resource management
     - Error handling
     - Response validation
     - Client caching
     - RESTful API design

9. **Semantic Data Analysis** (`09_semantic_data_analysis.py`)
   - Intelligent data field analysis and categorization
   - Pattern recognition using embeddings
   - Data categories:
     - Numeric, Text, Metadata
     - Mixed types
     - JSON structures
   - Field patterns:
     - Identifiers
     - Personal information
     - Temporal data
     - Financial data
     - Categorical data
     - Measurements
     - Location data
     - Contact information
     - System metadata
     - User preferences
   - Visualization features:
     - Color-coded categories
     - Quality indicators
     - Complexity markers
     - Pattern grouping
     - Detailed statistics

10. **Complex Data Analysis** (`10_semantic_complex_data_analysis.py`)
    - Advanced analysis for enterprise data structures
    - Enhanced pattern recognition
    - Support for:
      - Nested JSON structures
      - Mixed data types
      - Multi-value fields
      - Inconsistent formats
    - Analysis features:
      - Hierarchical data handling
      - Format validation
      - Anomaly detection
      - Quality scoring
    - Enterprise patterns:
      - Department hierarchies
      - Compensation structures
      - Performance metrics
      - System metadata
      - Contact details
      - Temporal sequences

11. **LLM Benchmarking** (`11_llm_benchmark.py`)
    - Comprehensive model comparison framework
    - Multiple evaluation metrics:
      - ROUGE scores (1, 2, L)
      - BLEU score
      - Perplexity
      - BM25 similarity
    - Complaint-specific metrics:
      - Negativity scoring
      - Emotional intensity
      - Structure analysis
      - Pattern density
    - Hardware optimization:
      - Apple Silicon (MPS) support
      - CUDA support
      - CPU fallback
    - Detailed analysis reporting:
      - Comparative metrics
      - Improvement percentages
      - Statistical significance
      - Visual progress tracking
    - Support for:
      - Custom datasets
      - Multiple model comparisons
      - Batch processing
      - Metric visualization

12. **GGUF Model Conversion** (`12_llm_gguf_conversion.py`)
    - Automated GGUF format conversion pipeline
    - Multiple quantization levels:
      - 2-bit to 8-bit options
      - FP16 and FP32 support
      - Size/quality trade-off variants
    - Hardware-specific optimizations:
      - Metal support for Apple Silicon
      - Multi-threaded processing
    - Hugging Face integration:
      - Automatic model download
      - GGUF model upload
      - Repository management
    - Key features:
      - Automated llama.cpp setup
      - Progress monitoring
      - Error handling
      - Resource management
    - Support for:
      - Custom quantization methods
      - Model verification
      - Batch processing
      - Version control

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
