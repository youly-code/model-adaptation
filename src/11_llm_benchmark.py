from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from datetime import datetime
import os
import dotenv
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from typing import Tuple
from textblob import TextBlob
import re
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download("punkt")

# Load environment variables
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class ComplaintModelBenchmark:
    """
    A comprehensive benchmarking system for comparing complaint-generation language models.
    Evaluates performance between a fine-tuned model and its base reference model using
    multiple NLP metrics.
    """

    def __init__(
        self,
        model_name: str = "leonvanbokhorst/Llama-3.2-1B-Instruct-Complaint",
        reference_model: str = "unsloth/Llama-3.2-1B-Instruct",
    ):
        """
        Initialize benchmark environment.

        Hardware Optimization:
        - Automatically selects best available hardware (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
        - Uses float16 precision for memory efficiency

        Models:
        - Loads both fine-tuned complaint model and original base model for comparison
        - Uses device_map="auto" for optimal multi-GPU utilization if available
        """
        # Check if MPS is available (for Apple Silicon)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        # Load models and tokenizers
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=HF_TOKEN,
            torch_dtype=torch.float16,
        ).to(
            self.device
        )  # Explicitly move to device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
        )

        # Load reference model for comparison
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            reference_model,
            device_map="auto",
            token=HF_TOKEN,
            torch_dtype=torch.float16,
        ).to(
            self.device
        )  # Explicitly move to device

        self.ref_tokenizer = AutoTokenizer.from_pretrained(
            reference_model,
            token=HF_TOKEN,
        )

        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.perplexity = evaluate.load("perplexity", module_type="metric")

        # Define complaint-specific patterns
        self.complaint_patterns = {
            'problem_indicators': r'\b(issue|problem|broken|terrible|awful|bad|wrong|horrible|disappointed|unacceptable)\b',
            'demand_indicators': r'\b(want|need|demand|expect|require|should|must|refund|compensation)\b',
            'emotional_intensifiers': r'\b(very|extremely|totally|absolutely|completely|utterly|deeply|seriously)\b',
            'negative_consequences': r'\b(wasted|lost|ruined|damaged|affected|suffered|inconvenient|frustrating)\b'
        }

    def generate_response(
        self,
        prompt: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> str:
        """
        Generate model response with controlled parameters:

        Generation Parameters:
        - max_length=256: Limits context window to manage memory
        - temperature=0.7: Balanced between creativity and coherence
        - top_p=0.9: Nucleus sampling for natural language variation
        - do_sample=True: Enables probabilistic sampling
        """
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        # Explicitly move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Move outputs back to CPU for tokenizer decoding
            outputs = outputs.cpu()

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def calculate_complaint_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate complaint-specific metrics including sentiment, structure, and intensity.
        
        Returns:
            Dict containing:
            - negativity_score: -1 to 1 (more negative is better for complaints)
            - emotional_intensity: 0 to 1 (higher is more intense)
            - structure_score: 0 to 1 (presence of complaint elements)
            - complaint_patterns_score: 0 to 1 (density of complaint-specific language)
        """
        # Sentiment Analysis
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity

        # Convert polarity to negativity score (invert and normalize to 0-1)
        negativity_score = (sentiment_polarity * -1 + 1) / 2

        # Emotional Intensity (combination of subjectivity and pattern matching)
        emotional_words = len(re.findall(self.complaint_patterns['emotional_intensifiers'], text.lower()))
        emotional_intensity = (emotional_words / len(text.split()) + sentiment_subjectivity) / 2

        # Structure Analysis
        structure_elements = {
            'problem': bool(re.search(self.complaint_patterns['problem_indicators'], text.lower())),
            'demand': bool(re.search(self.complaint_patterns['demand_indicators'], text.lower())),
            'consequence': bool(re.search(self.complaint_patterns['negative_consequences'], text.lower()))
        }
        structure_score = sum(structure_elements.values()) / len(structure_elements)

        total_patterns = sum(
            len(re.findall(pattern, text.lower()))
            for pattern in self.complaint_patterns.values()
        )
        words = len(text.split())
        pattern_density = min(1.0, total_patterns / (words * 0.3))  # Normalize to max of 1

        return {
            "negativity_score": negativity_score,
            "emotional_intensity": emotional_intensity,
            "structure_score": structure_score,
            "complaint_patterns_score": pattern_density
        }

    def calculate_metrics(
        self,
        generated_text: str,
        reference_text: str = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics combining quality and complaint-specific metrics.
        """
        metrics = {}
        
        # Existing quality metrics
        if reference_text:
            rouge_scores = self.rouge_scorer.score(reference_text, generated_text)
            metrics.update({
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
            })
            
            generated_tokens = nltk.word_tokenize(generated_text.lower())
            reference_tokens = [nltk.word_tokenize(reference_text.lower())]
            smoothing = SmoothingFunction().method1
            metrics["bleu"] = sentence_bleu(
                reference_tokens,
                generated_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )

        # Text fluency metric
        metrics["perplexity"] = self.perplexity.compute(
            predictions=[generated_text],
            model_id="gpt2",
        )["mean_perplexity"]
        
        # Add complaint-specific metrics
        complaint_metrics = self.calculate_complaint_metrics(generated_text)
        metrics.update(complaint_metrics)

        # Add BM25 (Best Match 25) scoring when reference text is available
        if reference_text:
            """
            BM25 (Best Match 25) Scoring Explanation:
            
            BM25 is a ranking function that:
            1. Estimates relevance of documents based on term frequency (TF) and inverse document frequency (IDF)
            2. Improves on basic TF-IDF by adding length normalization and diminishing returns for term frequency
            
            Formula: BM25 = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (docLength/avgDocLength)))
            
            Where:
            - IDF: Inverse Document Frequency (importance of term in corpus)
            - TF: Term Frequency (frequency of term in document)
            - k1: Term frequency saturation parameter (default=1.5)
            - b: Length normalization parameter (default=0.75)
            
            Advantages for complaint comparison:
            1. Weights distinctive complaint terms more heavily
            2. Prevents over-emphasis on repeated terms
            3. Accounts for varying complaint lengths
            4. Better handles domain-specific vocabulary
            """
            
            # Tokenize both texts into words, converting to lowercase for fair comparison
            tokenized_reference = word_tokenize(reference_text.lower())
            tokenized_generated = word_tokenize(generated_text.lower())
            
            # Create BM25 model using reference text as the corpus
            # BM25Okapi uses the Okapi implementation with default parameters:
            # k1=1.5 (term frequency saturation)
            # b=0.75 (length normalization)
            bm25 = BM25Okapi([tokenized_reference])
            
            # Score generated text against reference
            # Higher scores indicate better matching of important terms
            # Returns array of scores (one per document in corpus)
            bm25_score = bm25.get_scores(tokenized_generated)[0]
            metrics["bm25_score"] = float(bm25_score)  # Convert numpy float to Python float
        
        return metrics

    def run_benchmark(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive benchmark comparing models with detailed progress reporting.
        """
        dataset = load_dataset("leonvanbokhorst/synthetic-complaints-v2")["train"]
        test_samples = dataset.select(range(num_samples))

        results = {
            "fine_tuned_model": [],
            "reference_model": [],
            "metrics": [],
            "timestamps": [],
        }

        # Initialize running averages with all metrics
        running_metrics = {
            "ft": {
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "bleu": [],
                "perplexity": [],
                "negativity_score": [],
                "emotional_intensity": [],
                "structure_score": [],
                "complaint_patterns_score": [],
                "bm25_score": [],
            },
            "ref": {
                "rouge1": [],
                "rouge2": [],
                "rougeL": [],
                "bleu": [],
                "perplexity": [],
                "negativity_score": [],
                "emotional_intensity": [],
                "structure_score": [],
                "complaint_patterns_score": [],
                "bm25_score": [],
            },
        }

        self.print_results(
            "\n=== STARTING BENCHMARK ===",
            "Samples to process: ",
            num_samples,
            "Models being compared:",
        )
        print(f"  - Fine-tuned: {self.model.config._name_or_path}")
        print(f"  - Reference: {self.ref_model.config._name_or_path}")
        print("\n=== PROGRESS ===")

        for i, sample in enumerate(tqdm(test_samples, desc="Benchmarking")):
            prompt = f"Tell me about {sample['topic']}"

            # Print detailed progress every 10 samples
            if i % 10 == 0:
                tqdm.write(f"\n[Batch {i//10 + 1}]")
                tqdm.write(f"Processing topic: {sample['topic']}")

            # Generate responses from both models
            ft_response = self.generate_response(prompt, self.model, self.tokenizer)
            ref_response = self.generate_response(
                prompt, self.ref_model, self.ref_tokenizer
            )

            # Calculate metrics
            ft_metrics = self.calculate_metrics(ft_response, sample["output"])
            ref_metrics = self.calculate_metrics(ref_response, sample["output"])

            # Update running averages
            for metric in ft_metrics:
                if metric in running_metrics["ft"]:
                    running_metrics["ft"][metric].append(ft_metrics[metric])
                    running_metrics["ref"][metric].append(ref_metrics[metric])

            results["fine_tuned_model"].append(ft_response)
            results["reference_model"].append(ref_response)
            results["metrics"].append(
                {
                    "fine_tuned": ft_metrics,
                    "reference": ref_metrics,
                }
            )
            results["timestamps"].append(datetime.now().isoformat())

            # Print detailed metrics every 10 samples
            if i % 10 == 0:
                tqdm.write("\nRunning Averages:")
                for metric in ["rouge1", "bleu", "perplexity"]:
                    ft_avg = np.mean(running_metrics["ft"][metric])
                    ref_avg = np.mean(running_metrics["ref"][metric])
                    improvement = (
                        (ft_avg - ref_avg) / ref_avg * 100
                        if metric != "perplexity"
                        else ((ref_avg - ft_avg) / ref_avg * 100)
                    )

                    tqdm.write(
                        f"{metric.upper():10}: FT={ft_avg:.3f} REF={ref_avg:.3f} "
                        f"(Δ: {improvement:+.1f}%)"
                    )

                # Print sample outputs
                tqdm.write("\nSample Outputs:")
                tqdm.write("Topic: " + sample["topic"])
                tqdm.write("Fine-tuned (first 100 chars): " + ft_response[:100] + "...")
                tqdm.write("Reference (first 100 chars): " + ref_response[:100] + "...")
                tqdm.write("\n" + "=" * 50)

            # Print warning if metrics are concerning
            if ft_metrics["perplexity"] > 100:
                tqdm.write(
                    f"\n⚠️ High perplexity detected in fine-tuned model: {ft_metrics['perplexity']:.2f}"
                )
            if ft_metrics["bleu"] < 0.01:
                tqdm.write(
                    f"\n⚠️ Very low BLEU score in fine-tuned model: {ft_metrics['bleu']:.4f}"
                )

        self.print_results(
            "\n=== BENCHMARK COMPLETE ===",
            "Total samples processed: ",
            num_samples,
            "\nFinal Averages:",
        )
        for metric in ["rouge1", "rouge2", "rougeL", "bleu", "perplexity"]:
            ft_final = np.mean(running_metrics["ft"][metric])
            ref_final = np.mean(running_metrics["ref"][metric])
            improvement = (
                (ft_final - ref_final) / ref_final * 100
                if metric != "perplexity"
                else ((ref_final - ft_final) / ref_final * 100)
            )

            print(
                f"{metric.upper():10}: FT={ft_final:.3f} REF={ref_final:.3f} "
                f"(Δ: {improvement:+.1f}%)"
            )

        return results

    # TODO Rename this here and in `run_benchmark`
    def print_results(self, arg0, arg1, num_samples, arg3):
        print(arg0)
        print(f"{arg1}{num_samples}")
        print(arg3)

    def save_results(
        self, results: Dict[str, Any], output_dir: str = "benchmark_results"
    ):
        """
        Save benchmark results in multiple formats.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        with open(f"{output_dir}/results_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Create summary with metrics explanation:
        # ft_* prefix: metrics for fine-tuned model
        # ref_* prefix: metrics for reference (base) model
        summary_data = [
            {
                "timestamp": results["timestamps"][i],
                # ROUGE-1: Unigram overlap between generated and reference text
                # Range: 0-1, higher is better
                # Measures: Word-level similarity and content coverage
                "ft_rouge1": results["metrics"][i]["fine_tuned"]["rouge1"],
                "ref_rouge1": results["metrics"][i]["reference"]["rouge1"],
                # BLEU: Geometric mean of n-gram precision up to 4-grams
                # Range: 0-1, higher is better
                # Measures: Overall text generation quality and fluency
                "ft_bleu": results["metrics"][i]["fine_tuned"]["bleu"],
                "ref_bleu": results["metrics"][i]["reference"]["bleu"],
                # Perplexity: Exponential of the cross-entropy loss
                # Range: 1 to ∞, lower is better
                # Measures: Model's confidence in predicting the text
                # Good scores typically range from 10-50
                "ft_perplexity": results["metrics"][i]["fine_tuned"]["perplexity"],
                "ref_perplexity": results["metrics"][i]["reference"]["perplexity"],
            }
            for i in range(len(results["metrics"]))
        ]

        # Create DataFrame and add metric descriptions as column comments
        df = pd.DataFrame(summary_data)
        df.columns.name = """
        Metric Explanations:
        - rouge1: Measures word overlap with reference text (0-1, higher better)
        - bleu: Measures overall text quality (0-1, higher better)
        - perplexity: Measures prediction confidence (lower better, typically 10-50 is good)
        
        Prefixes:
        - ft_: Fine-tuned complaint model
        - ref_: Base reference model
        """

        # Save with metric explanations
        df.to_csv(f"{output_dir}/summary_{timestamp}.csv", index=False)

        # Calculate and save aggregate metrics with explanations
        aggregates = {
            "fine_tuned": {
                metric: np.mean([r["fine_tuned"][metric] for r in results["metrics"]])
                for metric in results["metrics"][0]["fine_tuned"].keys()
            },
            "reference": {
                metric: np.mean([r["reference"][metric] for r in results["metrics"]])
                for metric in results["metrics"][0]["reference"].keys()
            },
            "metric_guide": {
                "rouge1": "Word overlap (0-1, higher better)",
                "rouge2": "Two-word phrase overlap (0-1, higher better)",
                "rougeL": "Longest common sequence (0-1, higher better)",
                "bleu": "Overall text quality (0-1, higher better)",
                "perplexity": "Prediction confidence (lower better, 10-50 is good)",
            },
        }

        with open(f"{output_dir}/aggregates_{timestamp}.json", "w") as f:
            json.dump(aggregates, f, indent=2)

        # Add the analysis print at the end
        self.print_benchmark_analysis(aggregates)

    def print_benchmark_analysis(self, results: Dict[str, Any]) -> None:
        """
        Prints analysis including new complaint-specific metrics.
        """
        ft = results["fine_tuned"]
        ref = results["reference"]

        print("\n=== BENCHMARK ANALYSIS SUMMARY ===\n")

        # Calculate improvement percentages
        rouge1_improvement = ((ft["rouge1"] - ref["rouge1"]) / ref["rouge1"]) * 100
        rouge2_improvement = ((ft["rouge2"] - ref["rouge2"]) / ref["rouge2"]) * 100
        rougeL_improvement = ((ft["rougeL"] - ref["rougeL"]) / ref["rougeL"]) * 100
        bleu_improvement = ((ft["bleu"] - ref["bleu"]) / ref["bleu"]) * 100
        perplexity_improvement = (
            (ref["perplexity"] - ft["perplexity"]) / ref["perplexity"]
        ) * 100

        print("Content Overlap Metrics:")
        print("ROUGE-1 (Word matching):")
        print(f"  Fine-tuned: {ft['rouge1']:.3f} | Reference: {ref['rouge1']:.3f}")
        print(f"  Improvement: {rouge1_improvement:+.1f}%")
        print(f"  → {ft['rouge1']:.1%} of words match reference text\n")

        print("ROUGE-2 (Phrase matching):")
        print(f"  Fine-tuned: {ft['rouge2']:.3f} | Reference: {ref['rouge2']:.3f}")
        print(f"  Improvement: {rouge2_improvement:+.1f}%")
        print(f"  → {ft['rouge2']:.1%} of two-word phrases match reference\n")

        print("ROUGE-L (Longest sequence):")
        print(f"  Fine-tuned: {ft['rougeL']:.3f} | Reference: {ref['rougeL']:.3f}")
        print(f"  Improvement: {rougeL_improvement:+.1f}%")
        print(f"  → {ft['rougeL']:.1%} sequential content overlap\n")

        print("Generation Quality Metrics:")
        print("BLEU Score:")
        print(f"  Fine-tuned: {ft['bleu']:.3f} | Reference: {ref['bleu']:.3f}")
        print(f"  Improvement: {bleu_improvement:+.1f}%")
        print(f"  → Overall text quality score\n")

        print("Perplexity (lower is better):")
        print(
            f"  Fine-tuned: {ft['perplexity']:.2f} | Reference: {ref['perplexity']:.2f}"
        )
        print(f"  Improvement: {perplexity_improvement:+.1f}%")
        print(f"  → Model's confidence in predictions\n")

        print("\nComplaint-Specific Metrics:")
        
        print("Negativity Score (higher is better):")
        print(f"  Fine-tuned: {ft['negativity_score']:.3f} | Reference: {ref['negativity_score']:.3f}")
        neg_improvement = ((ft['negativity_score'] - ref['negativity_score']) / ref['negativity_score']) * 100
        print(f"  Improvement: {neg_improvement:+.1f}%")
        
        print("\nEmotional Intensity (higher is better):")
        print(f"  Fine-tuned: {ft['emotional_intensity']:.3f} | Reference: {ref['emotional_intensity']:.3f}")
        emo_improvement = ((ft['emotional_intensity'] - ref['emotional_intensity']) / ref['emotional_intensity']) * 100
        print(f"  Improvement: {emo_improvement:+.1f}%")
        
        print("\nComplaint Structure Score:")
        print(f"  Fine-tuned: {ft['structure_score']:.3f} | Reference: {ref['structure_score']:.3f}")
        struct_improvement = ((ft['structure_score'] - ref['structure_score']) / ref['structure_score']) * 100
        print(f"  Improvement: {struct_improvement:+.1f}%")
        
        print("\nComplaint Pattern Density:")
        print(f"  Fine-tuned: {ft['complaint_patterns_score']:.3f} | Reference: {ref['complaint_patterns_score']:.3f}")
        pattern_improvement = ((ft['complaint_patterns_score'] - ref['complaint_patterns_score']) / ref['complaint_patterns_score']) * 100
        print(f"  Improvement: {pattern_improvement:+.1f}%")

        print("\nLexical Similarity Metrics:")
        print("BM25 Score (Best Match 25):")
        print(f"  Fine-tuned: {ft['bm25_score']:.3f} | Reference: {ref['bm25_score']:.3f}")
        bm25_improvement = ((ft['bm25_score'] - ref['bm25_score']) / ref['bm25_score']) * 100
        print(f"  Improvement: {bm25_improvement:+.1f}%")
        print("  → Measures keyword importance with:")
        print("    • Term frequency saturation (prevents over-emphasis on repetition)")
        print("    • Length normalization (accounts for complaint size)")
        print("    • Inverse document frequency (weights distinctive terms higher)")

        # Overall assessment
        print("=== ANALYSIS ===")
        if ft["perplexity"] < ref["perplexity"]:
            print("✓ Fine-tuned model shows better prediction confidence")
        else:
            print("⚠ Fine-tuned model shows lower prediction confidence")

        if ft["bleu"] > ref["bleu"]:
            print("✓ Fine-tuned model generates higher quality text")
        else:
            print("⚠ Fine-tuned model generates lower quality text")

        if ft["rouge1"] > ref["rouge1"]:
            print("✓ Fine-tuned model shows better content matching")
        else:
            print("⚠ Fine-tuned model shows worse content matching")

        print("\n=== CONCLUSION ===")
        avg_improvement = (
            rouge1_improvement
            + rouge2_improvement
            + rougeL_improvement
            + bleu_improvement
            + perplexity_improvement
        ) / 5

        if avg_improvement > 0:
            print(f"Overall improvement: +{avg_improvement:.1f}%")
            print("The fine-tuning has successfully improved the model's performance.")
        else:
            print(f"Overall change: {avg_improvement:.1f}%")
            print("The fine-tuning has not improved the model's overall performance.")


if __name__ == "__main__":
    # Run benchmark
    benchmark = ComplaintModelBenchmark()
    results = benchmark.run_benchmark(num_samples=100)
    benchmark.save_results(results)
