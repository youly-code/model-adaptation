from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import logging
from pathlib import Path
from typing import Tuple, Set, List, Dict
import time
import random
from fake_useragent import UserAgent
import imagehash
import ollama
from nomic import atlas
import numpy as np
from collections import defaultdict
import argparse
import sys
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch import optim
import re
import uuid
from torchvision.utils import save_image
from tqdm import tqdm


class CustomImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class DynamicImageScraper:

    def __init__(
        self,
        target_concept: str,
        output_dir: str = None,
        min_resolution: Tuple[int, int] = (512, 512),
        max_images: int = 100,
        confidence_threshold: float = 0.97,
        min_delay: float = 8.0,
        max_delay: float = 15.0,
        interactive: bool = True,  # Add interactive mode
    ):
        """Initialize the scraper with target concept and parameters."""
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Initialize counters and tracking variables
        self.current_index = 0
        self.downloaded_images = set()
        self.processed_images = set()

        # Initialize adaptive threshold
        self.adaptive_threshold = confidence_threshold
        self.confidence_scores = []
        self.mean_target_confidence = 0.0

        # Define supported image extensions first
        self.image_extensions = ("[jJ][pP][gG]", "[jJ][pP][eE][gG]", "[pP][nN][gG]")

        # Initialize device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logging.info("Using Apple Silicon MPS device")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            logging.info("Using CPU device")

        # Continue with other initializations
        self.target_concept = target_concept

        # Sanitize target concept for folder naming
        self.folder_name = self._sanitize_folder_name(target_concept)

        # Set default output directory if none provided
        if output_dir is None:
            output_dir = f"dataset/{self.folder_name}"

        self.output_dir = Path(output_dir).absolute()
        self.min_resolution = min_resolution
        self.max_images = max_images
        self.confidence_threshold = confidence_threshold
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.interactive = interactive

        # Initialize categories dictionary
        self.target_categories = {}

        # Setup directory structure
        self.setup_output_directory()

        # Setup model and transform first
        self.setup_model()

        # Now bootstrap target folder (needs model and transform)
        self.bootstrap_target_folder()

        # Analyze target images and update categories
        self.target_categories = self._analyze_target_images()
        self.mean_target_confidence = self.target_categories.get("cassette", 0.0)

        # Add fine-tuning parameters
        self.num_epochs = 10
        self.batch_size = 4
        self.learning_rate = 1e-4

        # Fine-tune if we have target images
        if self._get_image_files(self.target_images_dir):
            self.fine_tune_model()

        # Continue with existing initialization...

        # Add concept-specific search modifiers
        self.search_modifiers = [
            "high quality photo",
            "professional photograph",
            "clear image",
            "product shot",
            "detailed view",
        ]

        # Add generic negative terms
        self.negative_terms = [
            "cartoon",
            "drawing",
            "illustration",
            "sketch",
            "artwork",
            "meme",
            "screenshot",
            "diagram",
        ]

        # Add hash tracking for duplicate detection
        self.image_hashes = set()

        # Increase images per search term
        self.images_per_term = 20  # Increased from default of 2-5

        # Add work directory for temporary downloads
        self.work_dir = self.output_dir / "work"
        self.work_dir.mkdir(exist_ok=True)

        # Add batch tracking
        self.current_batch = self._get_latest_batch() + 1

    def _sanitize_folder_name(self, name: str) -> str:
        """Convert target concept to valid folder name."""
        # Convert to lowercase and replace spaces/special chars with underscores
        sanitized = re.sub(r"[^\w\s-]", "", name.lower())
        sanitized = re.sub(r"[-\s]+", "_", sanitized)
        return sanitized.strip("_")

    def setup_output_directory(self) -> None:
        """Setup the output directory structure."""
        # Create main directory using sanitized name
        self.output_dir = Path(self.output_dir).absolute()
        logging.info(f"\nSetting up directory structure at: {self.output_dir}")

        # Create images subdirectory
        self.images_dir = self.output_dir / "images"
        self.target_images_dir = self.images_dir / "target"

        # Create all directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.target_images_dir.mkdir(exist_ok=True)

        logging.info(f"\nVerifying directory structure:")
        logging.info(f"- Root exists: {self.output_dir.exists()}")
        logging.info(f"  Path: {self.output_dir}")
        logging.info(f"- Images exists: {self.images_dir.exists()}")
        logging.info(f"  Path: {self.images_dir}")
        logging.info(f"- Target exists: {self.target_images_dir.exists()}")
        logging.info(f"  Path: {self.target_images_dir}")

    def _analyze_target_images(self) -> Dict[str, float]:
        """Analyze target images to determine target categories."""
        logging.info(f"\nAnalyzing target images in: {self.target_images_dir}")

        target_images = self._get_image_files(self.target_images_dir)
        if not target_images:
            raise ValueError(f"No images found in {self.target_images_dir}")

        # Initialize category tracking
        categories = defaultdict(float)
        total_images = len(target_images)

        for img_path in target_images:
            logging.info(f"\nAnalyzing: {img_path.name}")
            try:
                with Image.open(img_path) as img:
                    scaled_img = self._scale_image_for_models(img)
                    img_tensor = (
                        self.transform(scaled_img.convert("RGB"))
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    with torch.no_grad():
                        output = self.model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)

                        # Get top 3 predictions
                        top_probs, top_indices = torch.topk(probabilities, 3)
                        for prob, idx in zip(top_probs, top_indices):
                            category = self.categories[idx.item()]
                            prob_value = prob.item()
                            categories[category] += prob_value
                            logging.info(f"- {category:30} {prob_value:.2%}")

            except Exception as e:
                logging.error(f"Error analyzing {img_path}: {e}")
                logging.error("Stack trace:", exc_info=True)
                continue

        # Average the probabilities
        for category in categories:
            categories[category] /= total_images

        # Log aggregated results
        logging.info("\nAggregated category profile:")
        for category, prob in sorted(
            categories.items(), key=lambda x: x[1], reverse=True
        ):
            logging.info(f"{category:30} {prob:.2%}")

        # Add generic category grouping
        category_groups = {
            "object": ["object", "device", "item", "product"],
            "material": ["plastic", "metal", "wooden", "glass"],
            "color": ["black", "white", "colored", "silver"],
            "condition": ["new", "vintage", "used", "antique"],
        }

        grouped_categories = defaultdict(float)

        for category, score in categories.items():
            # Map to generic groups
            for group, terms in category_groups.items():
                if any(term in category.lower() for term in terms):
                    grouped_categories[group] += score

            # Keep original category as well
            grouped_categories[category] += score

        return dict(grouped_categories)

    def _build_category_filters(self) -> Tuple[Set[str], Set[str]]:
        """Build category filters based on target image analysis"""
        valid = set()

        # Add high-confidence categories (>20%)
        primary_categories = {
            cat for cat, score in self.target_categories.items() if score > 0.20
        }
        logging.info(f"\nPrimary categories (>20%): {primary_categories}")

        # Add medium-confidence categories (>10%)
        secondary_categories = {
            cat for cat, score in self.target_categories.items() if 0.10 < score <= 0.20
        }
        logging.info(f"Secondary categories (>10%): {secondary_categories}")

        # Build valid category set
        valid.update(primary_categories)
        valid.update(self.target_concept.lower().split())

        # Extract individual words from categories
        for category in primary_categories | secondary_categories:
            valid.update(category.split("_"))

        # Remove common stop words and generic terms
        stop_words = {"a", "an", "the", "in", "on", "at", "for", "to", "of", "with"}
        valid = {word for word in valid if word not in stop_words}

        # Build invalid categories
        invalid = set()

        # Add low-confidence noise categories (<1%)
        invalid.update(
            {
                cat
                for cat, score in self.target_categories.items()
                if score < 0.01 and cat not in valid
            }
        )

        # Add categories that might indicate wrong content
        invalid.update(
            {
                # Technical issues
                "blurry",
                "noise",
                "artifact",
                "distorted",
                # Wrong context
                "digital",
                "modern",
                "computer",
                "phone",
                # Generic/irrelevant
                "background",
                "texture",
                "pattern",
                "design",
                "abstract",
                "random",
                "misc",
                "other",
            }
        )

        logging.info(f"\nCategory Filters:")
        logging.info(f"Valid categories: {valid}")
        logging.info(f"Invalid categories: {invalid}")

        return valid, invalid

    def validate_image(self, image_path: Path) -> bool:
        """Validate image against learned category profile"""
        try:
            with Image.open(image_path) as img:
                scaled_img = self._scale_image_for_models(img)
                img_tensor = (
                    self.transform(scaled_img.convert("RGB"))
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    output = self.model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    top10_prob, top10_catid = torch.topk(probabilities, 10)

                    # Compare against target profile
                    similarity_score = 0.0
                    for prob, cat_id in zip(top10_prob, top10_catid):
                        category = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][
                            cat_id
                        ].lower()
                        target_score = self.target_categories.get(category, 0.0)
                        similarity_score += min(prob.item(), target_score)

                    return similarity_score >= self.confidence_threshold

        except Exception as e:
            logging.error(f"Validation error: {e}")
            return False

    def setup_model(self) -> None:
        """Initialize the image classification model."""
        try:
            # Load pre-trained ResNet model
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load ImageNet categories
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.categories = weights.meta["categories"]

            # Setup image transform
            self.transform = weights.transforms()

            logging.info(f"\nModel initialized on device: {self.device}")
            if self.device.type == "mps":
                logging.info("Running on Apple Silicon GPU")
            elif self.device.type == "cuda":
                logging.info("Running on NVIDIA GPU")
            else:
                logging.info("Running on CPU")

        except Exception as e:
            logging.error(f"Error setting up model: {e}")
            raise

    def setup_llm(self) -> None:
        """Initialize LLM connection"""
        try:
            # Test connection
            ollama.pull(self.llm_model)
        except Exception as e:
            logging.error(f"Failed to initialize LLM: {e}")
            raise

    def get_target_classes(self) -> Set[str]:
        """Use LLM to identify relevant ImageNet classes"""
        prompt = f"""
        Given the concept '{self.target_concept}', list 10 ImageNet classes that would be 
        most relevant for identifying such objects in images. Return only the class names,
        one per line, lowercase, no numbers or special characters.
        """

        try:
            response = ollama.generate(model=self.llm_model, prompt=prompt)
            classes = {
                cls.strip().lower()
                for cls in response["response"].split("\n")
                if cls.strip()
            }
            logging.info(f"Generated target classes: {classes}")
            return classes
        except Exception as e:
            logging.error(f"Failed to generate target classes: {e}")
            return {"object", "device"}  # Fallback classes

    def analyze_seed_images(self) -> dict:
        """Analyze existing images in target folder to establish baseline characteristics"""
        seed_data = {"successful_categories": {}, "avg_confidence": 0.0, "count": 0}

        seed_path = self.output_dir / "target"
        if not seed_path.exists():
            return seed_data

        for img_path in seed_path.glob("*.[jJ][pP][gG]"):
            try:
                with Image.open(img_path) as img:
                    scaled_img = self._scale_image_for_models(img)
                    img_tensor = (
                        self.transform(scaled_img.convert("RGB"))
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    with torch.no_grad():
                        output = self.model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        top5_prob, top5_catid = torch.topk(probabilities, 5)

                        # Track confidence scores and categories
                        for prob, cat_id in zip(top5_prob, top5_catid):
                            category = ResNet50_Weights.IMAGENET1K_V2.meta[
                                "categories"
                            ][cat_id]
                            if prob.item() > 0.1:  # Only significant categories
                                seed_data["successful_categories"][category] = (
                                    seed_data["successful_categories"].get(category, 0)
                                    + 1
                                )

                        seed_data["avg_confidence"] += top5_prob[0].item()
                        seed_data["count"] += 1

            except Exception as e:
                logging.error(f"Error analyzing seed image {img_path}: {e}")

        if seed_data["count"] > 0:
            seed_data["avg_confidence"] /= seed_data["count"]

        logging.info(f"Seed image analysis: {seed_data}")
        return seed_data

    def _get_category_guidance(self) -> str:
        """Get category-specific search guidance using seed image analysis"""
        # Analyze seed images first
        seed_data = self.analyze_seed_images()

        # Include seed analysis in the meta prompt
        meta_prompt = f"""
        You are a computer vision expert. For the target concept "{self.target_concept}",
        provide specific guidance for image search terms.
        
        Seed image analysis:
        Top categories found in verified good examples:
        {', '.join(f'{k}({v})' for k, v in sorted(seed_data['successful_categories'].items(), 
                   key=lambda x: x[1], reverse=True)[:5])}
        Average confidence: {seed_data['avg_confidence']:.2%}
        
        Using this analysis and the target concept, provide:
        1. Three primary object categories that define this concept
        2. Three categories to explicitly avoid
        3. Three specific visual characteristics to focus on
        
        Format your response as a simple list with three sections marked by headers:
        PRIMARY:
        AVOID:
        FOCUS:
        """

        try:
            response = ollama.generate(model="hermes3:latest", prompt=meta_prompt)
            return response["response"]
        except Exception as e:
            logging.error(f"Failed to get category guidance: {e}")
            return self._get_default_guidance()

    def generate_search_terms(self) -> List[str]:
        """Generate search terms based on target concept and modifiers."""
        base_terms = [
            self.target_concept,
            f"{self.target_concept} isolated",
            f"{self.target_concept} close up",
        ]

        # Add modified terms
        modified_terms = [
            f"{self.target_concept} {modifier}" for modifier in self.search_modifiers
        ]

        # Add negative terms to exclude unwanted results
        negative_suffix = " ".join(f"-{term}" for term in self.negative_terms)

        search_terms = []
        for term in base_terms + modified_terms:
            search_terms.append(f"{term} {negative_suffix}")

        logging.info("\nGenerated search terms:")
        for term in search_terms:
            logging.info(f"- {term}")

        return search_terms

    def _get_top_categories(self) -> str:
        """Get feedback about which ResNet categories are performing well"""
        if not hasattr(self, "_successful_categories"):
            self._successful_categories = {}

        categories_str = "\n".join(
            [
                f"- {category}: {count} successful matches"
                for category, count in self._successful_categories.items()
                if count > 0
            ]
        )

        return categories_str if categories_str else "No successful matches yet"

    def is_target_object(self, image_path: Path) -> bool:
        """Check if image contains target object using only ResNet"""
        try:
            resnet_result, confidence = self.is_target_resnet(image_path)

            if resnet_result:
                # Track successful categories
                with Image.open(image_path) as img:
                    scaled_img = self._scale_image_for_models(img)
                    img_tensor = (
                        self.transform(scaled_img.convert("RGB"))
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    with torch.no_grad():
                        output = self.model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output[0], dim=0)
                        top5_prob, top5_catid = torch.topk(probabilities, 5)

                        for prob, cat_id in zip(top5_prob, top5_catid):
                            if prob.item() > 0.1:  # Only track significant categories
                                category = ResNet50_Weights.IMAGENET1K_V2.meta[
                                    "categories"
                                ][cat_id]
                                self._successful_categories[category] = (
                                    self._successful_categories.get(category, 0) + 1
                                )

            return resnet_result

        except Exception as e:
            logging.error(f"Error in target object detection: {e}")
            return False

    def _scale_image_for_models(
        self, image: Image.Image, max_size: int = 512
    ) -> Image.Image:
        """Scale image while maintaining aspect ratio"""
        ratio = max_size / max(image.size)
        if ratio < 1:  # Only scale down, never up
            new_size = tuple(int(dim * ratio) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def verify_with_llm(self, image_path: Path) -> bool:
        """Use LLM to verify image content"""
        try:
            # Load and scale image
            with Image.open(image_path) as img:
                scaled_img = self._scale_image_for_models(img)
                # Save scaled version to temp file
                temp_path = image_path.parent / f"temp_{image_path.name}"
                scaled_img.save(temp_path, "JPEG", quality=85)

            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"""Analyze this image. Does it show a {self.target_concept}?
                    Answer with ONLY 'true' or 'false'.
                    Be inclusive - answer 'true' if the image shows anything clearly related to {self.target_concept}.
                    Consider different variations, styles, and contexts.
                    """,
                        "images": [str(temp_path)],
                    }
                ]

                response = ollama.chat(model=self.llm_model, messages=messages)

                result = response["message"]["content"].strip().lower() == "true"
                logging.info(f"LLM verification for {image_path}: {result}")
                return result

            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            logging.error(f"LLM verification failed: {e}")
            return True  # Default to accepting if LLM fails

    def is_duplicate(self, image_path: Path) -> bool:
        """Check if image is too similar to existing ones using multiple hash types."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB and resize for consistent comparison
                img = img.convert("RGB")

                # Calculate multiple types of perceptual hashes
                avg_hash = str(imagehash.average_hash(img, hash_size=16))
                dhash = str(imagehash.dhash(img, hash_size=16))
                phash = str(imagehash.phash(img, hash_size=16))
                whash = str(imagehash.whash(img, hash_size=16))

                # Calculate color histogram
                histogram = img.histogram()

                # Check against existing files on disk
                existing_images = [
                    p for p in self._get_image_files(self.images_dir) if p != image_path
                ]

                for existing_path in existing_images:
                    try:
                        with Image.open(existing_path) as existing_img:
                            existing_img = existing_img.convert("RGB")

                            # Calculate hashes for existing image
                            existing_avg = str(
                                imagehash.average_hash(existing_img, hash_size=16)
                            )
                            existing_dhash = str(
                                imagehash.dhash(existing_img, hash_size=16)
                            )
                            existing_phash = str(
                                imagehash.phash(existing_img, hash_size=16)
                            )
                            existing_whash = str(
                                imagehash.whash(existing_img, hash_size=16)
                            )

                            # Calculate histogram for existing image
                            existing_histogram = existing_img.histogram()

                            # Calculate hash differences
                            avg_diff = imagehash.hex_to_hash(
                                avg_hash
                            ) - imagehash.hex_to_hash(existing_avg)
                            dhash_diff = imagehash.hex_to_hash(
                                dhash
                            ) - imagehash.hex_to_hash(existing_dhash)
                            phash_diff = imagehash.hex_to_hash(
                                phash
                            ) - imagehash.hex_to_hash(existing_phash)
                            whash_diff = imagehash.hex_to_hash(
                                whash
                            ) - imagehash.hex_to_hash(existing_whash)

                            # Calculate histogram similarity
                            histogram_diff = sum(
                                abs(h1 - h2)
                                for h1, h2 in zip(histogram, existing_histogram)
                            )
                            histogram_similarity = 1 - (
                                histogram_diff
                                / (max(sum(histogram), sum(existing_histogram)) * 2)
                            )

                            # Define thresholds
                            HASH_THRESHOLD = 8  # Lower means more strict
                            HISTOGRAM_THRESHOLD = 0.95  # Higher means more strict

                            # Check if images are too similar based on multiple criteria
                            hash_match = (
                                avg_diff < HASH_THRESHOLD
                                and dhash_diff < HASH_THRESHOLD
                                and phash_diff < HASH_THRESHOLD
                                and whash_diff < HASH_THRESHOLD
                            )

                            histogram_match = histogram_similarity > HISTOGRAM_THRESHOLD

                            if hash_match and histogram_match:
                                logging.info(
                                    f"Found duplicate: {image_path} matches {existing_path}"
                                )
                                logging.info(
                                    f"Hash differences: avg={avg_diff}, dhash={dhash_diff}, "
                                    f"phash={phash_diff}, whash={whash_diff}"
                                )
                                logging.info(
                                    f"Histogram similarity: {histogram_similarity:.2%}"
                                )
                                return True

                    except Exception as e:
                        logging.error(
                            f"Error comparing with existing image {existing_path}: {e}"
                        )
                        continue

                return False

        except Exception as e:
            logging.error(f"Error checking for duplicates: {e}")
            return False

    def validate_image(self, image_path: Path) -> bool:
        """Validate image with more generic criteria."""
        try:
            with Image.open(image_path) as img:
                # Basic image quality checks
                if (
                    img.size[0] < self.min_resolution[0]
                    or img.size[1] < self.min_resolution[1]
                ):
                    logging.info(f"Image {image_path} below minimum resolution")
                    return False

                # Check image mode and convert if needed
                if img.mode not in ("RGB", "RGBA"):
                    logging.info(
                        f"Converting image {image_path} from {img.mode} to RGB"
                    )
                    img = img.convert("RGB")

                # Check for solid color or mostly empty images
                img_array = np.array(img)
                if np.std(img_array) < 20:  # Low variance indicates mostly solid color
                    logging.info(f"Image {image_path} appears to be mostly solid color")
                    return False

                # Continue with existing validation...
                return super().validate_image(image_path)

        except Exception as e:
            logging.error(f"Error validating image {image_path}: {e}")
            return False

    def setup_logging(self) -> None:
        """Configure logging for the scraper."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def cleanup_invalid_images(self) -> None:
        """Remove any invalid or low-confidence images."""
        for img_path in self._get_image_files(self.output_dir):
            if img_path.parent.name == "target":
                continue

            try:
                result, confidence = self.is_target_resnet(img_path)
                if not result:
                    img_path.unlink()
                    logging.info(
                        f"Removed invalid image {img_path} (confidence: {confidence:.2%})"
                    )
            except Exception as e:
                logging.error(f"Error validating {img_path}: {e}")
                img_path.unlink()

    def get_next_filename(self) -> str:
        """Generate sequential filename for downloaded images."""
        while True:
            filename = f"{self.current_index:06d}.jpg"
            filepath = self.images_dir / filename
            if not filepath.exists():
                self.current_index += 1
                return filename

    def _get_latest_batch(self) -> int:
        """Find the highest existing batch number from image filenames."""
        pattern = re.compile(r"batch(\d+)-\d+\.jpg")
        max_batch = 0

        for img_path in self._get_image_files(self.images_dir):
            if match := pattern.match(img_path.name):
                batch_num = int(match.group(1))
                max_batch = max(max_batch, batch_num)

        return max_batch

    def scrape(self) -> None:
        """Modified scrape method to use work directory."""
        try:
            ua = UserAgent()
            logging.info("\nInitialized User Agent rotation")
        except Exception as e:
            logging.error(f"Failed to initialize UserAgent: {e}")
            ua = None

        while self.current_index < self.max_images:
            # Clear work directory
            for file in self.work_dir.glob("*"):
                file.unlink()

            # Generate search terms
            self.search_terms = self.generate_search_terms()

            for term in self.search_terms:
                if self.current_index >= self.max_images:
                    break

                clean_term = term.split(". ", 1)[-1] if ". " in term else term

                try:
                    google_crawler = GoogleImageCrawler(
                        feeder_threads=2,
                        parser_threads=2,
                        downloader_threads=2,
                        storage={"root_dir": str(self.work_dir)},  # Use work directory
                    )

                    # Set user agent in the session
                    if ua:
                        random_ua = ua.random
                        google_crawler.session.headers.update({"User-Agent": random_ua})
                        logging.info(f"\nUsing User-Agent: {random_ua[:50]}...")

                    # Override the file namer
                    google_crawler.file_namer = lambda _: self.get_next_filename()

                    # Remove 'safe' filter and keep only supported filters
                    filters = {
                        "size": "large",
                        "type": "photo",
                        "license": "commercial,modify",
                    }

                    # Add random delay between requests
                    delay = random.uniform(self.min_delay, self.max_delay)
                    logging.info(f"Waiting {delay:.1f}s before next request...")
                    time.sleep(delay)

                    google_crawler.crawl(
                        keyword=clean_term,
                        max_num=self.images_per_term,
                        min_size=self.min_resolution,
                        filters=filters,
                        file_idx_offset="auto",
                        offset=self.current_index,
                    )

                    # Process downloaded images after each term
                    self.process_batch()

                except Exception as e:
                    logging.error(f"Error crawling for term '{clean_term}': {e}")
                    continue

    def process_batch(self) -> None:
        """Process images in work directory."""
        logging.info(f"\nProcessing batch {self.current_batch}")

        # First pass: validate all images and track results
        valid_images = {}  # path -> confidence score

        for img_path in self._get_image_files(self.work_dir):
            try:
                # Validate image
                is_valid, confidence = self.is_target_resnet(img_path)

                if not is_valid:
                    logging.info(
                        f"Invalid image {img_path} (confidence: {confidence:.2%})"
                    )
                    continue

                # Check for duplicates against main directory
                if self.is_duplicate(img_path):
                    logging.info(f"Duplicate image {img_path}")
                    continue

                valid_images[img_path] = confidence

            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                continue

        # Second pass: move valid images and cleanup
        valid_count = 0
        for img_path, confidence in valid_images.items():
            try:
                # Generate new filename
                new_name = f"batch{self.current_batch}-{valid_count:03d}.jpg"
                new_path = self.images_dir / new_name

                # Move file to main directory
                shutil.move(img_path, new_path)
                logging.info(
                    f"Moved valid image to {new_path} (confidence: {confidence:.2%})"
                )

                valid_count += 1
                self.current_index += 1

            except Exception as e:
                logging.error(f"Error moving {img_path}: {e}")
                continue

        # Cleanup: delete all remaining files in work directory
        for img_path in self._get_image_files(self.work_dir):
            img_path.unlink()

        logging.info(f"Batch {self.current_batch} complete: {valid_count} valid images")
        self.current_batch += 1

    def bootstrap_target_folder(self) -> None:
        """Initialize target folder with first-pass search results"""
        logging.info(f"\nChecking target folder: {self.target_images_dir}")

        if not self._get_image_files(self.target_images_dir):
            logging.info("No seed images found. Bootstrapping target folder...")

            # Temporary strict settings for initial search
            original_threshold = self.confidence_threshold
            self.confidence_threshold = 0.4  # Stricter for seed images

            bootstrap_terms = [
                f"{self.target_concept} isolated product shot",
                f"professional photo {self.target_concept}",
                f"high quality {self.target_concept} white background",
            ]

            try:
                for term in bootstrap_terms:
                    google_crawler = GoogleImageCrawler(
                        feeder_threads=1,
                        parser_threads=1,
                        downloader_threads=1,
                        storage={
                            "root_dir": str(self.target_images_dir)
                        },  # Use absolute path
                    )

                    filters = {
                        "size": "large",
                        "type": "photo",
                        "license": "commercial,modify",
                    }

                    # Try to get 2 images per term
                    google_crawler.crawl(
                        keyword=term,
                        max_num=2,
                        min_size=self.min_resolution,
                        filters=filters,
                        file_idx_offset="auto",
                    )

                    # Validate downloaded images strictly
                    for img_path in self._get_image_files(self.target_images_dir):
                        try:
                            result, confidence = self.is_target_resnet(img_path)
                            if not result or confidence < 0.4:  # Strict validation
                                img_path.unlink()
                                logging.info(
                                    f"Removed bootstrap image {img_path} (confidence: {confidence:.2%})"
                                )
                        except Exception as e:
                            logging.error(
                                f"Error validating bootstrap image {img_path}: {e}"
                            )
                            img_path.unlink()

                    time.sleep(random.uniform(self.min_delay, self.max_delay))

                # Keep only the top 5 highest confidence images
                scored_images = []
                for img_path in self._get_image_files(self.target_images_dir):
                    try:
                        _, confidence = self.is_target_resnet(img_path)
                        scored_images.append((img_path, confidence))
                    except Exception:
                        continue

                # Sort by confidence and keep top 5
                scored_images.sort(key=lambda x: x[1], reverse=True)
                for img_path, _ in scored_images[5:]:
                    img_path.unlink()

                if not self._get_image_files(self.target_images_dir):
                    logging.warning(
                        "Failed to bootstrap target folder with high-confidence images"
                    )
                else:
                    logging.info(
                        f"Successfully bootstrapped target folder with {len(self._get_image_files(self.target_images_dir))} images"
                    )

            finally:
                # Restore original threshold
                self.confidence_threshold = original_threshold

    def _interactive_bootstrap(self):
        """Interactive setup for target images"""
        print("\n=== Target Image Setup ===")
        print(
            f"We need 3-5 high-quality images of '{self.target_concept}' to get started."
        )
        print("\nOptions:")
        print("1. Download from Google Images")
        print("2. Provide local image files")
        print("3. Exit and add images manually")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            self._download_bootstrap_images()
        elif choice == "2":
            self._copy_local_images()
        else:
            print("\nExiting. Please add images manually to:", self.target_images_dir)
            sys.exit(0)

    def _download_bootstrap_images(self):
        """Download initial target images with rotating user agents."""
        print("\nDownloading sample images...")

        try:
            ua = UserAgent()
            logging.info("Initialized User Agent rotation for bootstrap")
        except Exception as e:
            logging.error(f"Failed to initialize UserAgent for bootstrap: {e}")
            ua = None

    def _copy_local_images(self):
        """Copy local images to target directory"""
        print("\nEnter paths to image files (one per line, empty line to finish):")
        while True:
            path = input("> ").strip()
            if not path:
                break

            try:
                src_path = Path(path)
                if src_path.exists() and src_path.suffix.lower() in [".jpg", ".jpeg"]:
                    dst_path = self.target_images_dir / src_path.name
                    shutil.copy2(src_path, dst_path)
                    print(f"Copied: {src_path.name}")
                else:
                    print("Invalid file or not a JPG image:", path)
            except Exception as e:
                print(f"Error copying file: {e}")

        self._review_images()

    def _review_images(self):
        """Review and validate target images"""
        print("\nReviewing images...")

        for img_path in self.target_images_dir.glob("*.[jJ][pP][gG]"):
            try:
                with Image.open(img_path) as img:
                    print(f"\nImage: {img_path.name}")
                    print(f"Size: {img.size}")

                    keep = input("Keep this image? (y/n): ").lower().strip() == "y"
                    if not keep:
                        img_path.unlink()
                        print("Deleted.")

            except Exception as e:
                print(f"Error with {img_path}: {e}")
                img_path.unlink()

        print("\nTarget image setup complete!")

    def fine_tune_model(self):
        """Fine-tune ResNet with optimized training."""
        logging.info("\n" + "=" * 50)
        logging.info("Starting Optimized Model Fine-Tuning")
        logging.info("=" * 50)

        try:
            # Training parameters
            self.num_epochs = 20
            self.batch_size = 8
            self.gradient_accumulation_steps = 4
            self.learning_rate = 7e-4
            self.patience = 4
            self.min_delta = 7e-4

            logging.info("\nTraining Configuration:")
            logging.info(f"- Max epochs: {self.num_epochs}")
            logging.info(f"- Batch size: {self.batch_size}")
            logging.info(f"- Learning rate: {self.learning_rate}")
            logging.info(f"- Early stopping patience: {self.patience}")
            logging.info(f"- Min delta: {self.min_delta}")

            # Model setup
            self.model = self.model.to("cpu")
            num_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_features, 2)
            self.model = self.model.to(self.device)

            # Create dataset with positive and negative samples
            transform_train = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            # Create balanced dataset
            target_images = list(self.target_images_dir.glob("*.[jJ][pP][gG]"))
            if not target_images:
                raise ValueError("No training images found!")

            # Create positive and negative samples
            samples = []
            labels = []

            # Add positive samples
            for img_path in target_images:
                samples.append(img_path)
                labels.append(1)  # Positive class

                # Add augmented positive samples to balance dataset
                samples.append(img_path)
                labels.append(1)

            # Create dataset
            dataset = [(path, label) for path, label in zip(samples, labels)]
            logging.info(f"\nDataset size: {len(dataset)} samples")
            logging.info(f"Positive samples: {sum(labels)}")

            # Create data loader
            train_dataset = CustomImageDataset(
                samples=dataset, transform=transform_train
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for debugging
                pin_memory=True if self.device.type != "cpu" else False,
            )

            if len(train_loader) == 0:
                raise ValueError("Empty data loader!")

            # Training setup
            criterion = torch.nn.CrossEntropyLoss().to(self.device)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                nesterov=True,
            )

            # Training loop
            best_loss = float("inf")
            patience_counter = 0
            best_model_state = None

            self.model.train()
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                batch_count = 0

                for inputs, labels in train_loader:
                    try:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad(set_to_none=True)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        batch_count += 1

                    except RuntimeError as e:
                        logging.error(f"Runtime error in training loop: {e}")
                        if "MPS" in str(e):
                            logging.warning("MPS error - falling back to CPU")
                            self.device = torch.device("cpu")
                            self.model = self.model.to(self.device)
                            continue
                        raise e

                if batch_count == 0:
                    raise ValueError("No batches processed in epoch!")

                avg_loss = epoch_loss / batch_count
                logging.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - Loss: {avg_loss:.4f}"
                )

                if avg_loss < best_loss - self.min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    logging.info("→ New best model saved!")
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logging.info(f"\nEarly stopping at epoch {epoch+1}")
                        break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                logging.info("Restored best model state")

            logging.info("\nTraining completed!")

        except Exception as e:
            logging.error(f"Error during fine-tuning: {e}")
            logging.error("Stack trace:", exc_info=True)
            raise

    def is_target_resnet(self, image_path: Path) -> Tuple[bool, float]:
        """Check if image contains target object using fine-tuned ResNet."""
        try:
            with Image.open(image_path) as img:
                img_tensor = (
                    self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
                )

                with torch.no_grad():
                    output = self.model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)

                    # Get probability for target class (index 1)
                    target_prob = probabilities[1].item()
                    non_target_prob = probabilities[0].item()

                    logging.info(f"\n=== Image Validation: {image_path.name} ===")
                    logging.info(f"Probabilities:")
                    logging.info(f"- Non-target: {non_target_prob:.2%}")
                    logging.info(f"- Target: {target_prob:.2%}")
                    logging.info(f"Threshold: {self.confidence_threshold:.2%}")

                    is_valid = target_prob >= self.confidence_threshold
                    logging.info(f"Decision: {'ACCEPTED' if is_valid else 'REJECTED'}")

                    return is_valid, target_prob

        except Exception as e:
            logging.error(f"Error in target detection: {e}")
            return False, 0.0

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all supported image files in directory."""
        image_files = []
        directory = Path(directory)  # Ensure it's a Path object

        logging.info(f"\nScanning directory: {directory.absolute()}")
        logging.info(f"Directory exists: {directory.exists()}")
        logging.info(f"Is directory: {directory.is_dir()}")

        if directory.exists() and directory.is_dir():
            logging.info("Directory contents:")
            for item in directory.iterdir():
                logging.info(f"- {item.name}")

        for ext in self.image_extensions:
            pattern = f"*.{ext}"
            found = list(directory.glob(pattern))
            logging.info(f"- Found {len(found)} files matching pattern: {pattern}")
            if found:
                logging.info("  Files:")
                for f in found:
                    logging.info(f"  - {f.name}")
            image_files.extend(found)

        return image_files

    def _detect_concept_type(self) -> str:
        """Detect the general type of the target concept."""
        try:
            prompt = f"What category best describes '{self.target_concept}'? Options: electronics, furniture, clothing, art, vehicle, other"
            response = ollama.generate(model="hermes3:latest", prompt=prompt)
            return response["response"].strip().lower()
        except Exception as e:
            logging.error(f"Error detecting concept type: {e}")
            return "other"

    def process_images(self) -> None:
        """Process downloaded images through validation and deduplication."""
        logging.info("Processing downloaded images...")

        # Get list of new images only
        new_images = [
            img_path
            for img_path in self.download_dir.glob("*.jpg")
            if img_path.name not in self.processed_images
        ]

        if not new_images:
            logging.info("No new images to process")
            return

        for img_path in tqdm(new_images, desc="Validating images"):
            self._process_single_image(img_path)

        self.save_processed_images()
        logging.info("Image processing complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Image Scraper")
    parser.add_argument(
        "--concept", default="compact cassette", help="Target concept to scrape"
    )
    parser.add_argument(
        "--output", default="dataset/compact_cassettes", help="Output directory"
    )
    parser.add_argument(
        "--max-images", type=int, default=50, help="Maximum images to collect"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Enable interactive setup"
    )

    args = parser.parse_args()

    scraper = DynamicImageScraper(
        target_concept=args.concept,
        output_dir=args.output,
        max_images=args.max_images,
        interactive=args.interactive,
    )
    scraper.scrape()
