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


class DynamicImageScraper:

    def __init__(
        self,
        target_concept: str,
        output_dir: str = "dataset/images",
        min_resolution: Tuple[int, int] = (512, 512),
        max_images: int = 100,
        confidence_threshold: float = 0.15,
        min_delay: float = 8.0,
        max_delay: float = 15.0,
        interactive: bool = True,  # Add interactive mode
    ):
        """Initialize the scraper with target concept and parameters."""
        self.target_concept = target_concept
        self.output_dir = Path(output_dir)
        self.min_resolution = min_resolution
        self.max_images = max_images
        self.confidence_threshold = confidence_threshold
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.interactive = interactive
        
        # Initialize counters and tracking
        self.current_index = 0
        self.downloaded_images = set()
        
        # Initialize empty containers
        self.target_categories = {}
        self.mean_target_confidence = None
        self.std_target_confidence = None
        
        # Setup directory
        self.setup_output_directory()
        
        # Setup model and analyze images
        self.setup_model()
        self.target_categories = self._analyze_target_images()
        self.mean_target_confidence = self.target_categories.get('cassette', 0.0)
        
        logging.info(f"\nMean target confidence threshold: {self.mean_target_confidence:.2%}")

        # Add confidence tracking
        self.confidence_scores = []
        self.adaptive_threshold = None

    def setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        # Set target images directory (where existing images are)
        self.target_images_dir = self.output_dir / "target"
        
        # Set output directory for new downloads
        target_dir = self.output_dir / self.target_concept.replace(" ", "_").lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = target_dir
        
        logging.info(f"\nTarget images directory: {self.target_images_dir}")
        logging.info(f"Output directory: {self.output_dir}")

    def _analyze_target_images(self) -> Dict[str, float]:
        """Analyze user-provided target images to build category profile"""
        target_files = list(self.target_images_dir.glob("*.[jJ][pP][gG]"))
        if not target_files:
            raise ValueError(f"No JPG images found in {self.target_images_dir}")

        logging.info(f"\nAnalyzing {len(target_files)} target images...")
        category_scores = defaultdict(float)
        category_counts = defaultdict(int)

        for img_path in target_files:
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
                        top10_prob, top10_catid = torch.topk(probabilities, 10)

                        logging.info(f"\nAnalyzing: {img_path.name}")
                        for prob, cat_id in zip(top10_prob, top10_catid):
                            category = ResNet50_Weights.IMAGENET1K_V2.meta[
                                "categories"
                            ][cat_id].lower()
                            confidence = prob.item()
                            if confidence > 0.01:  # Only track significant detections
                                category_scores[category] += confidence
                                category_counts[category] += 1
                                logging.info(f"- {category:<30} {confidence:.2%}")

            except Exception as e:
                logging.error(f"Error analyzing {img_path}: {e}")
                continue

        # Calculate average scores for categories that appear in multiple images
        normalized_scores = {}
        for category, total_score in category_scores.items():
            count = category_counts[category]
            if count > 1:  # Category must appear in multiple images
                normalized_scores[category] = total_score / len(target_files)

        # Log aggregated results
        logging.info("\nAggregated category profile:")
        for cat, score in sorted(
            normalized_scores.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            logging.info(f"{cat:<30} {score:.2%}")

        # After calculating aggregated profile, store the mean confidence
        self.mean_target_confidence = self.target_categories.get('cassette', 0.0)
        logging.info(f"\nMean target confidence threshold: {self.mean_target_confidence:.2%}")

        return normalized_scores

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
        # Load pre-trained ResNet model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device)
        self.model.eval()

        # Setup image transformation pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
        """Generate search terms based on category analysis"""
        search_terms = [self.target_concept]

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
        """Check if image is too similar to existing ones using perceptual hash."""
        try:
            with Image.open(image_path) as img:
                # Calculate perceptual hash
                current_hash = str(imagehash.average_hash(img, hash_size=16))

                # Get all existing image paths except the current one
                existing_images = [
                    p for p in self.output_dir.glob("*.jpg") if p != image_path
                ]

                # Compare with existing images
                for existing_path in existing_images:
                    try:
                        with Image.open(existing_path) as existing_img:
                            existing_hash = str(
                                imagehash.average_hash(existing_img, hash_size=16)
                            )
                            if current_hash == existing_hash:
                                logging.info(
                                    f"Found duplicate: {image_path} matches {existing_path}"
                                )
                                return True
                    except Exception as e:
                        logging.error(
                            f"Error processing existing image {existing_path}: {e}"
                        )
                        continue

                return False

        except Exception as e:
            logging.error(f"Error checking for duplicates: {e}")
            return False

    def validate_image(self, image_path: Path) -> bool:
        """Check if image meets all requirements and update metrics."""
        try:
            # Check if file exists and is a valid image
            if not image_path.exists():
                logging.info(f"File does not exist: {image_path}")
                return False

            with Image.open(image_path) as img:
                # Check resolution
                if (
                    img.size[0] < self.min_resolution[0]
                    or img.size[1] < self.min_resolution[1]
                ):
                    logging.info(
                        f"Image {image_path} below minimum resolution: {img.size}"
                    )
                    return False

                # Check if it's the target object
                is_target, confidence = self.is_target_resnet(image_path)
                if not is_target:
                    logging.info(
                        f"Image {image_path} rejected with confidence {confidence:.2%}"
                    )
                    return False

                # Check for duplicates
                if self.is_duplicate(image_path):
                    logging.info(f"Image {image_path} is a duplicate")
                    return False

                logging.info(
                    f"Image {image_path} accepted with confidence {confidence:.2%}"
                )
                return True

        except Exception as e:
            logging.error(f"Error validating image {image_path}: {e}")
            return False

    def setup_logging(self) -> None:
        """Configure logging for the scraper."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def cleanup_invalid_images(self) -> None:
        """Remove images that don't meet the requirements."""
        # Only process images in the main output directory, not the target (seed) folder
        for img_path in self.output_dir.glob("*.jpg"):
            # Skip files in the target subdirectory
            if "target" in str(img_path):
                continue

            try:
                # Only remove if validation explicitly fails
                if not self.validate_image(img_path):
                    logging.info(f"Removing invalid image: {img_path}")
                    img_path.unlink()
            except Exception as e:
                logging.error(f"Error during cleanup of {img_path}: {e}")

    def get_next_filename(self) -> str:
        """Generate unique filename for each image."""
        self.current_index += 1
        return f"{self.current_index:06d}.jpg"

    def scrape(self) -> None:
        """Scrape images with adaptive search terms based on ResNet performance"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        while self.current_index < self.max_images:
            # Generate new search terms based on ResNet performance
            self.search_terms = self.generate_search_terms()

            images_per_term = max(2, self.max_images // len(self.search_terms))

            for term in self.search_terms:
                if self.current_index >= self.max_images:
                    break

                # Clean up the search term (remove numbering if present)
                clean_term = term.split(". ", 1)[-1] if ". " in term else term

                google_crawler = GoogleImageCrawler(
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=1,
                    storage={"root_dir": str(self.output_dir)},
                )

                # Override the file namer
                google_crawler.file_namer = lambda _: self.get_next_filename()

                filters = {
                    "size": "large",
                    "type": "photo",
                    "license": "commercial,modify",
                }

                try:
                    # Add random delay between requests
                    time.sleep(random.uniform(self.min_delay, self.max_delay))

                    google_crawler.crawl(
                        keyword=clean_term,
                        max_num=images_per_term,
                        min_size=self.min_resolution,
                        filters=filters,
                        file_idx_offset="auto",
                    )

                    # Add delay after batch completion
                    time.sleep(random.uniform(self.min_delay, self.max_delay))

                except Exception as e:
                    logging.error(f"Error crawling for term '{clean_term}': {e}")
                    continue

                # Clean up invalid images after each term
                self.cleanup_invalid_images()

    def bootstrap_target_folder(self) -> None:
        """Initialize target folder with first-pass search results"""
        target_path = self.output_dir / "target"
        target_path.mkdir(parents=True, exist_ok=True)

        if not list(target_path.glob("*.[jJ][pP][gG]")):
            logging.info("No seed images found. Bootstrapping target folder...")

            # Temporary strict settings for initial search
            original_threshold = self.confidence_threshold
            self.confidence_threshold = 0.4  # Stricter for seed images

            # Basic search term for bootstrap
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
                        storage={"root_dir": str(target_path)},
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
                    for img_path in target_path.glob("*.[jJ][pP][gG]"):
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
                for img_path in target_path.glob("*.[jJ][pP][gG]"):
                    try:
                        _, confidence = self.is_target_resnet(img_path)
                        scored_images.append((img_path, confidence))
                    except Exception:
                        continue

                # Sort by confidence and keep top 5
                scored_images.sort(key=lambda x: x[1], reverse=True)
                for img_path, _ in scored_images[5:]:
                    img_path.unlink()

                if not list(target_path.glob("*.[jJ][pP][gG]")):
                    logging.warning(
                        "Failed to bootstrap target folder with high-confidence images"
                    )
                else:
                    logging.info(
                        f"Successfully bootstrapped target folder with {len(list(target_path.glob('*.[jJ][pP][gG]')))} images"
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
        """Download initial target images"""
        print("\nDownloading sample images...")

        # Basic search terms for bootstrap
        terms = [
            f"{self.target_concept} high quality photo",
            f"{self.target_concept} professional product shot",
            f"{self.target_concept} clear detailed image",
        ]

        for term in terms:
            crawler = GoogleImageCrawler(
                storage={"root_dir": str(self.target_images_dir)}
            )

            filters = {"size": "large", "type": "photo", "license": "commercial,modify"}

            crawler.crawl(
                keyword=term, max_num=2, min_size=self.min_resolution, filters=filters
            )
            time.sleep(random.uniform(2, 4))

        print("\nDownloaded potential target images.")
        self._review_images()

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

    def update_adaptive_threshold(self, new_confidence: float):
        """Update adaptive threshold based on new valid matches."""
        self.confidence_scores.append(new_confidence)
        
        if len(self.confidence_scores) >= 3:  # Start adapting after 3 samples
            # Calculate mean and standard deviation of recent scores
            recent_scores = self.confidence_scores[-5:]  # Use last 5 scores
            mean_confidence = sum(recent_scores) / len(recent_scores)
            
            # Start with mean_target_confidence, then gradually shift towards recent mean
            weight = min(len(self.confidence_scores) / 10, 0.8)  # Max 80% weight to recent scores
            self.adaptive_threshold = (
                (1 - weight) * self.mean_target_confidence +
                weight * mean_confidence
            ) * 0.85  # Still keep 85% as safety factor
            
            logging.info(f"\nUpdated adaptive threshold:")
            logging.info(f"- Recent mean confidence: {mean_confidence:.2%}")
            logging.info(f"- Adaptation weight: {weight:.2%}")
            logging.info(f"- New threshold: {self.adaptive_threshold:.2%}")

    def is_target_resnet(self, image_path: Path) -> Tuple[bool, float]:
        """Check if image contains target object using ResNet model."""
        try:
            with Image.open(image_path) as img:
                scaled_img = self._scale_image_for_models(img)
                img_tensor = self.transform(scaled_img.convert('RGB')).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    
                    # Get top prediction only
                    top_prob, top_catid = torch.topk(probabilities, 1)
                    
                    # Get primary category from target analysis
                    primary_category = max(self.target_categories.items(), key=lambda x: x[1])[0]
                    
                    # Check only the top category
                    category = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][top_catid[0]].lower()
                    confidence = top_prob[0].item()
                    
                    # Must be primary category AND meet confidence threshold
                    if primary_category in category:
                        # Use adaptive threshold if available, otherwise use initial mean
                        threshold = self.adaptive_threshold if self.adaptive_threshold else self.mean_target_confidence * 0.85
                        is_valid = confidence >= threshold
                        
                        logging.info(f"\nValidating {image_path.name}:")
                        logging.info(f"- Top category: {category}")
                        logging.info(f"- Confidence: {confidence:.2%}")
                        logging.info(f"- Current threshold: {threshold:.2%}")
                        logging.info(f"- Result: {'✓' if is_valid else '✗'}")
                        
                        if is_valid:
                            self.update_adaptive_threshold(confidence)
                        
                        return is_valid, confidence
                    
                    return False, confidence
                    
        except Exception as e:
            logging.error(f"Error in ResNet validation: {e}")
            return False, 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Image Scraper")
    parser.add_argument(
        "--concept", default="tape cassette", help="Target concept to scrape"
    )
    parser.add_argument(
        "--output", default="dataset/tape_cassettes", help="Output directory"
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
