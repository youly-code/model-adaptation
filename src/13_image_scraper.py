from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import logging
from pathlib import Path
from typing import Tuple, Set
import time
import random
from fake_useragent import UserAgent
import imagehash

class DelayedGoogleImageCrawler(GoogleImageCrawler):
    def __init__(self, min_delay, max_delay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.ua = UserAgent()

    def process_links(self, links):
        for link in links:
            time.sleep(random.uniform(self.min_delay, self.max_delay))
            super().process_links([link])

    def get_headers(self):
        """Generate headers for the request."""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

class CassetteImageScraper:
    def __init__(
        self,
        output_dir: str = "dataset/cassettes",
        min_resolution: Tuple[int, int] = (512, 512),
        max_images: int = 100,
        confidence_threshold: float = 0.3,
        min_delay: float = 8.0,
        max_delay: float = 15.0,
        hash_threshold: int = 8  # Maximum hash difference to consider images similar
    ):
        self.output_dir = Path(output_dir)
        self.min_resolution = min_resolution
        self.max_images = max_images
        self.confidence_threshold = confidence_threshold
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.setup_model()
        self.setup_logging()
        self.current_index = 0  # Add image counter
        self.image_hashes: Set[str] = set()
        self.hash_threshold = hash_threshold
    
    def setup_model(self) -> None:
        """Initialize the image classification model."""
        # Load pre-trained ResNet model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(self.device)
        self.model.eval()
        
        # Setup image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Expanded cassette-related ImageNet classes
        self.target_classes = {
            'cassette player', 'tape player', 'audio tape',
            'tape deck', 'audio system', 'stereo', 
            'radio', 'sound system', 'recorder'
        }
        
    def is_cassette(self, image_path: Path) -> bool:
        """Check if the image contains a cassette using the model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if the image likely contains a cassette
        """
        try:
            with Image.open(image_path).convert('RGB') as img:
                # Transform image for model input
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    
                    # Get top 10 predictions
                    top10_prob, top10_catid = torch.topk(probabilities, 10)
                    
                    for prob, cat_id in zip(top10_prob, top10_catid):
                        category = ResNet50_Weights.IMAGENET1K_V2.meta["categories"][cat_id].lower()
                        # Log all predictions for debugging
                        logging.info(f"Category: {category}, Probability: {prob.item():.2%}")
                        
                        if any(term in category for term in self.target_classes):
                            logging.info(f"Matched category in {image_path} "
                                       f"({category}: {prob.item():.2%})")
                            return True
                            
                logging.info(f"No cassette detected in {image_path}")
                return False
                
        except Exception as e:
            logging.error(f"Error classifying image {image_path}: {e}")
            return False

    def is_duplicate(self, image_path: Path) -> bool:
        """Check if image is too similar to existing ones using perceptual hash.
        
        Args:
            image_path: Path to the image to check
            
        Returns:
            bool: True if image is too similar to existing ones
        """
        try:
            with Image.open(image_path) as img:
                # Calculate perceptual hash of new image
                new_hash = str(imagehash.average_hash(img))
                
                # For exact duplicates
                if new_hash in self.image_hashes:
                    logging.info(f"Exact duplicate found: {image_path}")
                    return True
                
                # Check for similar images
                for existing_hash in self.image_hashes:
                    hash_diff = sum(c1 != c2 for c1, c2 in zip(new_hash, existing_hash))
                    if hash_diff <= self.hash_threshold:
                        logging.info(f"Similar image found: {image_path} "
                                   f"(hash difference: {hash_diff})")
                        return True
                
                # If not duplicate, add to hash set
                self.image_hashes.add(new_hash)
                return False
                
        except Exception as e:
            logging.error(f"Error checking for duplicates: {e}")
            return False

    def validate_image(self, image_path: Path) -> bool:
        """Check if image meets all requirements.

        Args:
            image_path: Path to the image file

        Returns:
            bool: True if image meets all requirements, False otherwise
        """
        # Check resolution first (faster than ML inference)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if not (width >= self.min_resolution[0] and 
                       height >= self.min_resolution[1]):
                    return False
                
                # Check for duplicates before running ML
                if self.is_duplicate(image_path):
                    logging.info(f"Removing duplicate image: {image_path}")
                    return False
                
                # If not duplicate and resolution good, check if it's a cassette
                return self.is_cassette(image_path)
                
        except Exception as e:
            logging.error(f"Error validating image {image_path}: {e}")
            return False

    def setup_logging(self) -> None:
        """Configure logging for the scraper."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def cleanup_invalid_images(self) -> None:
        """Remove images that don't meet the minimum resolution requirements."""
        for img_path in self.output_dir.glob("*.jpg"):
            if not self.validate_image(img_path):
                logging.info(f"Removing low-resolution image: {img_path}")
                img_path.unlink()

    def get_next_filename(self) -> str:
        """Generate unique filename for each image."""
        self.current_index += 1
        return f"{self.current_index:06d}.jpg"
    
    def scrape(self) -> None:
        """Scrape cassette tape images from Google Images."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        search_terms = [
            'vintage cassette tape',
            'audio cassette 80s',
            'retro music tape',
            'compact cassette',
            'analog tape recording',
            'cassette tape collection',  # Added more search terms
            'music cassette vintage',
            'tape deck cassette',
            'audio tape collection',
            'classic cassette tape'
        ]
        
        images_per_term = max(2, self.max_images // len(search_terms))
        
        for term in search_terms:
            google_crawler = DelayedGoogleImageCrawler(
                min_delay=self.min_delay,
                max_delay=self.max_delay,
                storage={'root_dir': str(self.output_dir)},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=1
            )
            
            # Override the file namer
            google_crawler.file_namer = lambda _: self.get_next_filename()
            
            filters = {
                'size': 'large',
                'type': 'photo',
                'license': 'commercial,modify'
            }
            
            try:
                google_crawler.crawl(
                    keyword=term,
                    max_num=images_per_term,
                    filters=filters,
                    file_idx_offset='auto'  # Let the crawler handle offsets
                )
                time.sleep(random.uniform(self.min_delay, self.max_delay))  # Add delay between search terms
                
            except Exception as e:
                logging.error(f"Error crawling for term '{term}': {e}")
                continue

if __name__ == "__main__":
    scraper = CassetteImageScraper(
        output_dir="dataset/cassettes",
        min_resolution=(512, 512),
        max_images=50,
        confidence_threshold=0.3,
        min_delay=8.0,
        max_delay=15.0,
        hash_threshold=8  # Adjust based on how strict you want duplicate detection to be
    )
    scraper.scrape()
