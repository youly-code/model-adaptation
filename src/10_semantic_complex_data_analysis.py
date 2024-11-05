from typing import Dict, Any, List
import pandas as pd
import numpy as np
from enum import Enum
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import init, Fore, Style

# Initialize colorama for cross-platform color support
init()

# Basic logging setup with custom formatter
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove any existing handlers and disable propagation to root logger
logger.handlers = []
logger.propagate = False  # This prevents messages from propagating to root logger

# Add a single handler with our formatter
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



class DataCategory(Enum):
    """Basic categories for data fields"""
    NUMERIC = "numeric"
    TEXT = "text"
    METADATA = "metadata"  # Fields with few unique values
    MIXED = "mixed"       # Fields with multiple types
    JSON = "json"         # JSON-structured data

class FieldPattern(Enum):
    """Common patterns found in data fields"""
    IDENTIFIER = "identifier"      # IDs, keys, codes
    PERSONAL = "personal"          # Names, ages, etc.
    TEMPORAL = "temporal"          # Dates and times
    FINANCIAL = "financial"        # Money-related
    CATEGORICAL = "categorical"    # Types, categories
    MEASUREMENT = "measurement"    # Scores, ratings
    LOCATION = "location"          # Places, addresses
    CONTACT = "contact"            # Email, phone
    SYSTEM = "system"             # System metadata
    PREFERENCE = "preference"      # User settings

class SemanticRouter:
    """Analyzes and categorizes data fields based on their content and meaning."""

    def __init__(self):
        logger.info("Initializing SemanticRouter...")
        # Load the model for understanding field meanings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize pattern descriptions
        self.pattern_descriptions = {
            FieldPattern.IDENTIFIER: "id key code identifier uuid",
            FieldPattern.PERSONAL: "name age gender birth personal",
            FieldPattern.TEMPORAL: "date time timestamp year month",
            FieldPattern.FINANCIAL: "price cost salary payment amount",
            FieldPattern.CATEGORICAL: "type category status level class",
            FieldPattern.MEASUREMENT: "score rating value measure",
            FieldPattern.LOCATION: "address city country location",
            FieldPattern.CONTACT: "email phone contact",
            FieldPattern.SYSTEM: "created modified version status",
            FieldPattern.PREFERENCE: "setting preference option config"
        }
        
        # Pre-compute embeddings for patterns
        self.pattern_embeddings = {
            pattern: self.model.encode(desc)
            for pattern, desc in self.pattern_descriptions.items()
        }

    def analyze_field(self, field_name: str, data: pd.Series) -> Dict[str, Any]:
        """Analyze a single data field."""
        # Get sample of non-null values
        sample_values = data.dropna().head(5).tolist()
        
        # Get basic category
        category = self._get_basic_category(data)
        
        # Get semantic pattern
        pattern = self._get_field_pattern(field_name, sample_values)
        
        # Analyze value distribution
        value_stats = self._analyze_values(data)
        
        return {
            "category": category.value,
            "pattern": pattern.value,
            "sample_values": sample_values,
            "null_percentage": (data.isna().sum() / len(data)) * 100,
            "stats": value_stats
        }

    def _get_basic_category(self, data: pd.Series) -> DataCategory:
        """Determine the basic category of the data."""
        if data.isna().all():
            return DataCategory.TEXT  # Default for empty fields

        # Get non-null values
        non_null = data.dropna()
        if len(non_null) == 0:
            return DataCategory.TEXT

        # Check for mixed types
        types = {type(x) for x in non_null}
        if len(types) > 1:
            return DataCategory.MIXED

        # Handle complex types (dict, list)
        first_valid = next(x for x in non_null if x is not None)
        if isinstance(first_valid, dict):
            return DataCategory.JSON
        if isinstance(first_valid, list):
            return DataCategory.MIXED

        # Check if numeric
        if pd.api.types.is_numeric_dtype(data):
            return DataCategory.NUMERIC

        # For simple types, check if metadata (few unique values)
        try:
            if len(non_null.unique()) / len(non_null) < 0.1:
                return DataCategory.METADATA
        except TypeError:
            # If unique() fails due to unhashable types, treat as mixed
            return DataCategory.MIXED

        return DataCategory.TEXT

    def _get_field_pattern(self, field_name: str, sample_values: List[Any]) -> FieldPattern:
        """Determine the semantic pattern of the field."""
        # Combine field name with samples for context
        field_text = f"{field_name} " + " ".join(str(x) for x in sample_values)
        field_embedding = self.model.encode(field_text)
        
        # Find best matching pattern
        similarities = {
            pattern: cosine_similarity([field_embedding], [pattern_emb])[0][0]
            for pattern, pattern_emb in self.pattern_embeddings.items()
        }
        
        return max(similarities.items(), key=lambda x: x[1])[0]

    def _analyze_values(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of values in the field."""
        non_null = data.dropna()
        if len(non_null) == 0:
            return {"empty": True}

        try:
            # Try standard analysis for hashable types
            return {
                "unique_count": len(non_null.unique()),
                "unique_ratio": len(non_null.unique()) / len(non_null),
                "most_common": non_null.value_counts().head(3).to_dict()
            }
        except TypeError:
            # Handle unhashable types (lists, dicts, etc.)
            logger.debug("Unhashable type detected, using string representation")
            # Convert to string representation for analysis
            non_null_str = non_null.astype(str)
            return {
                "unique_count": len(non_null_str.unique()),
                "unique_ratio": len(non_null_str.unique()) / len(non_null_str),
                "most_common": non_null_str.value_counts().head(3).to_dict(),
                "contains_complex_types": True
            }

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze all fields in a dataset."""
        logger.info(f"Analyzing dataset with {len(df.columns)} columns")
        
        results = {}
        for column in df.columns:
            logger.info(f"Analyzing field: {column}")
            results[column] = self.analyze_field(column, df[column])
        
        return results

def visualize_analysis(results: Dict[str, Dict[str, Any]]) -> None:
    """Create a visual representation of the data analysis."""
    
    # Group fields by pattern
    pattern_groups = {}
    for field, analysis in results.items():
        pattern = analysis['pattern']
        if pattern not in pattern_groups:
            pattern_groups[pattern] = []
        pattern_groups[pattern].append((field, analysis))

    logger.info("\n=== Data Structure Visualization ===\n")

    # Print header
    logger.info(f"{Fore.CYAN}📊 Data Field Analysis Summary{Style.RESET_ALL}")
    logger.info("=" * 50)

    # Print pattern groups
    for pattern, fields in pattern_groups.items():
        # Pattern header
        logger.info(f"\n{Fore.GREEN}📌 {pattern.upper()}{Style.RESET_ALL}")
        logger.info("  " + "-" * 40)

        # Fields in this pattern
        for field_name, analysis in fields:
            # Category indicator
            category_color = {
                'numeric': Fore.BLUE,
                'text': Fore.MAGENTA,
                'metadata': Fore.YELLOW,
                'mixed': Fore.RED,
                'json': Fore.CYAN
            }.get(analysis['category'], Fore.WHITE)

            # Quality indicator based on null percentage
            null_pct = analysis['null_percentage']
            if null_pct <= 5:
                quality = "🟢"    # Excellent
            elif null_pct <= 20:
                quality = "🟡"    # Good
            elif null_pct <= 50:
                quality = "🟠"    # Fair
            else:
                quality = "🔴"    # Poor

            # Complexity indicator
            complexity = "📦" if analysis.get('stats', {}).get('contains_complex_types') else "📄"

            # Print field info
            logger.info(f"  {quality} {complexity} {category_color}{field_name}{Style.RESET_ALL}")

            # Print details with indentation
            sample_str = str(analysis['sample_values'][:2])[:50] + "..."
            logger.info(f"     └─ Samples: {Fore.WHITE}{sample_str}{Style.RESET_ALL}")
            logger.info(f"     └─ Nulls: {Fore.WHITE}{analysis['null_percentage']:.1f}%{Style.RESET_ALL}")

            # Print unique value ratio if available
            if 'stats' in analysis and 'unique_ratio' in analysis['stats']:
                unique_ratio = analysis['stats']['unique_ratio']
                unique_str = f"{unique_ratio:.1%}"
                logger.info(f"     └─ Unique: {Fore.WHITE}{unique_str}{Style.RESET_ALL}")

    # Print legend
    logger.info("\n" + "=" * 50)
    logger.info(f"{Fore.CYAN}📋 Legend:{Style.RESET_ALL}")
    logger.info(
        "Quality: 🟢 Excellent (≤5% nulls) 🟡 Good (≤20%) 🟠 Fair (≤50%) 🔴 Poor")
    logger.info("Type: 📄 Simple 📦 Complex")
    logger.info(f"Categories: "
               f"{Fore.BLUE}Numeric {Fore.MAGENTA}Text {Fore.YELLOW}Metadata "
               f"{Fore.RED}Mixed {Fore.CYAN}JSON{Style.RESET_ALL}")

def demonstrate_semantic_routing():
    """Show how to use the semantic router with complex enterprise data."""
    # Set consistent length for all arrays
    n_rows = 22
    
    data = {
        # Identifiers with different formats
        "employee_id": ([f"EMP{i:05d}" for i in range(1000, 1020)] + [None, "TEMP-123", "CONT#45"])[:n_rows],
        "legacy_id": (["OLD" + str(i) for i in range(5)] + [None] * 10 + ["LEGACY-X", "UNKNOWN", "N/A"] + [None] * 4)[:n_rows],
        
        # Personal information with quality issues
        "full_name": ([
            "John Doe", "Jane Smith", "Bob Wilson", "Alice Brown",
            "李明", "María García", "Johann Schmidt", None,
            "N/A", "UNKNOWN", "Test User",
            "Smith, John", "doe, jane", "  Bob  Wilson  ",
            "Alice/Brown", "#Invalid Name#", "", "☺Jane☺",
            "John Doe Jr.", "Jane van der Smith", "Bob O'Wilson", "Brown, Alice K."
        ])[:n_rows],
        
        # Contact information with various formats
        "email": ([
            "john@company.com", "jane.smith@company.com", None,
            "bob.wilson@legacy-company.net", "alice.brown@company.com",
            "invalid-email", "test@", "@company.com", "",
            "multiple@emails,another@email.com", "李明@company.cn",
            "jane@company.com", None, "BOB@COMPANY.COM",
            "alice+test@company.com", "no-reply@company.com",
            "support@company.com", None, "sales@company.com",
            "info@company.com", "contact@company.com", None
        ])[:n_rows],
        
        # Complex nested data
        "contact_details": ([
            {"phone": "123-456-7890", "address": "123 Main St", "preferred": "email"},
            {"phone": None, "address": "456 Oak Rd", "preferred": "phone"},
            None,
            {"phone": "987-654-3210", "address": None, "preferred": None},
            {"phone": "invalid", "address": "789 Pine Ave", "preferred": "mail"},
            {},  # Empty dict
            {"phone": ["123-456-7890", "098-765-4321"], "address": "Multi Phone"},
            {"invalid_key": "value"},
            {"phone": "123-456-7890", "address": {"street": "Complex", "city": "Test"}},
            None,
        ] + [None] * (n_rows - 10))[:n_rows],
        
        # Temporal data with inconsistencies
        "join_date": ([
            "2022-01-01", "2021-06-15", None, "2022-03-01",
            "01/01/2023", "Dec 1, 2021", "2022-08-15T00:00:00",
            "invalid_date", "2022", "2023-13-01",
            "2022-04-31", None, "ongoing", "TBD",
            "2023-01-01 09:00:00", "2024", None, "Q1 2023",
            "2023-W01", "2023-01", None, "yesterday"
        ])[:n_rows],
        
        # Hierarchical department structure
        "department": ([
            {"name": "Sales", "division": "North", "cost_center": "CC001"},
            {"name": "Sales", "division": "South", "cost_center": "CC002"},
            {"name": "IT", "division": "Global", "cost_center": "CC003"},
            {"name": "HR", "division": None, "cost_center": None},
            None,
            {"name": "Engineering", "division": "R&D", "cost_center": "CC004"},
            {"name": "Sales", "division": "East", "cost_center": "CC005"},
            {"invalid_structure": True},
            {},
            {"name": "Marketing", "division": "Global", "cost_center": "CC006"},
        ] + [None] * (n_rows - 10))[:n_rows],
        
        # Financial data with anomalies
        "compensation": ([
            {"base": 50000, "bonus": 5000, "currency": "USD"},
            {"base": 65000, "bonus": None, "currency": "USD"},
            {"base": -85000, "bonus": 0, "currency": "EUR"},
            {"base": "67000", "bonus": "N/A", "currency": "USD"},
            {"base": None, "bonus": None, "currency": None},
            {"base": 88000},
            {"base": 63000, "bonus": "15000", "currency": "GBP"},
            {"invalid_format": True},
            {},
            {"base": "TBD", "bonus": None, "currency": "USD"},
        ] + [None] * (n_rows - 10))[:n_rows],
        
        # Performance metrics
        "performance_scores": ([
            [4, 5, 3, 4], [5, 5, 5, 5], None,
            [2, "N/A", 4, 3], [1, 2, "pending", 4],
            [], [1], [1, 2, 3, 4, 5, 6],
            "Under Review", None, "4,4,4,4",
            [None, None, None, None], ["A", "B", "C", "D"],
        ] + [None] * (n_rows - 13))[:n_rows],
        
        # System metadata
        "last_updated": [pd.Timestamp.now() - pd.Timedelta(days=i) for i in range(n_rows)],
        "version": (["1.0"] * 5 + ["2.0"] * 5 + ["3.0-beta"] * 5 + [None] * 7)[:n_rows],
        "status": (["ACTIVE"] * 10 + ["INACTIVE"] * 5 + ["PENDING"] * 3 + ["SUSPENDED"] * 2 + [None] * 2)[:n_rows]
    }

    df = pd.DataFrame(data)
    router = SemanticRouter()
    results = router.analyze_dataset(df)
    
    visualize_analysis(results)

if __name__ == "__main__":
    demonstrate_semantic_routing()
