import os
import subprocess
import sys
from pathlib import Path
from typing import List
from huggingface_hub import create_repo, HfApi
from enum import Enum
from typing import List, Optional
import dotenv

dotenv.load_dotenv()


class GGUFQuantization(Enum):
    """Available GGUF quantization levels."""

    Q2_K = "q2_K"  # 2-bit quantization with variable K
    Q3_K_S = "q3_K_S"  # 3-bit quantization with variable K and small context
    Q3_K_M = "q3_K_M"  # 3-bit quantization with variable K and medium context
    Q3_K_L = "q3_K_L"  # 3-bit quantization with variable K and large context
    Q4_0 = "q4_0"  # 4-bit quantization (legacy)
    Q4_K_S = "q4_K_S"  # 4-bit quantization with variable K and small context
    Q4_K_M = "q4_K_M"  # 4-bit quantization with variable K and medium context
    Q4_1 = "q4_1"  # 4-bit quantization (alternative)
    Q5_0 = "q5_0"  # 5-bit quantization (legacy)
    Q5_K_S = "q5_K_S"  # 5-bit quantization with variable K and small context
    Q5_K_M = "q5_K_M"  # 5-bit quantization with variable K and medium context
    Q5_1 = "q5_1"  # 5-bit quantization (alternative)
    Q6_K = "q6_K"  # 6-bit quantization with variable K
    Q8_0 = "q8_0"  # 8-bit quantization
    F16 = "f16"  # 16-bit floating point
    F32 = "f32"  # 32-bit floating point


class LLMConverter:
    def __init__(
        self,
        model_id: str,
        quantization_methods: List[str],
        hf_token: str,
        username: str,
    ):
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.quantization_methods = quantization_methods
        self.hf_token = hf_token
        self.username = username
        self.base_path = Path.cwd()

    def setup_llamacpp(self) -> None:
        """Clone and build llama.cpp for Apple Silicon."""
        llamacpp_path = self.base_path / "llama.cpp"

        if not llamacpp_path.exists():
            print("Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp"], check=True
            )

        print("Building llama.cpp...")
        # First clean
        subprocess.run(["make", "clean"], cwd=llamacpp_path, check=True)

        # Build with Metal support
        subprocess.run(
            [
                "make",
                "LLAMA_METAL=1",  # Enable Metal support for M3
                "-j4",  # Use 4 cores for compilation
            ],
            cwd=llamacpp_path,
            check=True,
        )

        # Verify quantize binary was built
        quantize_path = llamacpp_path / "llama-quantize"
        if not quantize_path.exists():
            raise FileNotFoundError(
                f"Failed to build quantize executable at {quantize_path}\n"
                "Please check the llama.cpp build output for errors."
            )

        print(f"Successfully built quantize at: {quantize_path}")

        # Make the quantize binary executable
        subprocess.run(["chmod", "+x", str(quantize_path)], check=True)

        # Set the conversion script path
        self.convert_script_path = llamacpp_path / "convert_hf_to_gguf.py"
        if not self.convert_script_path.exists():
            raise FileNotFoundError(
                f"Conversion script not found at {self.convert_script_path}"
            )

        # Install requirements
        requirements_file = (
            llamacpp_path / "requirements" / "requirements-convert_hf_to_gguf.txt"
        )
        print(f"Installing requirements from: {requirements_file}")
        subprocess.run(["pip", "install", "-r", str(requirements_file)], check=True)

    def download_model(self) -> None:
        """Download model from Hugging Face."""
        subprocess.run(["git", "lfs", "install"], check=True)
        if not Path(self.model_name).exists():
            subprocess.run(
                ["git", "clone", f"https://huggingface.co/{self.model_id}"], check=True
            )

    def convert_to_fp16(self) -> Path:
        """Convert model to fp16 format."""
        model_path = self.base_path / self.model_name
        fp16_path = model_path / f"{self.model_name.lower()}.fp16.bin"

        print(f"Converting model using: {self.convert_script_path}")
        print(f"Input path: {model_path}")
        print(f"Output path: {fp16_path}")

        subprocess.run(
            [
                "python",
                str(self.convert_script_path),
                "--outfile",
                str(fp16_path),
                "--outtype",
                "f16",
                str(model_path),
            ],
            check=True,
        )
        return fp16_path

    def quantize_model(self, fp16_path: Path) -> None:
        """Quantize the model using specified methods."""
        llamacpp_path = self.base_path / "llama.cpp"
        quantize_path = llamacpp_path / "llama-quantize"

        if not quantize_path.exists():
            raise FileNotFoundError(
                f"Quantize executable not found at {quantize_path}. "
                "Make sure llama.cpp was built successfully."
            )

        for method in self.quantization_methods:
            # Get the value from the enum
            method_str = (
                method.value if isinstance(method, GGUFQuantization) else method
            )
            output_path = (
                Path(self.model_name) / f"{self.model_name.lower()}.{method_str}.gguf"
            )
            print(f"Quantizing with {method_str} to {output_path}")

            subprocess.run(
                [str(quantize_path), str(fp16_path), str(output_path), method_str],
                check=True,
            )

    def upload_to_hub(self) -> None:
        """Upload GGUF files to Hugging Face Hub."""
        api = HfApi()

        # Create repository
        create_repo(
            repo_id=f"{self.username}/{self.model_name}-GGUF",
            repo_type="model",
            exist_ok=True,
            token=self.hf_token,
        )

        # Upload files
        api.upload_folder(
            folder_path=self.model_name,
            repo_id=f"{self.username}/{self.model_name}-GGUF",
            allow_patterns=["*.gguf", "*.md"],
            token=self.hf_token,
        )

    def run(self):
        """Execute the full conversion pipeline."""
        try:
            print("Setting up llama.cpp...")
            self.setup_llamacpp()

            print("Downloading model...")
            self.download_model()

            print("Converting to FP16...")
            fp16_path = self.convert_to_fp16()

            print("Quantizing model...")
            self.quantize_model(fp16_path)

            print("Uploading to Hugging Face Hub...")
            self.upload_to_hub()

            print("Conversion completed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"Error during execution: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)


def main():
    # Configuration
    MODEL_ID = "leonvanbokhorst/Llama-3.2-1B-Instruct-Complaint"
    HF_TOKEN = os.getenv("HF_TOKEN")
    USERNAME = "leonvanbokhorst"

    # Validate HF_TOKEN
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable not set. Please set it with your Hugging Face token:\n"
            "export HF_TOKEN='your_token_here'"
        )

    # Use specific quantization methods
    converter = LLMConverter(
        model_id=MODEL_ID,
        hf_token=HF_TOKEN,
        username=USERNAME,
        quantization_methods=[
            GGUFQuantization.Q2_K,    # Smallest, lowest quality
            GGUFQuantization.Q3_K_S,   
            GGUFQuantization.Q3_K_M,   
            GGUFQuantization.Q3_K_L,   
            GGUFQuantization.Q4_0,     
            GGUFQuantization.Q4_K_S,   
            GGUFQuantization.Q4_K_M,   # Good balance of size/quality
            GGUFQuantization.Q4_1,     
            GGUFQuantization.Q5_0,     
            GGUFQuantization.Q5_K_S,   
            GGUFQuantization.Q5_K_M,   
            GGUFQuantization.Q5_1,     
            GGUFQuantization.Q6_K,     
            GGUFQuantization.Q8_0,     # High quality, larger size
        ],
    )

    # Run the conversion
    converter.run()


if __name__ == "__main__":
    main()
