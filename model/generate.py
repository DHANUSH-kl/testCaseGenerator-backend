import os
import re
import warnings
from typing import List, Dict

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    set_seed
)

warnings.filterwarnings("ignore")
load_dotenv()

class AdvancedFreeModelGenerator:
    """
    Advanced Test Case Generator leveraging multiple Hugging Face models
    with fallback to intelligent template-based generation.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv("HF_TOKEN", None)
        self.current_model = None
        self.current_tokenizer = None
        self.model_config = None

        # Login if token available
        if self.hf_token:
            login(token=self.hf_token)

        self.models = self._define_model_hierarchy()
        self._load_best_model()

    def _define_model_hierarchy(self) -> List[Dict]:
        """Defines the prioritized list of models to attempt loading."""
        return [
            {
                "name": "microsoft/DialoGPT-large",
                "type": "causal",
                "description": "Large conversational model - Best quality",
                "max_length": 1024,
                "requires_gpu": True,
            },
            {
                "name": "facebook/blenderbot-400M-distill",
                "type": "seq2seq",
                "description": "Conversational AI - Good for structured output",
                "max_length": 512,
                "requires_gpu": False,
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "type": "causal",
                "description": "Medium conversational model - Balanced",
                "max_length": 768,
                "requires_gpu": False,
            },
            {
                "name": "distilgpt2",
                "type": "causal",
                "description": "Lightweight GPT - Fast and reliable",
                "max_length": 512,
                "requires_gpu": False,
            },
            {
                "name": "google/flan-t5-large",
                "type": "seq2seq",
                "description": "Large instruction-following model",
                "max_length": 512,
                "requires_gpu": True,
            },
            {
                "name": "google/flan-t5-base",
                "type": "seq2seq",
                "description": "Base instruction model - Fallback",
                "max_length": 512,
                "requires_gpu": False,
            },
        ]

    def _load_best_model(self):
        """Attempt to load best model fitting current system resources."""
        has_gpu = torch.cuda.is_available()
        gpu_mem_gb = 0.0
        if has_gpu:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Detected GPU with {gpu_mem_gb:.1f} GB VRAM")

        for model_cfg in self.models:
            try:
                # Skip if model requires GPU but conditions not met
                if model_cfg["requires_gpu"] and (not has_gpu or gpu_mem_gb < 4):
                    print(f"Skipping {model_cfg['name']} due to GPU requirements.")
                    continue

                print(f"Loading model: {model_cfg['name']} ({model_cfg['description']})")

                if model_cfg["type"] == "causal":
                    if self._load_causal_model(model_cfg):
                        print(f"✅ Loaded causal model: {model_cfg['name']}")
                        return
                else:
                    if self._load_seq2seq_model(model_cfg):
                        print(f"✅ Loaded seq2seq model: {model_cfg['name']}")
                        return
            except Exception as e:
                print(f"❌ Failed to load {model_cfg['name']}: {str(e)[:100]}")

        print("⚠️ No suitable model loaded, using template-based fallback.")

    def _load_causal_model(self, cfg: Dict) -> bool:
        """Load a causal language model with its tokenizer."""
        try:
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                cfg["name"], use_auth_token=self.hf_token, padding_side="left"
            )
            # Ensure pad token exists
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token

            self.current_model = AutoModelForCausalLM.from_pretrained(
                cfg["name"],
                use_auth_token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if self.device == "cpu":
                self.current_model.to(self.device)

            self.model_config = cfg
            return True
        except Exception as e:
            print(f"Causal model load error: {e}")
            return False

    def _load_seq2seq_model(self, cfg: Dict) -> bool:
        """Load a seq2seq model with its tokenizer."""
        try:
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                cfg["name"], use_auth_token=self.hf_token
            )
            self.current_model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg["name"],
                use_auth_token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.current_model.to(self.device)

            self.model_config = cfg
            return True
        except Exception as e:
            print(f"Seq2Seq model load error: {e}")
            return False

    def generate_test_cases(self, srs_text: str) -> List[str]:
        """
        Generate test cases from an SRS text.
        Fallbacks to template-based generation if no model loaded or error occurs.
        """
        if self.current_model is None:
            print("No model loaded, using template-based generation.")
            return self._generate_template_based(srs_text)

        try:
            if self.model_config["type"] == "causal":
                return self._generate_causal(srs_text)
            else:
                return self._generate_seq2seq(srs_text)
        except Exception as e:
            print(f"Generation error: {e}\nUsing template-based fallback.")
            return self._generate_template_based(srs_text)

    def _generate_causal(self, srs_text: str) -> List[str]:
        """Generate test cases using a causal language model."""
        prompt = (
            "<|system|>You are a senior QA engineer creating test cases.<|endoftext|>\n"
            "<|user|>Create test cases for this software requirement:\n\n"
            f"{srs_text[:500]}\n\n"
            "Format each test case as:\n"
            "TC-001: [Test Objective] - [Steps] - [Expected Result]\n\n"
            "Test Cases:\nTC-001:"
        )

        inputs = self.current_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config["max_length"] - 300,
            padding=True,
        ).to(self.device)

        set_seed(42)
        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.current_tokenizer.pad_token_id,
                eos_token_id=self.current_tokenizer.eos_token_id,
                early_stopping=True,
            )

        generated_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_test_cases(generated_text)

    def _generate_seq2seq(self, srs_text: str) -> List[str]:
        """Generate test cases using a seq2seq model."""
        prompt = (
            "Generate 8 professional test cases for this software specification:\n\n"
            f"{srs_text[:400]}\n\n"
            "Requirements:\n"
            "- Format: TC-XXX: [Clear objective] - [Detailed steps] - [Expected outcome]\n"
            "- Cover: functionality, validation, error handling, security\n"
            "- Be specific and actionable\n\n"
            "Test cases:"
        )

        inputs = self.current_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config["max_length"],
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                early_stopping=True,
                num_beams=2,
            )

        generated_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_test_cases(generated_text)

    def _extract_test_cases(self, text: str) -> List[str]:
        """Extract and clean test cases from generated text using regex patterns."""
        patterns = [
            r'TC-\d{3}:.*?(?=TC-\d{3}:|$)',      # TC-001: ... TC-002: ...
            r'TC\d{3}:.*?(?=TC\d{3}:|$)',        # TC001: ... TC002: ...
            r'\d+\.\s*TC-\d{3}:.*?(?=\d+\.\s*TC-\d{3}:|$)',  # 1. TC-001: ... 2. TC-002: ...
        ]

        test_cases = []
        for pat in patterns:
            matches = re.findall(pat, text, re.DOTALL | re.IGNORECASE)
            if matches:
                test_cases.extend(matches)
                break

        cleaned_cases = []
        seen = set()
        for i, tc in enumerate(test_cases[:10], start=1):
            cleaned = self._clean_test_case(tc.strip())
            if (
                len(cleaned) > 50
                and cleaned not in seen
                and self._contains_test_structure(cleaned)
            ):
                formatted = re.sub(r'TC-?\d{3}:?', f'TC-{i:03}:', cleaned)
                if not formatted.startswith("TC-"):
                    formatted = f"TC-{i:03}: {formatted}"
                cleaned_cases.append(formatted)
                seen.add(cleaned)

        # Add templates if too few test cases
        if len(cleaned_cases) < 6:
            templates = self._get_smart_templates(text)
            for tmpl in templates:
                if len(cleaned_cases) >= 8:
                    break
                if tmpl not in seen:
                    cleaned_cases.append(tmpl)

        return cleaned_cases[:8]

    def _clean_test_case(self, tc: str) -> str:
        """Clean individual test case text."""
        cleaned = " ".join(tc.split())  # remove excess whitespace
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)  # remove leading numbering
        return cleaned

    def _contains_test_structure(self, text: str) -> bool:
        """Check if the text looks like a valid test case (contains steps and expected)."""
        steps_keywords = ["steps", "step", "execute", "action", "input", "perform"]
        expected_keywords = ["expected", "result", "outcome", "verify", "should be"]

        text_lower = text.lower()
        return any(k in text_lower for k in steps_keywords) and any(k in text_lower for k in expected_keywords)

    def _get_smart_templates(self, text: str) -> List[str]:
        """Generate fallback template test cases for better coverage."""
        return [
            "TC-001: Verify login with valid credentials - Enter username and password - User is logged in successfully.",
            "TC-002: Check form validation for empty fields - Submit the form without input - Proper validation error messages are displayed.",
            "TC-003: Test password reset functionality - Request password reset email - Email is sent with reset instructions.",
            "TC-004: Validate data encryption during transmission - Transmit sensitive data - Data is encrypted using TLS.",
            "TC-005: Test user logout process - Click logout button - User is redirected to login page.",
            "TC-006: Check error handling for invalid inputs - Enter invalid data in form fields - Appropriate error messages are shown.",
            "TC-007: Verify session timeout - Stay inactive for session duration - User is logged out automatically.",
            "TC-008: Validate accessibility features - Use screen reader and keyboard navigation - Application is accessible as per standards."
        ]

    def _generate_template_based(self, srs_text: str) -> List[str]:
        """Fallback method: generate test cases from predefined templates."""
        print("Using static template-based test cases fallback.")
        return self._get_smart_templates(srs_text)


if __name__ == "__main__":
    # Quick test run example
    sample_srs = (
        "The system shall allow users to login with username and password. "
        "If login fails 3 times, account is locked. "
        "Password reset shall be possible via email verification."
    )
    generator = AdvancedFreeModelGenerator()
    cases = generator.generate_test_cases(sample_srs)
    print("\nGenerated Test Cases:")
    for c in cases:
        print(c)
