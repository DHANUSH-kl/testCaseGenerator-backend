from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    set_seed
)
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv
import re
from typing import List, Dict, Optional
import warnings
import logging
import gc
import psutil
from functools import lru_cache

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RailwayOptimizedGenerator:
    """Memory-optimized generator for Railway deployment (512MB RAM)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure single model loading"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = "cpu"
        self.hf_token = os.getenv("HF_TOKEN")
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.memory_threshold = 400 * 1024 * 1024  # 400MB threshold
        
        # Login to HF if token available
        if self.hf_token:
            try:
                login(token=self.hf_token, add_to_git_credential=False)
                logger.info("✅ HuggingFace login successful")
            except Exception as e:
                logger.warning(f"⚠️ HF login failed: {e}")
        
        # Load single optimal model
        self._load_optimal_model()
        self._initialized = True
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage in MB"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Current memory usage: {memory_mb:.1f}MB")
            return memory_mb
        except:
            return 0
    
    def _load_optimal_model(self):
        """Load single lightweight model optimized for Railway"""
        
        # Check initial memory
        initial_memory = self._check_memory_usage()
        logger.info(f"Starting model loading with {initial_memory:.1f}MB used")
        
        # Use only the most memory-efficient model
        model_name = "distilgpt2"
        
        try:
            logger.info(f"Loading {model_name}...")
            
            # Load tokenizer with minimal cache
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.hf_token,
                cache_dir=None,  # No caching to save memory
                local_files_only=False,
                use_fast=True,  # Use fast tokenizer
                padding_side="left"
            )
            
            # Add pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Force garbage collection before model loading
            gc.collect()
            
            # Load model with aggressive memory optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=self.hf_token,
                torch_dtype=torch.float16,  # Use half precision
                device_map=None,
                low_cpu_mem_usage=True,
                cache_dir=None,  # No caching
                local_files_only=False,
                use_safetensors=True if hasattr(AutoModelForCausalLM, 'use_safetensors') else False
            )
            
            # Move to CPU and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Enable memory efficient attention if available
            if hasattr(self.model.config, 'use_memory_efficient_attention'):
                self.model.config.use_memory_efficient_attention = True
            
            self.model_name = model_name
            
            # Final memory check
            final_memory = self._check_memory_usage()
            logger.info(f"✅ Model loaded successfully! Memory usage: {final_memory:.1f}MB")
            
            # Force cleanup
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            self.model = None
            self.tokenizer = None
            self.model_name = "template-fallback"
            
            # Clean up on failure
            gc.collect()
    
    @lru_cache(maxsize=32)  # Cache common SRS patterns
    def _get_feature_keywords(self, srs_lower: str) -> tuple:
        """Cached feature detection to avoid repeated processing"""
        features = []
        
        feature_patterns = {
            'auth': ['login', 'signin', 'password', 'credential', 'authenticate'],
            'registration': ['register', 'signup', 'account creation', 'sign up'],
            'ecommerce': ['cart', 'product', 'order', 'purchase', 'payment'],
            'form': ['form', 'input', 'field', 'submit', 'validation'],
            'api': ['api', 'endpoint', 'request', 'response', 'service'],
            'security': ['security', 'encrypt', 'token', 'permission', 'access']
        }
        
        for feature, keywords in feature_patterns.items():
            if any(keyword in srs_lower for keyword in keywords):
                features.append(feature)
        
        return tuple(features)  # Return tuple for hashing
    
    def generate_test_cases(self, srs_text: str) -> List[str]:
        """Generate test cases with memory monitoring"""
        
        # Check memory before generation
        current_memory = self._check_memory_usage()
        if current_memory > self.memory_threshold:
            logger.warning(f"⚠️ High memory usage ({current_memory:.1f}MB), using templates")
            return self._generate_intelligent_templates(srs_text)
        
        # Try AI generation if model loaded
        if self.model and self.tokenizer:
            try:
                return self._generate_with_model(srs_text)
            except Exception as e:
                logger.error(f"Model generation failed: {e}")
                # Clean up and fallback
                gc.collect()
                return self._generate_intelligent_templates(srs_text)
        else:
            return self._generate_intelligent_templates(srs_text)
    
    def _generate_with_model(self, srs_text: str) -> List[str]:
        """Generate using the loaded model with strict memory limits"""
        
        # Create concise prompt
        prompt = f"""Create test cases for software requirement:

{srs_text[:200]}

Test cases:
TC-001:"""
        
        try:
            # Tokenize with strict limits
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=200,  # Very conservative
                padding=False,  # No padding to save memory
                return_attention_mask=False  # Don't need attention mask
            ).to(self.device)
            
            # Conservative generation
            set_seed(42)
            with torch.no_grad():
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,  # Conservative limit
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    use_cache=False  # Don't cache to save memory
                )
            
            # Decode and clean up immediately
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            del outputs, inputs  # Immediate cleanup
            gc.collect()
            
            # Extract test cases
            test_cases = self._extract_test_cases(generated_text)
            
            # If AI generation insufficient, supplement with templates
            if len(test_cases) < 4:
                template_cases = self._generate_intelligent_templates(srs_text)
                test_cases.extend(template_cases[len(test_cases):])
            
            return test_cases[:6]
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            gc.collect()
            return self._generate_intelligent_templates(srs_text)
    
    def _extract_test_cases(self, text: str) -> List[str]:
        """Extract test cases with improved parsing"""
        
        # Multiple regex patterns
        patterns = [
            r'TC-\d{3}:[^T]*?(?=TC-\d{3}:|$)',
            r'TC\d{3}:[^T]*?(?=TC\d{3}:|$)',
            r'\d+\.\s*TC-?\d{3}:[^0-9]*?(?=\d+\.\s*TC-?\d{3}:|$)'
        ]
        
        test_cases = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                test_cases.extend(matches)
                break
        
        # Clean and format
        cleaned_cases = []
        for i, tc in enumerate(test_cases[:6], 1):
            cleaned = self._clean_test_case(tc.strip())
            
            if len(cleaned) > 25 and self._is_valid_test_case(cleaned):
                # Ensure proper numbering
                formatted = re.sub(r'TC-?\d{3}:?', f'TC-{i:03}:', cleaned)
                if not formatted.startswith('TC-'):
                    formatted = f'TC-{i:03}: {formatted}'
                
                cleaned_cases.append(formatted)
        
        return cleaned_cases
    
    def _clean_test_case(self, test_case: str) -> str:
        """Clean individual test case"""
        # Remove extra whitespace
        cleaned = ' '.join(test_case.split())
        
        # Remove numbering prefixes
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        
        # Ensure proper ending
        if not cleaned.endswith('.') and len(cleaned) > 0:
            cleaned += '.'
        
        return cleaned
    
    def _is_valid_test_case(self, text: str) -> bool:
        """Validate test case quality"""
        text_lower = text.lower()
        
        # Must have action words
        action_words = ['verify', 'test', 'check', 'validate', 'ensure', 'confirm', 'execute']
        has_action = any(word in text_lower for word in action_words)
        
        # Must have structure indicators
        has_structure = '-' in text or ',' in text or 'and' in text_lower
        
        # Must be substantial
        is_substantial = len(text) > 25
        
        return has_action and has_structure and is_substantial
    
    def _generate_intelligent_templates(self, srs_text: str) -> List[str]:
        """High-quality template generation based on SRS analysis"""
        
        logger.info("Using intelligent template generation")
        
        # Analyze SRS content
        content_lower = srs_text.lower() if srs_text else ""
        detected_features = self._get_feature_keywords(content_lower)
        
        test_cases = []
        
        # Feature-specific templates
        templates_map = {
            'auth': [
                "TC-001: Verify successful user authentication - Enter valid credentials and submit login form - User successfully logged in and redirected to dashboard.",
                "TC-002: Test authentication with invalid credentials - Enter incorrect password and submit - Authentication fails with appropriate error message."
            ],
            'registration': [
                "TC-003: Complete user registration workflow - Fill all required fields with valid data and submit - User account created successfully with confirmation message.",
                "TC-004: Validate registration field requirements - Submit form with missing required fields - Validation errors displayed for empty mandatory fields."
            ],
            'form': [
                "TC-005: Submit form with valid input data - Enter correct data in all fields and submit - Form processed successfully with confirmation.",
                "TC-006: Test form validation rules - Enter invalid data formats and submit - Appropriate validation errors displayed for invalid inputs."
            ],
            'ecommerce': [
                "TC-007: Add product to shopping cart - Select product and click add to cart - Product successfully added with correct quantity and price.",
                "TC-008: Complete purchase transaction - Proceed through checkout with valid payment info - Order processed successfully with confirmation."
            ],
            'api': [
                "TC-009: Test API endpoint with valid request - Send properly formatted request to endpoint - API returns expected response with correct status code.",
                "TC-010: Validate API error handling - Send malformed request to endpoint - API returns appropriate error response with error details."
            ],
            'security': [
                "TC-011: Verify access control restrictions - Attempt to access restricted resource without permission - Access denied with appropriate security message.",
                "TC-012: Test data encryption functionality - Submit sensitive data through secure form - Data transmitted and stored with proper encryption."
            ]
        }
        
        # Add feature-specific test cases
        for feature in detected_features:
            if feature in templates_map:
                test_cases.extend(templates_map[feature])
                if len(test_cases) >= 6:
                    break
        
        # Add general templates if needed
        if len(test_cases) < 4:
            general_templates = [
                "TC-001: Test primary system functionality - Execute main user workflow end-to-end - All system features work correctly without errors.",
                "TC-002: Validate user interface responsiveness - Interact with all UI elements and controls - Interface responds appropriately to user actions.",
                "TC-003: Test error handling and recovery - Trigger system error conditions - Errors handled gracefully with informative messages.",
                "TC-004: Verify data integrity and validation - Input various data types and formats - System validates and processes data correctly.",
                "TC-005: Test system performance under normal load - Execute typical user operations - System maintains acceptable response times.",
                "TC-006: Validate cross-browser compatibility - Access system using different web browsers - Functionality works consistently across browsers."
            ]
            
            # Fill remaining slots
            remaining_slots = 6 - len(test_cases)
            test_cases.extend(general_templates[:remaining_slots])
        
        return test_cases[:6]
    
    def get_model_info(self) -> Dict:
        """Get current model information"""
        return {
            "model_name": self.model_name or "template-fallback",
            "status": "loaded" if self.model else "template-mode",
            "memory_usage": f"{self._check_memory_usage():.1f}MB",
            "device": self.device,
            "optimization": "railway-512mb"
        }
    
    def cleanup(self):
        """Manual cleanup method"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        logger.info("Model cleanup completed")

# Global instance
_generator_instance = None

def get_generator() -> RailwayOptimizedGenerator:
    """Get singleton generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = RailwayOptimizedGenerator()
    return _generator_instance

def generate_test_cases(srs_text: str) -> List[str]:
    """Main function for test case generation"""
    try:
        generator = get_generator()
        
        # Log model status
        model_info = generator.get_model_info()
        logger.info(f"Using: {model_info['model_name']} | Memory: {model_info['memory_usage']}")
        
        # Generate test cases
        test_cases = generator.generate_test_cases(srs_text)
        
        # Ensure we always return valid test cases
        if not test_cases or len(test_cases) == 0:
            logger.warning("No test cases generated, using fallback")
            return generator._generate_intelligent_templates(srs_text)
        
        logger.info(f"✅ Generated {len(test_cases)} test cases successfully")
        return test_cases
        
    except Exception as e:
        logger.error(f"Critical error in generate_test_cases: {e}")
        
        # Ultimate fallback with quality templates
        return [
            "TC-001: Test core system functionality - Execute primary business workflow - System performs all required operations correctly.",
            "TC-002: Validate input data processing - Enter test data through user interface - Data accepted, validated, and processed accurately.",
            "TC-003: Test error condition handling - Trigger known error scenarios - System handles errors gracefully with appropriate user feedback.",
            "TC-004: Verify user interface functionality - Navigate through all interface elements - All UI components respond correctly to user interactions.",
            "TC-005: Test system integration points - Execute workflows involving multiple components - All integrated systems work together seamlessly.",
            "TC-006: Validate system security measures - Attempt various access scenarios - Security controls function properly to protect system resources."
        ]

# Memory monitoring decorator for Flask routes
def monitor_memory(func):
    """Decorator to monitor memory usage in Flask routes"""
    def wrapper(*args, **kwargs):
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Route started with {initial_memory:.1f}MB")
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Route completed with {final_memory:.1f}MB")
            
            # Force cleanup if memory usage is high
            if final_memory > 450:  # 450MB threshold
                gc.collect()
                logger.info("Forced garbage collection due to high memory usage")
    
    wrapper.__name__ = func.__name__
    return wrapper