from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    pipeline,
    set_seed
)
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv
import re
from typing import List, Dict
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AdvancedFreeModelGenerator:
    def __init__(self):
        self.device = "cpu"  # Force CPU for Railway
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Login to Hugging Face if token available
        if self.hf_token:
            try:
                login(token=self.hf_token)
                logger.info("Successfully logged into Hugging Face")
            except Exception as e:
                logger.warning(f"HF login failed: {e}")
        
        self.models = self._get_model_hierarchy()
        self.current_model = None
        self.current_tokenizer = None
        self.model_config = None
        self._load_best_available_model()
    
    def _get_model_hierarchy(self) -> List[Dict]:
        """Define model hierarchy optimized for Railway deployment"""
        return [
            {
                "name": "distilgpt2",
                "type": "causal", 
                "description": "Lightweight GPT - Fast and reliable",
                "max_length": 512,
                "requires_gpu": False,
                "memory_efficient": True
            },
            {
                "name": "google/flan-t5-small",
                "type": "seq2seq",
                "description": "Small instruction model - Memory efficient",
                "max_length": 512,
                "requires_gpu": False,
                "memory_efficient": True
            },
            {
                "name": "facebook/blenderbot-small-90M",
                "type": "seq2seq", 
                "description": "Very small conversational model",
                "max_length": 512,
                "requires_gpu": False,
                "memory_efficient": True
            }
        ]
    
    def _load_best_available_model(self):
        """Load the best available model for Railway environment"""
        logger.info("Starting model loading process...")
        
        for model_config in self.models:
            try:
                logger.info(f"Attempting to load: {model_config['name']}")
                
                if model_config["type"] == "causal":
                    success = self._load_causal_model(model_config)
                else:
                    success = self._load_seq2seq_model(model_config)
                
                if success:
                    logger.info(f"✅ Successfully loaded: {model_config['description']}")
                    return
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {model_config['name']}: {str(e)}")
                continue
        
        logger.warning("⚠️ All models failed to load, using template-based fallback")
        self.model_config = {
            "name": "template-fallback",
            "type": "template",
            "description": "Template-based generation"
        }
    
    def _load_causal_model(self, config: Dict) -> bool:
        """Load causal language model with Railway optimizations"""
        try:
            # Load tokenizer first
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                config["name"],
                token=self.hf_token,
                padding_side="left",
                cache_dir="/tmp/model_cache"  # Use tmp for Railway
            )
            
            # Add pad token if missing
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            # Load model with memory optimizations
            self.current_model = AutoModelForCausalLM.from_pretrained(
                config["name"],
                token=self.hf_token,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None,
                low_cpu_mem_usage=True,
                cache_dir="/tmp/model_cache"
            )
            
            self.current_model = self.current_model.to(self.device)
            self.current_model.eval()  # Set to evaluation mode
            
            self.model_config = config
            logger.info(f"Successfully loaded causal model: {config['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Causal model loading failed: {e}")
            return False
    
    def _load_seq2seq_model(self, config: Dict) -> bool:
        """Load sequence-to-sequence model with Railway optimizations"""
        try:
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                config["name"],
                token=self.hf_token,
                cache_dir="/tmp/model_cache"
            )
            
            # Load model
            self.current_model = AutoModelForSeq2SeqLM.from_pretrained(
                config["name"],
                token=self.hf_token,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True,
                cache_dir="/tmp/model_cache"
            )
            
            self.current_model = self.current_model.to(self.device)
            self.current_model.eval()
            
            self.model_config = config
            logger.info(f"Successfully loaded seq2seq model: {config['name']}")
            return True
            
        except Exception as e:
            logger.error(f"Seq2seq model loading failed: {e}")
            return False
    
    def generate_test_cases(self, srs_text: str) -> List[str]:
        """Generate test cases with proper error handling"""
        try:
            if not self.current_model or self.model_config["type"] == "template":
                logger.info("Using template-based generation")
                return self._generate_template_based(srs_text)
            
            logger.info(f"Generating with {self.model_config['name']}")
            
            # Create optimized prompt based on model type
            if self.model_config["type"] == "causal":
                return self._generate_causal(srs_text)
            else:
                return self._generate_seq2seq(srs_text)
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._generate_template_based(srs_text)
    
    def _generate_causal(self, srs_text: str) -> List[str]:
        """Generate using causal language model with timeout protection"""
        try:
            # Shorter, more focused prompt
            prompt = f"""Generate test cases for software requirement:

{srs_text[:300]}

Test Cases:
TC-001:"""
            
            # Tokenize with strict limits
            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=300,  # Reduced for Railway
                padding=True
            ).to(self.device)
            
            # Conservative generation parameters
            set_seed(42)
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=200,  # Reduced
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.current_tokenizer.pad_token_id,
                    eos_token_id=self.current_tokenizer.eos_token_id,
                    early_stopping=True,
                    timeout=30  # 30 second timeout
                )
            
            # Decode and parse
            generated_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_test_cases(generated_text)
            
        except Exception as e:
            logger.error(f"Causal generation failed: {e}")
            return self._generate_template_based(srs_text)
    
    def _generate_seq2seq(self, srs_text: str) -> List[str]:
        """Generate using sequence-to-sequence model with timeout protection"""
        try:
            # Focused prompt for seq2seq
            prompt = f"""Create test cases for: {srs_text[:200]}

Format: TC-XXX: [objective] - [steps] - [expected result]"""
            
            # Tokenize with limits
            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            ).to(self.device)
            
            # Generate with timeout
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.1,
                    early_stopping=True,
                    timeout=30
                )
            
            # Decode
            generated_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._extract_test_cases(generated_text)
            
        except Exception as e:
            logger.error(f"Seq2seq generation failed: {e}")
            return self._generate_template_based(srs_text)
    
    def _extract_test_cases(self, text: str) -> List[str]:
        """Extract and clean test cases from generated text"""
        # Multiple regex patterns to catch different formats
        patterns = [
            r'TC-\d{3}:.*?(?=TC-\d{3}:|$)',
            r'TC\d{3}:.*?(?=TC\d{3}:|$)',
            r'\d+\.\s*TC-\d{3}:.*?(?=\d+\.\s*TC-\d{3}:|$)'
        ]
        
        test_cases = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                test_cases.extend(matches)
                break
        
        # Clean and validate
        cleaned_cases = []
        seen = set()
        
        for i, tc in enumerate(test_cases[:8], 1):
            cleaned = self._clean_test_case(tc.strip())
            
            if (len(cleaned) > 30 and 
                cleaned not in seen and 
                self._contains_test_structure(cleaned)):
                
                # Ensure proper numbering
                formatted = re.sub(r'TC-?\d{3}:?', f'TC-{i:03}:', cleaned)
                if not formatted.startswith('TC-'):
                    formatted = f'TC-{i:03}: {formatted}'
                
                cleaned_cases.append(formatted)
                seen.add(cleaned)
        
        # Supplement with templates if needed
        if len(cleaned_cases) < 4:
            template_cases = self._get_smart_templates(text)
            for tc in template_cases:
                if len(cleaned_cases) >= 6:
                    break
                if tc not in seen:
                    cleaned_cases.append(tc)
        
        return cleaned_cases[:6] if cleaned_cases else self._generate_template_based("")
    
    def _clean_test_case(self, test_case: str) -> str:
        """Clean individual test case"""
        # Remove extra whitespace and line breaks
        cleaned = ' '.join(test_case.split())
        
        # Remove numbering prefixes like "1. TC-001:"
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        
        # Ensure proper ending
        if not cleaned.endswith('.') and not cleaned.endswith('!'):
            cleaned += '.'
        
        return cleaned
    
    def _contains_test_structure(self, text: str) -> bool:
        """Check if text has proper test case structure"""
        has_action_words = any(word in text.lower() for word in 
                              ['verify', 'test', 'check', 'validate', 'ensure', 'confirm'])
        has_separators = text.count('-') >= 1 or text.count(',') >= 1
        
        return has_action_words and has_separators and len(text) > 30
    
    def _get_smart_templates(self, context: str) -> List[str]:
        """Generate smart templates based on context"""
        context_lower = context.lower()
        
        templates = []
        
        if any(word in context_lower for word in ['login', 'user', 'password', 'auth']):
            templates.extend([
                "TC-005: Verify login with valid credentials - Enter correct username and password, click login - User successfully authenticated and redirected to dashboard.",
                "TC-006: Test login with invalid credentials - Enter wrong password - Error message displayed, access denied."
            ])
        
        if any(word in context_lower for word in ['form', 'input', 'field', 'data']):
            templates.extend([
                "TC-007: Validate required field validation - Leave mandatory fields empty, submit form - Error messages displayed for required fields.",
                "TC-008: Test form submission with valid data - Fill all fields correctly, submit - Form submitted successfully, confirmation displayed."
            ])
        
        # Default templates
        if not templates:
            templates = [
                "TC-005: Test basic functionality - Execute main feature workflow - Feature works as expected without errors.",
                "TC-006: Validate error handling - Trigger error condition - Appropriate error message displayed, system remains stable.",
                "TC-007: Test user interface elements - Interact with UI components - All elements respond correctly and display properly.",
                "TC-008: Verify system performance - Execute normal operations - System responds within acceptable time limits."
            ]
        
        return templates[:4]
    
    def _generate_template_based(self, srs_text: str) -> List[str]:
        """High-quality template-based generation as fallback"""
        logger.info("Using intelligent template-based generation...")
        
        # Analyze SRS content for keywords
        content = srs_text.lower() if srs_text else ""
        
        # Feature detection
        feature_keywords = {
            'authentication': ['login', 'signin', 'password', 'credential', 'auth', 'user'],
            'registration': ['register', 'signup', 'account', 'user creation', 'sign up'],
            'ecommerce': ['cart', 'product', 'order', 'purchase', 'shop', 'buy', 'payment'],
            'form': ['form', 'input', 'field', 'submit', 'validation', 'data entry'],
            'api': ['api', 'endpoint', 'request', 'response', 'service', 'rest'],
            'security': ['security', 'encrypt', 'token', 'permission', 'access', 'authorize']
        }
        
        detected_features = []
        for feature, keywords in feature_keywords.items():
            if any(keyword in content for keyword in keywords):
                detected_features.append(feature)
        
        # Generate targeted test cases
        test_cases = []
        
        # Feature-specific templates
        if 'authentication' in detected_features:
            test_cases.extend([
                "TC-001: Verify successful login with valid credentials - Enter correct username and password, click Login - User authenticated and redirected to main dashboard.",
                "TC-002: Test login failure with invalid password - Enter valid username with wrong password - Error message displayed, access denied."
            ])
        
        if 'registration' in detected_features:
            test_cases.extend([
                "TC-003: Complete user registration with valid data - Fill all required fields, submit registration - User account created successfully, confirmation displayed.",
                "TC-004: Test email validation during registration - Enter invalid email format - Validation error displayed, registration prevented."
            ])
        
        if 'form' in detected_features:
            test_cases.extend([
                "TC-005: Submit form with all required fields - Fill mandatory fields with valid data - Form submitted successfully, data saved.",
                "TC-006: Validate required field error handling - Leave required fields empty, submit - Validation errors displayed for empty fields."
            ])
        
        # Add general test cases if needed
        general_templates = [
            "TC-001: Test basic system functionality - Execute primary user workflow - System performs expected operations correctly.",
            "TC-002: Validate input data handling - Enter valid data in input fields - Data accepted and processed correctly.",
            "TC-003: Test error condition handling - Trigger system error scenario - Appropriate error message displayed, system remains stable.",
            "TC-004: Verify user interface responsiveness - Interact with UI elements - All interface elements respond appropriately.",
            "TC-005: Test cross-browser compatibility - Access system using different browsers - Functionality works consistently across browsers.",
            "TC-006: Validate system security measures - Attempt unauthorized access - Security controls prevent unauthorized operations."
        ]
        
        # Fill remaining slots
        case_num = len(test_cases) + 1
        while len(test_cases) < 6:
            if case_num - 1 < len(general_templates):
                test_cases.append(general_templates[case_num - 1])
            else:
                break
            case_num += 1
        
        return test_cases[:6]
    
    def get_model_info(self) -> Dict:
        """Get information about currently loaded model"""
        if self.current_model and self.model_config:
            return {
                "model_name": self.model_config["name"],
                "description": self.model_config["description"],
                "type": self.model_config["type"],
                "device": str(self.device),
                "status": "loaded"
            }
        else:
            return {
                "model_name": "Template-based fallback",
                "description": "Intelligent template generation",
                "type": "rule-based",
                "device": "CPU",
                "status": "fallback"
            }

# Updated main function with better error handling
def generate_test_cases(srs_text: str) -> List[str]:
    """Enhanced test case generation with Railway deployment support"""
    try:
        generator = AdvancedFreeModelGenerator()
        
        # Log model info
        model_info = generator.get_model_info()
        logger.info(f"Using: {model_info['model_name']} ({model_info['status']})")
        
        # Generate test cases
        test_cases = generator.generate_test_cases(srs_text)
        
        # Ensure we always return something
        if not test_cases:
            logger.warning("No test cases generated, using fallback")
            return generator._generate_template_based(srs_text)
        
        return test_cases
        
    except Exception as e:
        logger.error(f"Critical error in generate_test_cases: {e}")
        # Ultimate fallback
        return [
            "TC-001: Test basic system functionality - Execute primary workflow - System operates as expected.",
            "TC-002: Validate input handling - Enter test data - Data processed correctly.",
            "TC-003: Test error conditions - Trigger error scenario - Error handled gracefully.",
            "TC-004: Verify user interface - Interact with UI elements - Interface responds properly.",
            "TC-005: Test system integration - Execute integrated workflow - All components work together.",
            "TC-006: Validate system performance - Execute under normal load - System maintains acceptable performance."
        ]