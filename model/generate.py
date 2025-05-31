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
warnings.filterwarnings("ignore")

load_dotenv()

class AdvancedFreeModelGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv("HF_TOKEN")  # Free Hugging Face token
        
        # Login to Hugging Face if token available
        if self.hf_token:
            login(token=self.hf_token)
        
        self.models = self._get_model_hierarchy()
        self.current_model = None
        self.current_tokenizer = None
        self._load_best_available_model()
    
    def _get_model_hierarchy(self) -> List[Dict]:
        """Define model hierarchy from best to fallback"""
        return [
            {
                "name": "microsoft/DialoGPT-large",
                "type": "causal",
                "description": "Large conversational model - Best quality",
                "max_length": 1024,
                "requires_gpu": True
            },
            {
                "name": "facebook/blenderbot-400M-distill",
                "type": "seq2seq", 
                "description": "Conversational AI - Good for structured output",
                "max_length": 512,
                "requires_gpu": False
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "type": "causal",
                "description": "Medium conversational model - Balanced",
                "max_length": 768,
                "requires_gpu": False
            },
            {
                "name": "distilgpt2",
                "type": "causal", 
                "description": "Lightweight GPT - Fast and reliable",
                "max_length": 512,
                "requires_gpu": False
            },
            {
                "name": "google/flan-t5-large",
                "type": "seq2seq",
                "description": "Large instruction-following model",
                "max_length": 512,
                "requires_gpu": True
            },
            {
                "name": "google/flan-t5-base",
                "type": "seq2seq",
                "description": "Base instruction model - Your current model",
                "max_length": 512,
                "requires_gpu": False
            }
        ]
    
    def _load_best_available_model(self):
        """Load the best available model based on system capabilities"""
        has_gpu = torch.cuda.is_available()
        gpu_memory = 0
        
        if has_gpu:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU detected: {gpu_memory:.1f}GB VRAM")
        
        for model_config in self.models:
            try:
                # Skip GPU-required models if no GPU or insufficient memory
                if model_config["requires_gpu"] and (not has_gpu or gpu_memory < 4):
                    continue
                
                print(f"Attempting to load: {model_config['name']}")
                
                if model_config["type"] == "causal":
                    success = self._load_causal_model(model_config)
                else:
                    success = self._load_seq2seq_model(model_config)
                
                if success:
                    print(f"✅ Successfully loaded: {model_config['description']}")
                    return
                    
            except Exception as e:
                print(f"❌ Failed to load {model_config['name']}: {str(e)[:100]}")
                continue
        
        print("⚠️ All models failed to load, using template-based fallback")
    
    def _load_causal_model(self, config: Dict) -> bool:
        """Load causal language model"""
        try:
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                config["name"],
                token=self.hf_token,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            self.current_model = AutoModelForCausalLM.from_pretrained(
                config["name"],
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.current_model = self.current_model.to(self.device)
            
            self.model_config = config
            return True
            
        except Exception as e:
            print(f"Causal model loading failed: {e}")
            return False
    
    def _load_seq2seq_model(self, config: Dict) -> bool:
        """Load sequence-to-sequence model"""
        try:
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                config["name"],
                token=self.hf_token
            )
            
            self.current_model = AutoModelForSeq2SeqLM.from_pretrained(
                config["name"],
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.current_model = self.current_model.to(self.device)
            
            self.model_config = config
            return True
            
        except Exception as e:
            print(f"Seq2seq model loading failed: {e}")
            return False
    
    def generate_test_cases(self, srs_text: str) -> List[str]:
        """Generate test cases with the loaded model"""
        if not self.current_model:
            return self._generate_template_based(srs_text)
        
        try:
            # Create optimized prompt based on model type
            if self.model_config["type"] == "causal":
                return self._generate_causal(srs_text)
            else:
                return self._generate_seq2seq(srs_text)
                
        except Exception as e:
            print(f"Generation failed: {e}")
            return self._generate_template_based(srs_text)
    
    def _generate_causal(self, srs_text: str) -> List[str]:
        """Generate using causal language model"""
        # Optimized prompt for causal models
        prompt = f"""<|system|>You are a senior QA engineer creating test cases.<|endoftext|>
<|user|>Create test cases for this software requirement:

{srs_text[:500]}

Format each test case as:
TC-001: [Test Objective] - [Steps] - [Expected Result]

Test Cases:
TC-001:"""
        
        # Tokenize input
        inputs = self.current_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config["max_length"] - 300,
            padding=True
        ).to(self.device)
        
        # Generate with optimized parameters
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
                early_stopping=True
            )
        
        # Decode and parse
        generated_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_test_cases(generated_text)
    
    def _generate_seq2seq(self, srs_text: str) -> List[str]:
        """Generate using sequence-to-sequence model"""
        # Optimized prompt for seq2seq models
        prompt = f"""Generate 8 professional test cases for this software specification:

{srs_text[:400]}

Requirements:
- Format: TC-XXX: [Clear objective] - [Detailed steps] - [Expected outcome]
- Cover: functionality, validation, error handling, security
- Be specific and actionable

Test cases:"""
        
        # Tokenize
        inputs = self.current_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_config["max_length"],
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.current_model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                early_stopping=True,
                num_beams=2
            )
        
        # Decode
        generated_text = self.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_test_cases(generated_text)
    
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
        
        for i, tc in enumerate(test_cases[:10], 1):
            cleaned = self._clean_test_case(tc.strip())
            
            if (len(cleaned) > 50 and 
                cleaned not in seen and 
                self._contains_test_structure(cleaned)):
                
                # Ensure proper numbering
                formatted = re.sub(r'TC-?\d{3}:?', f'TC-{i:03}:', cleaned)
                if not formatted.startswith('TC-'):
                    formatted = f'TC-{i:03}: {formatted}'
                
                cleaned_cases.append(formatted)
                seen.add(cleaned)
        
        # Supplement with templates if needed
        if len(cleaned_cases) < 6:
            template_cases = self._get_smart_templates(text)
            for tc in template_cases:
                if len(cleaned_cases) >= 8:
                    break
                if tc not in seen:
                    cleaned_cases.append(tc)
        
        return cleaned_cases[:8]
    
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
        # Should contain test objective, steps, and expected result
        has_action_words = any(word in text.lower() for word in 
                              ['verify', 'test', 'check', 'validate', 'ensure', 'confirm'])
        has_separators = text.count('-') >= 1 or text.count(',') >= 2
        
        return has_action_words and has_separators
    
    def _get_smart_templates(self, context: str) -> List[str]:
        """Generate smart templates based on context"""
        context_lower = context.lower()
        
        # Detect domain from context
        templates = []
        
        if any(word in context_lower for word in ['login', 'user', 'password', 'auth']):
            templates.extend([
                "TC-007: Verify login with valid credentials - Enter correct username and password, click login - User successfully authenticated and redirected to dashboard.",
                "TC-008: Test password reset functionality - Click forgot password, enter email, submit - Password reset link sent to registered email address."
            ])
        
        if any(word in context_lower for word in ['form', 'input', 'field', 'data']):
            templates.extend([
                "TC-009: Validate required field validation - Leave mandatory fields empty, submit form - Error messages displayed for all required fields.",
                "TC-010: Test data persistence - Enter data, navigate away, return - Previously entered data is preserved and displayed."
            ])
        
        # Default comprehensive templates
        if not templates:
            templates = [
                "TC-007: Cross-browser compatibility testing - Access application on Chrome, Firefox, Safari - All features work consistently across browsers.",
                "TC-008: Mobile responsiveness validation - View application on mobile devices - Interface adapts properly to different screen sizes.",
                "TC-009: Error handling verification - Trigger system error conditions - Appropriate error messages displayed, system remains stable.",
                "TC-010: Performance under load testing - Simulate multiple concurrent users - System maintains acceptable response times and functionality."
            ]
        
        return templates[:3]
    
    def _generate_template_based(self, srs_text: str) -> List[str]:
        """High-quality template-based generation when models fail"""
        print("Using intelligent template-based generation...")
        
        # Analyze SRS content
        content = srs_text.lower()
        detected_features = []
        
        feature_patterns = {
            'authentication': ['login', 'signin', 'password', 'credential', 'auth'],
            'registration': ['register', 'signup', 'account', 'user creation'],
            'ecommerce': ['cart', 'product', 'order', 'purchase', 'shop', 'buy', 'payment'],
            'form_handling': ['form', 'input', 'field', 'submit', 'validation'],
            'security': ['security', 'encrypt', 'token', 'permission', 'access'],
            'api': ['api', 'endpoint', 'request', 'response', 'service'],
            'database': ['database', 'data', 'store', 'save', 'retrieve']
        }
        
        for feature, keywords in feature_patterns.items():
            if any(keyword in content for keyword in keywords):
                detected_features.append(feature)
        
        # Generate targeted test cases
        test_cases = []
        case_num = 1
        
        # Feature-specific test cases
        feature_templates = {
            'authentication': [
                "TC-{:03d}: Verify successful login with valid credentials - Enter correct username and password, click Login button - User successfully authenticated and redirected to main dashboard with welcome message.",
                "TC-{:03d}: Test login failure with invalid password - Enter valid username with incorrect password - Error message 'Invalid credentials' displayed, login attempt blocked, account not locked.",
                "TC-{:03d}: Validate account lockout mechanism - Enter wrong password 5 consecutive times - Account temporarily locked, appropriate lockout message displayed with unlock timeframe."
            ],
            'registration': [
                "TC-{:03d}: Complete user registration with valid data - Fill all required fields with valid information, submit registration form - User account created successfully, confirmation email sent, user redirected to login page.",
                "TC-{:03d}: Test email validation during registration - Enter invalid email format in email field - Real-time validation error displayed, registration form submission prevented until valid email entered.",
                "TC-{:03d}: Verify password strength requirements - Enter weak password not meeting criteria - Password strength indicator shows requirements, registration blocked until strong password entered."
            ],
            'ecommerce': [
                "TC-{:03d}: Add product to shopping cart successfully - Select product, specify quantity, click Add to Cart - Product added with correct details, cart counter updated, confirmation message displayed.",
                "TC-{:03d}: Complete checkout process with valid payment - Add items to cart, proceed to checkout, enter valid payment details - Order processed successfully, confirmation number generated, email receipt sent.",
                "TC-{:03d}: Test search functionality with product keywords - Enter product name in search field, click search - Relevant products displayed in results with correct filtering and sorting options."
            ],
            'form_handling': [
                "TC-{:03d}: Submit form with all required fields completed - Fill all mandatory fields with valid data, click Submit - Form submitted successfully, confirmation message displayed, data saved to system.",
                "TC-{:03d}: Validate required field error handling - Leave mandatory fields empty, attempt form submission - Validation errors displayed for each empty required field, form submission prevented.",
                "TC-{:03d}: Test form data persistence on page refresh - Enter data in form fields, refresh browser page - Previously entered data maintained in form fields, no data loss occurred."
            ]
        }
        
        # Generate test cases based on detected features
        for feature in detected_features[:3]:  # Max 3 features
            if feature in feature_templates:
                for template in feature_templates[feature][:2]:  # 2 test cases per feature
                    test_cases.append(template.format(case_num))
                    case_num += 1
        
        # Add general test cases to reach minimum count
        general_templates = [
            "TC-{:03d}: Cross-browser compatibility verification - Access application using Chrome, Firefox, and Safari browsers - All functionality works consistently across different browsers without layout issues.",
            "TC-{:03d}: Mobile device responsiveness testing - View application on smartphone and tablet devices - User interface adapts properly to different screen sizes, all features remain accessible.",
            "TC-{:03d}: Session timeout functionality validation - Remain inactive on authenticated page for extended period - User automatically logged out after timeout, redirected to login page with session expired message.",
            "TC-{:03d}: Error handling and recovery testing - Trigger system error by invalid operations - Appropriate error messages displayed, system remains stable, user can recover gracefully.",
            "TC-{:03d}: Data validation and sanitization check - Enter special characters and scripts in input fields - System properly sanitizes input, prevents injection attacks, maintains data integrity.",
            "TC-{:03d}: Performance under normal load conditions - Simulate typical user load on system - Application maintains acceptable response times, all features function normally without degradation."
        ]
        
        # Fill remaining slots with general test cases
        while len(test_cases) < 8:
            remaining_templates = [t for t in general_templates if case_num <= 10]
            if not remaining_templates:
                break
            
            test_cases.append(remaining_templates[0].format(case_num))
            case_num += 1
            general_templates = general_templates[1:]  # Remove used template
        
        return test_cases[:8]
    
    def get_model_info(self) -> Dict:
        """Get information about currently loaded model"""
        if self.current_model:
            return {
                "model_name": self.model_config["name"],
                "description": self.model_config["description"],
                "type": self.model_config["type"],
                "device": str(self.device),
                "max_length": self.model_config["max_length"]
            }
        else:
            return {
                "model_name": "Template-based fallback",
                "description": "Intelligent template generation",
                "type": "rule-based",
                "device": "CPU",
                "max_length": "N/A"
            }

# Updated main function with model info
def generate_test_cases(srs_text: str) -> List[str]:
    """Enhanced free model test case generation with fallback"""
    generator = AdvancedFreeModelGenerator()
    
    # Print model info for debugging
    model_info = generator.get_model_info()
    print(f"Using: {model_info['model_name']} on {model_info['device']}")
    
    return generator.generate_test_cases(srs_text)