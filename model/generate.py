import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import psutil
import re
import gc

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# List of memory-optimized models
MEMORY_OPTIMIZED_MODELS = [
    "gpt2",  # ~500MB
    "distilgpt2",  # ~250MB
    "microsoft/DialoGPT-small",  # ~250MB
    "huggingface/CodeBERTa-small-v1",  # Code tasks
]

# Singleton state
_generator_instance = None

def get_optimal_model_for_memory():
    """Select the best model based on available memory."""
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    logger.info(f"Available memory: {available_memory:.1f}MB")

    if available_memory < 300:
        return None  # Use template fallback
    elif available_memory < 600:
        return "microsoft/DialoGPT-small"
    else:
        return "distilgpt2"

def load_model_with_memory_optimization(model_name):
    """Load model with low memory settings."""
    try:
        logger.info(f"Loading {model_name} with memory optimizations...")

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            use_cache=False,
        )

        model.eval()
        model.gradient_checkpointing_enable()
        logger.info(f"✅ Model {model_name} loaded successfully")
        return tokenizer, model

    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {e}")
        return None, None

def extract_keywords(text):
    common_keywords = [
        'login', 'authentication', 'user', 'password', 'database', 'data',
        'interface', 'api', 'function', 'feature', 'requirement', 'system',
        'input', 'output', 'validation', 'error', 'security', 'performance'
    ]
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word in common_keywords]

def generate_template_based_test_cases(srs_text):
    keywords = extract_keywords(srs_text)
    test_cases = []

    if any(word in keywords for word in ['login', 'authentication', 'user', 'password']):
        test_cases.extend([
            {
                "id": "TC_001",
                "title": "Valid Login Test",
                "description": "Test login with valid credentials",
                "steps": ["Enter valid username", "Enter valid password", "Click login"],
                "expected": "User should be logged in successfully"
            },
            {
                "id": "TC_002",
                "title": "Invalid Login Test",
                "description": "Test login with invalid credentials",
                "steps": ["Enter invalid username", "Enter invalid password", "Click login"],
                "expected": "Error message should be displayed"
            }
        ])

    if any(word in keywords for word in ['database', 'data', 'store', 'save']):
        test_cases.append({
            "id": "TC_003",
            "title": "Data Storage Test",
            "description": "Test data storage functionality",
            "steps": ["Enter data", "Save data", "Verify storage"],
            "expected": "Data should be stored correctly"
        })

    if not test_cases:
        test_cases = [
            {
                "id": "TC_001",
                "title": "Basic Functionality Test",
                "description": "Test basic system functionality",
                "steps": ["Access the system", "Perform basic operations", "Verify results"],
                "expected": "System should work as expected"
            }
        ]

    return test_cases

def parse_generated_test_cases(generated_text):
    lines = generated_text.split('\n')
    test_cases = []
    current_case = {}
    case_counter = 1

    for line in lines:
        line = line.strip()
        if line.startswith(('1.', '2.', '3.', 'TC', 'Test')):
            if current_case:
                test_cases.append(current_case)
            current_case = {
                "id": f"TC_{case_counter:03d}",
                "title": line,
                "description": line,
                "steps": ["Execute the test"],
                "expected": "Test should pass"
            }
            case_counter += 1

    if current_case:
        test_cases.append(current_case)

    if not test_cases:
        return [{
            "id": "TC_001",
            "title": "Generated Test Case",
            "description": "Auto-generated test case based on requirements",
            "steps": ["Review requirements", "Execute test", "Verify results"],
            "expected": "Requirements should be met"
        }]

    return test_cases

def generate_with_ai_model(srs_text, tokenizer, model):
    max_input_length = 200
    if len(srs_text) > max_input_length:
        srs_text = srs_text[:max_input_length]

    prompt = f"""Generate test cases for this software requirement:
{srs_text}

Test Cases:
1."""

    try:
        inputs = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=150,
            truncation=True
        )

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        del inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return parse_generated_test_cases(generated_text)

    except Exception as e:
        logger.error(f"❌ AI generation failed: {e}")
        raise

def generate_with_fallback(srs_text):
    model_name = get_optimal_model_for_memory()

    if model_name:
        tokenizer, model = load_model_with_memory_optimization(model_name)
        if tokenizer and model:
            try:
                return generate_with_ai_model(srs_text, tokenizer, model), model_name
            except Exception as e:
                logger.warning(f"AI generation failed: {e}, falling back to templates")

    logger.info("⚠️ Using fallback template-based generation")
    return generate_template_based_test_cases(srs_text), "Template-Based Generator"

# ✅ Function exposed to app.py
def generate_test_cases(srs_text):
    return generate_with_fallback(srs_text)[0]

def get_generator():
    global _generator_instance
    if _generator_instance is None:
        class Generator:
            def __init__(self):
                self.model_name = get_optimal_model_for_memory()
                self.tokenizer = None
                self.model = None
                if self.model_name:
                    self.tokenizer, self.model = load_model_with_memory_optimization(self.model_name)

            def get_model_info(self):
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                return {
                    "model_name": self.model_name if self.model_name else "Template-Based Generator",
                    "status": "loaded" if self.model else "template_mode",
                    "memory_usage": f"{mem:.1f}MB",
                    "optimization": "low_memory"
                }

        _generator_instance = Generator()

    return _generator_instance

def monitor_memory():
    mem = psutil.Process().memory_info().rss / 1024 / 1024
    logger.info(f"Memory usage: {mem:.1f}MB")
    if mem > 450:
        gc.collect()
        logger.info("Memory cleanup triggered")
