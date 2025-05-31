# Add this to your model/generate.py or create a new config file

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

# Memory-optimized model configuration
MEMORY_OPTIMIZED_MODELS = [
    "gpt2",  # Smallest GPT-2 model (~500MB)
    "distilgpt2",  # Your current choice (~250MB)
    "microsoft/DialoGPT-small",  # Very small conversational model
    "huggingface/CodeBERTa-small-v1",  # For code-related tasks
]

def get_optimal_model_for_memory():
    """Select the best model based on available memory"""
    import psutil
    
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    logger.info(f"Available memory: {available_memory:.1f}MB")
    
    if available_memory < 300:
        # Use template-based generation for very low memory
        return None
    elif available_memory < 600:
        # Use smallest possible model
        return "microsoft/DialoGPT-small"
    else:
        # Use distilgpt2
        return "distilgpt2"

def load_model_with_memory_optimization(model_name):
    """Load model with aggressive memory optimization"""
    try:
        logger.info(f"Loading {model_name} with memory optimization...")
        
        # Load with reduced precision and CPU-only
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left',
            use_fast=True,  # Use fast tokenizer for memory efficiency
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with memory optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            device_map="cpu",  # Force CPU usage
            low_cpu_mem_usage=True,  # Enable memory optimization
            use_cache=False,  # Disable KV cache to save memory
        )
        
        # Set model to evaluation mode
        model.eval()
        
        # Reduce model size further
        model.gradient_checkpointing_enable()
        
        logger.info(f"✅ Model {model_name} loaded successfully with optimizations")
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name}: {e}")
        return None, None

def generate_with_fallback(srs_text):
    """Generate test cases with fallback to template-based generation"""
    
    # Try to use AI model first
    model_name = get_optimal_model_for_memory()
    
    if model_name:
        tokenizer, model = load_model_with_memory_optimization(model_name)
        
        if tokenizer and model:
            try:
                return generate_with_ai_model(srs_text, tokenizer, model)
            except Exception as e:
                logger.warning(f"AI generation failed: {e}, falling back to templates")
    
    # Fallback to template-based generation
    logger.info("Using template-based test case generation")
    return generate_template_based_test_cases(srs_text)

def generate_with_ai_model(srs_text, tokenizer, model):
    """Generate using AI model with memory constraints"""
    
    # Truncate input to prevent memory issues
    max_input_length = 200  # Very conservative
    if len(srs_text) > max_input_length:
        srs_text = srs_text[:max_input_length]
    
    prompt = f"""Generate test cases for this software requirement:
{srs_text}

Test Cases:
1."""
    
    try:
        # Tokenize with length limits
        inputs = tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            max_length=150,  # Very short input
            truncation=True
        )
        
        # Generate with strict memory limits
        with torch.no_grad():  # Disable gradients
            outputs = model.generate(
                inputs,
                max_new_tokens=100,  # Very short output
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Disable cache
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract test cases from generated text
        test_cases = parse_generated_test_cases(generated_text)
        
        # Clean up immediately
        del inputs, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return test_cases
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

def generate_template_based_test_cases(srs_text):
    """Fallback template-based test case generation"""
    
    # Simple keyword-based test case generation
    keywords = extract_keywords(srs_text.lower())
    
    test_cases = []
    
    # Basic functional test cases
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
    
    # Add generic test cases if none found
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

def extract_keywords(text):
    """Extract relevant keywords from SRS text"""
    import re
    
    # Common software requirement keywords
    common_keywords = [
        'login', 'authentication', 'user', 'password', 'database', 'data',
        'interface', 'api', 'function', 'feature', 'requirement', 'system',
        'input', 'output', 'validation', 'error', 'security', 'performance'
    ]
    
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word in common_keywords]

def parse_generated_test_cases(generated_text):
    """Parse test cases from generated text"""
    import re
    
    # Simple parsing logic
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
    
    return test_cases if test_cases else [{
        "id": "TC_001",
        "title": "Generated Test Case",
        "description": "Auto-generated test case based on requirements",
        "steps": ["Review requirements", "Execute test", "Verify results"],
        "expected": "Requirements should be met"
    }]