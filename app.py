from flask import Flask, request, jsonify
from flask_cors import CORS
from model.generate import generate_test_cases, get_generator, monitor_memory
import os
import logging
import gc
import psutil
from functools import wraps
import time
import threading

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration for Railway
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Reduce response size

# Thread-safe initialization
_init_lock = threading.Lock()
_initialized = False

def init_model():
    """Initialize model on startup"""
    try:
        # Skip AI model loading in low memory environments
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > 200 or os.environ.get('RAILWAY_ENVIRONMENT'):
            logger.info("⚠️ Skipping AI model loading due to memory constraints")
            logger.info("🔧 Using template-based generation mode")
            return True
            
        logger.info("🚀 Initializing AI model...")
        generator = get_generator()
        model_info = generator.get_model_info()
        logger.info(f"✅ Model initialized: {model_info['model_name']} | Memory: {model_info['memory_usage']}")
        return True
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.info("🔧 Falling back to template-based generation")
        return False

def health_check():
    """Check system health"""
    try:
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        return {
            "status": "healthy" if memory_mb < 450 else "warning",
            "memory_usage": f"{memory_mb:.1f}MB",
            "memory_limit": "512MB"
        }
    except:
        return {"status": "unknown", "memory_usage": "unavailable"}

# Enhanced memory monitoring with automatic cleanup
def smart_memory_monitor(func):
    """Enhanced memory monitoring with automatic cleanup"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            # Pre-execution memory check
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"🔍 {func.__name__} started | Memory: {initial_memory:.1f}MB")
            
            # Force cleanup if memory is already high
            if initial_memory > 400:
                logger.warning("⚠️ High memory detected, forcing cleanup...")
                gc.collect()
            
            # Execute function
            result = func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__}: {str(e)}")
            # Return error response instead of crashing
            return jsonify({
                "error": "Internal server error occurred",
                "message": "Please try again or contact support"
            }), 500
            
        finally:
            # Post-execution cleanup and logging
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            execution_time = time.time() - start_time
            
            logger.info(f"✅ {func.__name__} completed | Memory: {final_memory:.1f}MB | Time: {execution_time:.2f}s")
            
            # Aggressive cleanup if needed
            if final_memory > 450:
                logger.warning("🧹 High memory usage, forcing aggressive cleanup...")
                gc.collect()
                
                # Double-check memory after cleanup
                post_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"🧹 Post-cleanup memory: {post_cleanup_memory:.1f}MB")
    
    return wrapper

def ensure_initialized():
    """Ensure model is initialized (thread-safe)"""
    global _initialized
    if not _initialized:
        with _init_lock:
            if not _initialized:
                logger.info("🚀 Flask app starting up on Railway...")
                success = init_model()
                if success:
                    logger.info("✅ Startup completed successfully")
                else:
                    logger.warning("⚠️ Model initialization failed, using template mode")
                _initialized = True

@app.before_request
def before_request():
    """Initialize model on first request (Flask 2.2+ compatible)"""
    ensure_initialized()

@app.route('/')
def home():
    """Health check endpoint with system status"""
    health = health_check()
    
    try:
        generator = get_generator()
        model_info = generator.get_model_info()
    except Exception:
        # Fallback when model is not available
        model_info = {
            "model_name": "Template-Based Generator",
            "status": "template_mode",
            "optimization": "memory_safe"
        }
    
    return jsonify({
        "message": "AI Test Case Generator Backend is running",
        "status": health["status"],
        "memory_usage": health["memory_usage"],
        "model": {
            "name": model_info["model_name"],
            "status": model_info["status"],
            "optimization": model_info.get("optimization", "standard")
        },
        "version": "1.0.0-railway-optimized"
    })

@app.route('/health')
def health():
    """Dedicated health check for Railway monitoring"""
    health_status = health_check()
    
    try:
        generator = get_generator()
        model_info = generator.get_model_info()
        model_loaded = model_info["status"] == "loaded"
    except Exception:
        model_loaded = False
    
    return jsonify({
        "status": health_status["status"],
        "memory": health_status["memory_usage"],
        "model_loaded": model_loaded,
        "uptime": "ok"
    })

@app.route('/generate_test_cases', methods=['POST'])
@smart_memory_monitor
def generate():
    """Generate test cases with enhanced error handling"""
    
    # Validate request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    srs_text = data.get('srs', '').strip()
    
    # Validate SRS content
    if not srs_text:
        return jsonify({"error": "No SRS content provided"}), 400
    
    if len(srs_text) > 5000:  # Limit input size for memory safety
        logger.warning(f"SRS text truncated from {len(srs_text)} to 5000 characters")
        srs_text = srs_text[:5000]
    
    try:
        logger.info(f"🎯 Generating test cases for SRS ({len(srs_text)} chars)")
        
        # Generate test cases
        test_cases = generate_test_cases(srs_text)
        
        # Validate output
        if not test_cases or len(test_cases) == 0:
            logger.error("No test cases generated")
            return jsonify({"error": "Failed to generate test cases"}), 500
        
        # Get model info for response
        try:
            generator = get_generator()
            model_info = generator.get_model_info()
            model_used = model_info["model_name"]
            generation_method = model_info["status"]
        except Exception:
            model_used = "Template-Based Generator"
            generation_method = "template_mode"
        
        logger.info(f"✅ Successfully generated {len(test_cases)} test cases")
        
        return jsonify({
            "test_cases": test_cases,
            "count": len(test_cases),
            "model_used": model_used,
            "generation_method": generation_method
        })
        
    except Exception as e:
        logger.error(f"❌ Test case generation failed: {str(e)}")
        return jsonify({
            "error": "Failed to generate test cases",
            "message": "Please try again with different input"
        }), 500

@app.route('/model_info')
def model_info():
    """Get current model information"""
    try:
        generator = get_generator()
        info = generator.get_model_info()
        health = health_check()
        
        return jsonify({
            "model": info,
            "system": health
        })
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": "Unable to get model information"}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Railway deployment configuration
    port = int(os.environ.get("PORT", 5000))
    
    # Production settings for Railway
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"🚀 Starting Flask app on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"🖥️ Environment: {'Railway' if os.environ.get('RAILWAY_ENVIRONMENT') else 'Local'}")
    
    # Initialize model before starting server (if not Railway)
    if not os.environ.get('RAILWAY_ENVIRONMENT'):
        ensure_initialized()
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode,
        threaded=True,  # Enable threading for Railway
        use_reloader=False  # Disable reloader in production
    )