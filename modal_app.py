import os
import io
import base64
import random
from datetime import datetime
from typing import Optional, List, Union
from dataclasses import dataclass

import modal
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Modal app setup
app = modal.App("omnigen2-api")

# Simplified image without flash-attn to avoid CUDA compilation issues
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install([
        "torch==2.6.0",
        "torchvision==0.21.0", 
        "timm",
        "einops",
        "accelerate",
        "transformers==4.51.3",
        "diffusers",
        "opencv-python-headless",
        "scipy",
        "matplotlib",
        "Pillow",
        "tqdm",
        "omegaconf",
        "python-dotenv",
        "ninja",
        "fastapi",
        "pydantic",
        "huggingface_hub",
        "jinja2",
        "python-multipart",
    ])
    .add_local_dir(".", "/app", copy=True)
    .workdir("/app")
    .env({"HF_HUB_CACHE": "/models/huggingface"})
)

# Request/Response Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    width: int = Field(default=1024, ge=256, le=1024, description="Output image width")
    height: int = Field(default=1024, ge=256, le=1024, description="Output image height")
    scheduler: str = Field(default="euler", pattern="^(euler|dpmsolver)$", description="Scheduler type")
    num_inference_steps: int = Field(default=50, ge=20, le=100, description="Number of inference steps")
    negative_prompt: str = Field(
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, Different furniture design, altered proportions, changed structure, different style, modified dimensions, new furniture shape",
        description="Negative prompt"
    )
    text_guidance_scale: float = Field(default=5.0, ge=1.0, le=8.0, description="Text guidance scale")
    image_guidance_scale: float = Field(default=2.0, ge=1.0, le=10.0, description="Image guidance scale")
    cfg_range_start: float = Field(default=0.0, ge=0.0, le=1.0, description="CFG range start")
    cfg_range_end: float = Field(default=1.0, ge=0.0, le=1.0, description="CFG range end")
    num_images_per_prompt: int = Field(default=1, ge=1, le=4, description="Number of images per prompt")
    max_input_image_side_length: int = Field(default=2048, ge=256, le=2048, description="Max input image side length")
    max_pixels: int = Field(default=1024*1024, ge=256*256, le=1536*1536, description="Max pixels")
    seed: int = Field(default=-1, ge=-1, le=2147483647, description="Seed for generation (-1 for random)")
    
    # Base64 encoded images (optional)
    image_1: Optional[str] = Field(default=None, description="Base64 encoded first input image")
    image_2: Optional[str] = Field(default=None, description="Base64 encoded second input image")
    image_3: Optional[str] = Field(default=None, description="Base64 encoded third input image")

class GenerationResponse(BaseModel):
    success: bool
    images: List[str] = Field(description="List of base64 encoded generated images")
    seed_used: int = Field(description="Seed that was used for generation")
    error: Optional[str] = None

# GPU volume for model caching - increased timeout
volume = modal.Volume.from_name("omnigen2-models", create_if_missing=True)

# Separate function to download model (runs once)
@app.function(
    image=image,
    volumes={"/models": volume},
    timeout=7200,  # 2 hours for model download
    cpu=4,
    memory=8192,
)
def download_model():
    """Download and cache the OmniGen2 model"""
    import os
    from huggingface_hub import snapshot_download
    
    cache_path = "/models/OmniGen2"
    
    if not os.path.exists(cache_path):
        print("Downloading OmniGen2 model...")
        try:
            # Download model with progress tracking
            snapshot_download(
                repo_id="OmniGen2/OmniGen2",
                cache_dir="/models/huggingface",
                local_dir=cache_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"Model downloaded successfully to {cache_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print(f"Model already cached at {cache_path}")
    
    # Commit changes to volume
    volume.commit()
    return cache_path

@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=300,
    # Reduce memory pressure
    memory=32768,
)
class OmniGen2Model:
    accelerator = None
    pipeline = None
    FlowMatchEulerDiscreteScheduler = None
    DPMSolverMultistepScheduler = None
    create_collage = None
    model_loaded = False
        
    @modal.enter()
    def load_model(self):
        """Load the OmniGen2 model with better error handling"""
        import os
        import time
        from accelerate import Accelerator
        
        print("Starting model loading process...")
        start_time = time.time()
        
        try:
            # Import with error handling
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
            from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
            from omnigen2.utils.img_util import create_collage
            
            print("Imports successful")
            
            # Store classes for later use
            self.FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler
            self.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
            self.create_collage = create_collage
            
            # Initialize accelerator with better settings
            self.accelerator = Accelerator(mixed_precision="bf16")
            weight_dtype = torch.bfloat16
            
            # Check for cached model
            cache_path = "/models/OmniGen2"
            
            if not os.path.exists(cache_path):
                print("Model not found in cache, downloading...")
                # Trigger model download if not cached
                cache_path = download_model.remote()
            
            print(f"Loading model from: {cache_path}")
            
            # Load pipeline with timeout handling
            try:
                self.pipeline = OmniGen2Pipeline.from_pretrained(
                    cache_path,
                    torch_dtype=weight_dtype,
                    trust_remote_code=True,
                )
                print("Pipeline loaded successfully")
                
                # Load transformer
                self.pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
                    cache_path,
                    subfolder="transformer",
                    torch_dtype=weight_dtype,
                )
                print("Transformer loaded successfully")
                
                # Enable optimizations
                self.pipeline.enable_model_cpu_offload()
                
                # Optional: Enable attention slicing for memory efficiency
                try:
                    self.pipeline.enable_attention_slicing(1)
                except:
                    print("Attention slicing not available")
                
                self.model_loaded = True
                load_time = time.time() - start_time
                print(f"Model loaded successfully in {load_time:.2f} seconds!")
                
            except Exception as e:
                print(f"Error loading pipeline: {e}")
                raise
                
        except Exception as e:
            print(f"Critical error in model loading: {e}")
            raise

    @modal.method()
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate images with improved error handling"""
        if not self.model_loaded:
            return GenerationResponse(
                success=False,
                images=[],
                seed_used=-1,
                error="Model not loaded properly"
            )
        
        try:
            print(f"Starting generation for prompt: {request.prompt[:50]}...")
            
            # Process input images
            input_images = []
            second_image_dimensions = None
            for i, img_b64 in enumerate([request.image_1, request.image_2, request.image_3]):
                if img_b64:
                    try:
                        # Decode base64 image
                        img_data = base64.b64decode(img_b64)
                        img = Image.open(io.BytesIO(img_data))
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        input_images.append(img)
                        print(f"Processed input image {i+1}: {img.size}")
                        
                        # Store dimensions of the second image (index 1)
                        if i == 1:
                            second_image_dimensions = img.size
                            print(f"Second image dimensions: {second_image_dimensions}")
                    except Exception as e:
                        print(f"Error processing input image {i+1}: {e}")
                        continue
            
            if len(input_images) == 0:
                input_images = None
                print("No input images provided")
            else:
                print(f"Using {len(input_images)} input images")
            
            # Handle seed
            seed = request.seed
            if seed == -1:
                seed = random.randint(0, 2**16 - 1)
            
            generator = torch.Generator(device=self.accelerator.device).manual_seed(seed)
            print(f"Using seed: {seed}")
            
            # Set scheduler
            if request.scheduler == 'euler':
                self.pipeline.scheduler = self.FlowMatchEulerDiscreteScheduler()
            elif request.scheduler == 'dpmsolver':
                self.pipeline.scheduler = self.DPMSolverMultistepScheduler(
                    algorithm_type="dpmsolver++",
                    solver_type="midpoint",
                    solver_order=2,
                    prediction_type="flow_prediction",
                )
            
            print(f"Using scheduler: {request.scheduler}")
            
            # Use second image dimensions if available, otherwise use request dimensions
            output_width = request.width
            output_height = request.height
            if second_image_dimensions:
                output_width, output_height = second_image_dimensions
                print(f"Using second image dimensions: {output_width}x{output_height}")
            else:
                print(f"Using request dimensions: {output_width}x{output_height}")
            
            # Generate images with progress tracking
            print("Starting image generation...")
            results = self.pipeline(
                prompt=request.prompt,
                input_images=input_images,
                width=output_width,
                height=output_height,
                max_input_image_side_length=request.max_input_image_side_length,
                max_pixels=request.max_pixels,
                num_inference_steps=request.num_inference_steps,
                max_sequence_length=1024,
                text_guidance_scale=request.text_guidance_scale,
                image_guidance_scale=request.image_guidance_scale,
                cfg_range=(request.cfg_range_start, request.cfg_range_end),
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images_per_prompt,
                generator=generator,
                output_type="pil",
            )
            
            print("Image generation completed")
            
            # Convert images to base64
            output_images = []
            for i, img in enumerate(results.images):
                try:
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG", optimize=True)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode()
                    output_images.append(img_b64)
                    print(f"Encoded output image {i+1}")
                except Exception as e:
                    print(f"Error encoding image {i+1}: {e}")
                    continue
            
            print(f"Successfully generated {len(output_images)} images")
            
            return GenerationResponse(
                success=True,
                images=output_images,
                seed_used=seed
            )
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return GenerationResponse(
                success=False,
                images=[],
                seed_used=seed if 'seed' in locals() else -1,
                error=str(e)
            )

    @modal.method()
    def health_check(self) -> dict:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": self.model_loaded,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

# FastAPI app with better error handling and web UI
web_app = FastAPI(
    title="OmniGen2 API",
    description="API for OmniGen2 multimodal generation model",
    version="1.0.0"
)

# HTML template for the web UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniGen2 Image Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 800px;
            width: 90%;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }
        
        input, select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            height: 100px;
            resize: vertical;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .image-upload {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: border-color 0.3s ease;
            margin-bottom: 20px;
        }
        
        .image-upload:hover {
            border-color: #667eea;
        }
        
        .image-uploads {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .generate-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .generate-btn:hover {
            transform: translateY(-2px);
        }
        
        .generate-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            margin-top: 30px;
        }
        
        .result-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        .advanced {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .advanced h3 {
            margin-bottom: 15px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 OmniGen2 Image Generator</h1>
        
        <form id="generateForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="prompt">✨ Prompt</label>
                <textarea 
                    id="prompt" 
                    name="prompt" 
                    placeholder="Describe the image you want to generate..."
                    required
                ></textarea>
            </div>
            
            <div class="image-uploads">
                <div class="image-upload">
                    <label for="image_1">📷 Input Image 1 (Optional)</label>
                    <input type="file" id="image_1" name="image_1" accept="image/*">
                </div>
                <div class="image-upload">
                    <label for="image_2">📷 Input Image 2 (Optional)</label>
                    <input type="file" id="image_2" name="image_2" accept="image/*">
                </div>
                <div class="image-upload">
                    <label for="image_3">📷 Input Image 3 (Optional)</label>
                    <input type="file" id="image_3" name="image_3" accept="image/*">
                </div>
            </div>
            
            <div class="advanced">
                <h3>⚙️ Advanced Settings</h3>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="width">Width</label>
                        <input type="number" id="width" name="width" value="1024" min="256" max="1024">
                    </div>
                    <div class="form-group">
                        <label for="height">Height</label>
                        <input type="number" id="height" name="height" value="1024" min="256" max="1024">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="scheduler">Scheduler</label>
                        <select id="scheduler" name="scheduler">
                            <option value="euler">Euler</option>
                            <option value="dpmsolver">DPM Solver</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="num_inference_steps">Inference Steps</label>
                        <input type="number" id="num_inference_steps" name="num_inference_steps" value="50" min="20" max="100">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="text_guidance_scale">Text Guidance Scale</label>
                        <input type="number" id="text_guidance_scale" name="text_guidance_scale" value="5.0" min="1.0" max="8.0" step="0.1">
                    </div>
                    <div class="form-group">
                        <label for="image_guidance_scale">Image Guidance Scale</label>
                        <input type="number" id="image_guidance_scale" name="image_guidance_scale" value="2.0" min="1.0" max="10.0" step="0.1">
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="seed">Seed (-1 for random)</label>
                    <input type="number" id="seed" name="seed" value="-1" min="-1" max="2147483647">
                </div>
            </div>
            
            <button type="submit" class="generate-btn">🚀 Generate Image</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Generating your image... This may take a minute.</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results"></div>
    </div>

    <script>
        document.getElementById('generateForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const button = document.querySelector('.generate-btn');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const results = document.getElementById('results');
            
            // Clear previous results
            error.style.display = 'none';
            results.innerHTML = '';
            
            // Show loading state
            button.disabled = true;
            button.textContent = '⏳ Generating...';
            loading.style.display = 'block';
            
            try {
                // Create FormData from form
                const formData = new FormData(this);
                
                // Send request
                const response = await fetch('/api/generate-web', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success && result.images && result.images.length > 0) {
                    // Display generated images
                    result.images.forEach((imageBase64, index) => {
                        const img = document.createElement('img');
                        img.src = `data:image/png;base64,${imageBase64}`;
                        img.className = 'result-image';
                        img.alt = `Generated image ${index + 1}`;
                        results.appendChild(img);
                    });
                    
                    // Show seed info
                    const seedInfo = document.createElement('p');
                    seedInfo.textContent = `Seed used: ${result.seed_used}`;
                    seedInfo.style.textAlign = 'center';
                    seedInfo.style.color = '#666';
                    seedInfo.style.marginTop = '10px';
                    results.appendChild(seedInfo);
                } else {
                    error.textContent = result.error || 'Generation failed';
                    error.style.display = 'block';
                }
            } catch (err) {
                error.textContent = 'Network error: ' + err.message;
                error.style.display = 'block';
            } finally {
                button.disabled = false;
                button.textContent = '🚀 Generate Image';
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
'''

@web_app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Web UI for OmniGen2"""
    return HTMLResponse(content=HTML_TEMPLATE)

@web_app.get("/api")
async def api_root():
    return {"message": "OmniGen2 API is running", "status": "healthy"}

@web_app.post("/api/generate-web")
async def generate_web(
    request: Request,
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    scheduler: str = Form("euler"),
    num_inference_steps: int = Form(50),
    text_guidance_scale: float = Form(5.0),
    image_guidance_scale: float = Form(2.0),
    seed: int = Form(-1),
    image_1: UploadFile = File(None),
    image_2: UploadFile = File(None),
    image_3: UploadFile = File(None)
):
    """Generate images via web form"""
    try:
        # Process uploaded files
        image_1_b64 = None
        image_2_b64 = None
        image_3_b64 = None
        
        if image_1 and image_1.size > 0:
            content = await image_1.read()
            image_1_b64 = base64.b64encode(content).decode()
            
        if image_2 and image_2.size > 0:
            content = await image_2.read()
            image_2_b64 = base64.b64encode(content).decode()
            
        if image_3 and image_3.size > 0:
            content = await image_3.read()
            image_3_b64 = base64.b64encode(content).decode()
        
        # Create generation request
        gen_request = GenerationRequest(
            prompt=prompt,
            width=width,
            height=height,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            seed=seed,
            image_1=image_1_b64,
            image_2=image_2_b64,
            image_3=image_3_b64
        )
        
        # Generate images
        model = OmniGen2Model()
        result = model.generate.remote(gen_request)
        
        return JSONResponse(result.model_dump())
        
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@web_app.get("/health")
async def health():
    """Enhanced health check"""
    try:
        model = OmniGen2Model()
        health_status = model.health_check.remote()
        return health_status
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@web_app.post("/api/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """Generate images using OmniGen2 model (API endpoint)"""
    try:
        model = OmniGen2Model()
        result = model.generate.remote(request)
        return result
    except Exception as e:
        print(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@web_app.post("/download-model")
async def trigger_model_download():
    """Manually trigger model download"""
    try:
        cache_path = download_model.remote()
        return {"success": True, "cache_path": cache_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the FastAPI app with increased timeout
@app.function(
    image=image,
    min_containers=0,  # Allow scaling to zero
    timeout=1800,  # 30 minutes
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# CLI for local testing and deployment
if __name__ == "__main__":
    import uvicorn
    
    # For local development
    uvicorn.run(web_app, host="0.0.0.0", port=8000)