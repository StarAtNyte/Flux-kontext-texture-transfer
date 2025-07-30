import os
import io
import base64
import random
from datetime import datetime
from typing import Optional, List, Union
from dataclasses import dataclass

import modal
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Modal app setup
app = modal.App("flux-kontext-api")

# Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install([
        "torch==2.6.0",
        "torchvision==0.21.0", 
        "transformers==4.51.3",
        "diffusers",
        "accelerate",
        "opencv-python-headless",
        "scipy",
        "matplotlib",
        "Pillow",
        "tqdm",
        "fastapi",
        "pydantic",
        "huggingface_hub",
        "python-multipart",
        "numpy",
        "sentencepiece",
    ])
    .add_local_dir(".", "/app", copy=True)
    .workdir("/app")
    .env({"HF_HUB_CACHE": "/models/huggingface"})
)

# Request/Response Models
class FluxKontextRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    width: Optional[int] = Field(default=None, ge=256, le=2048, description="Output image width")
    height: Optional[int] = Field(default=None, ge=256, le=2048, description="Output image height") 
    num_inference_steps: int = Field(default=28, ge=10, le=100, description="Number of inference steps")
    guidance_scale: float = Field(default=3.5, ge=1.0, le=20.0, description="Guidance scale")
    true_cfg_scale: float = Field(default=1.0, ge=1.0, le=10.0, description="True CFG scale")
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt")
    negative_prompt_2: Optional[str] = Field(default=None, description="Secondary negative prompt")
    max_area: int = Field(default=1024*1024, ge=256*256, le=2048*2048, description="Max area in pixels")
    num_images_per_prompt: int = Field(default=1, ge=1, le=4, description="Number of images per prompt")
    seed: int = Field(default=-1, ge=-1, le=2147483647, description="Seed for generation (-1 for random)")
    
    # Multi-image support - base64 encoded images
    images: Optional[List[str]] = Field(default=None, description="List of base64 encoded input images")
    concatenate_direction: str = Field(default="horizontal", pattern="^(horizontal|vertical)$", description="Image concatenation direction")

class FluxKontextResponse(BaseModel):
    success: bool
    images: List[str] = Field(description="List of base64 encoded generated images")
    seed_used: int = Field(description="Seed that was used for generation")
    error: Optional[str] = None
    processing_time: Optional[float] = None

# GPU volume for model caching
volume = modal.Volume.from_name("flux-kontext-models", create_if_missing=True)

def concatenate_images(images: List[Image.Image], direction: str = "horizontal") -> Image.Image:
    """
    Concatenate multiple PIL images either horizontally or vertically.
    """
    if not images:
        raise ValueError("No images provided for concatenation")
    
    # Filter out None images and convert to RGB
    valid_images = [img.convert("RGB") for img in images if img is not None]
    
    if not valid_images:
        raise ValueError("No valid images provided for concatenation")
    
    if len(valid_images) == 1:
        return valid_images[0]
    
    if direction == "horizontal":
        # Calculate total width and max height
        total_width = sum(img.width for img in valid_images)
        max_height = max(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        # Paste images
        x_offset = 0
        for img in valid_images:
            # Center image vertically if heights differ
            y_offset = (max_height - img.height) // 2
            concatenated.paste(img, (x_offset, y_offset))
            x_offset += img.width
            
    else:  # vertical
        # Calculate max width and total height
        max_width = max(img.width for img in valid_images)
        total_height = sum(img.height for img in valid_images)
        
        # Create new image
        concatenated = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # Paste images
        y_offset = 0
        for img in valid_images:
            # Center image horizontally if widths differ
            x_offset = (max_width - img.width) // 2
            concatenated.paste(img, (x_offset, y_offset))
            y_offset += img.height
    
    return concatenated

# Separate function to download model
@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours for model download
    cpu=4,
    memory=8192,
)
def download_model():
    """Download and cache the FLUX.1-Kontext model"""
    import os
    from huggingface_hub import snapshot_download
    
    cache_path = "/models/flux-kontext"
    
    # Always try to download/verify the model
    print("Downloading/verifying FLUX.1-Kontext model...")
    try:
        # Download model with progress tracking
        downloaded_path = snapshot_download(
            repo_id="black-forest-labs/FLUX.1-Kontext-dev",
            cache_dir="/models/huggingface",
            local_dir=cache_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"Model downloaded successfully to {downloaded_path}")
        
        # Use the actual downloaded path if different from expected
        actual_cache_path = downloaded_path if downloaded_path != cache_path else cache_path
        
        # Verify key files exist
        model_index_path = os.path.join(actual_cache_path, "model_index.json")
        if os.path.exists(model_index_path):
            print("✓ model_index.json found")
        else:
            print("✗ model_index.json NOT found")
            # List files in directory for debugging
            if os.path.exists(actual_cache_path):
                files = os.listdir(actual_cache_path)
                print(f"Files in {actual_cache_path}: {files}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise
    
    # Commit changes to volume
    volume.commit()
    return downloaded_path  # Return the actual download path

@app.cls(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=300,
    memory=32768,
)
class FluxKontextModel:
    pipeline = None
    model_loaded = False
        
    @modal.enter()
    def load_model(self):
        """Load the FLUX.1-Kontext model"""
        import os
        import time
        import sys
        
        # Add current directory to path for local imports
        sys.path.append('/app')
        
        print("Starting FLUX Kontext model loading process...")
        start_time = time.time()
        
        try:
            # Import the pipeline from local files
            from pipeline_flux_kontext import FluxKontextPipeline
            
            print("Pipeline import successful")
            
            # Check for cached model and download if needed
            cache_path = "/models/flux-kontext"
            actual_model_path = cache_path
            
            # Check if model exists and is complete
            model_index_path = os.path.join(cache_path, "model_index.json")
            if not os.path.exists(cache_path) or not os.path.exists(model_index_path):
                print("Model not found in cache or incomplete, downloading...")
                actual_model_path = download_model.remote()
                print(f"Download returned path: {actual_model_path}")
            
            print(f"Loading model from: {actual_model_path}")
            
            # Try to load the pipeline from the actual model path
            try:
                self.pipeline = FluxKontextPipeline.from_pretrained(
                    actual_model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    token=os.environ.get("HF_TOKEN"),
                )
                print("Pipeline loaded successfully from downloaded path")
            except Exception as e:
                print(f"Failed to load from downloaded path {actual_model_path}: {e}")
                print("Trying to load directly from Hugging Face repo...")
                # Try loading directly from repo as fallback
                try:
                    self.pipeline = FluxKontextPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-Kontext-dev",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True,
                        token=os.environ.get("HF_TOKEN"),
                        cache_dir="/models/huggingface",
                    )
                    print("Pipeline loaded directly from Hugging Face")
                except Exception as e2:
                    print(f"Failed to load from Hugging Face: {e2}")
                    raise
            
            # Move to GPU
            self.pipeline = self.pipeline.to("cuda")
            
            # Enable optimizations
            self.pipeline.enable_model_cpu_offload()
            
            # Optional: Enable memory optimizations
            try:
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_vae_tiling()
            except:
                print("VAE optimizations not available")
            
            self.model_loaded = True
            load_time = time.time() - start_time
            print(f"FLUX Kontext model loaded successfully in {load_time:.2f} seconds!")
            
        except Exception as e:
            print(f"Critical error in model loading: {e}")
            raise

    @modal.method()
    def generate(self, request: FluxKontextRequest) -> FluxKontextResponse:
        """Generate images using FLUX Kontext model"""
        if not self.model_loaded:
            return FluxKontextResponse(
                success=False,
                images=[],
                seed_used=-1,
                error="Model not loaded properly"
            )
        
        try:
            import time
            start_time = time.time()
            
            print(f"Starting generation for prompt: {request.prompt[:50]}...")
            
            # Process input images if provided
            input_image = None
            if request.images and len(request.images) > 0:
                try:
                    # Decode base64 images
                    pil_images = []
                    for i, img_b64 in enumerate(request.images):
                        if img_b64:
                            img_data = base64.b64decode(img_b64)
                            img = Image.open(io.BytesIO(img_data))
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            pil_images.append(img)
                            print(f"Processed input image {i+1}: {img.size}")
                    
                    if pil_images:
                        # Concatenate images
                        input_image = concatenate_images(pil_images, request.concatenate_direction)
                        print(f"Concatenated image size: {input_image.size}")
                        
                except Exception as e:
                    print(f"Error processing input images: {e}")
                    return FluxKontextResponse(
                        success=False,
                        images=[],
                        seed_used=-1,
                        error=f"Error processing input images: {str(e)}"
                    )
            
            # Handle seed
            seed = request.seed
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            
            generator = torch.Generator(device="cuda").manual_seed(seed)
            print(f"Using seed: {seed}")
            
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": request.prompt,
                "guidance_scale": request.guidance_scale,
                "num_inference_steps": request.num_inference_steps,
                "num_images_per_prompt": request.num_images_per_prompt,
                "generator": generator,
                "output_type": "pil",
            }
            
            # Add optional parameters
            if input_image is not None:
                generation_kwargs["image"] = input_image
                
                # Use dimensions of the second image if available, otherwise use input_image dimensions
                if len(pil_images) >= 2:
                    # Use dimensions from the second image
                    second_img = pil_images[1]
                    generation_kwargs["width"] = second_img.size[0]
                    generation_kwargs["height"] = second_img.size[1]
                    print(f"Using second image dimensions: {second_img.size[0]}x{second_img.size[1]}")
                elif not (request.width and request.height):
                    # Fallback to input image dimensions if no second image and no explicit dimensions
                    generation_kwargs["width"] = input_image.size[0]
                    generation_kwargs["height"] = input_image.size[1]
                    print(f"Using input image dimensions: {input_image.size[0]}x{input_image.size[1]}")
                
            if request.negative_prompt:
                generation_kwargs["negative_prompt"] = request.negative_prompt
                
            if request.negative_prompt_2:
                generation_kwargs["negative_prompt_2"] = request.negative_prompt_2
                
            if request.true_cfg_scale > 1.0:
                generation_kwargs["true_cfg_scale"] = request.true_cfg_scale
                
            # Override with explicit dimensions if provided
            if request.width and request.height:
                generation_kwargs["width"] = request.width
                generation_kwargs["height"] = request.height
                
            generation_kwargs["max_area"] = request.max_area
            
            # Generate images
            print("Starting image generation...")
            results = self.pipeline(**generation_kwargs)
            
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
            
            processing_time = time.time() - start_time
            print(f"Successfully generated {len(output_images)} images in {processing_time:.2f}s")
            
            return FluxKontextResponse(
                success=True,
                images=output_images,
                seed_used=seed,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return FluxKontextResponse(
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

# FastAPI app with web UI
web_app = FastAPI(
    title="FLUX.1-Kontext Multi-Image API",
    description="API for FLUX.1-Kontext multi-image generation model",
    version="1.0.0"
)

# HTML template for the web UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FLUX.1-Kontext Multi-Image Generator</title>
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
            max-width: 1000px;
            width: 95%;
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
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
        
        .image-container {
            position: relative;
            display: inline-block;
            margin-bottom: 20px;
        }
        
        .download-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(102, 126, 234, 0.9);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .download-btn:hover {
            background: rgba(102, 126, 234, 1);
        }
        
        .download-btn:active {
            transform: scale(0.95);
        }
        
        .multiview-section {
            margin-top: 30px;
            border-top: 2px solid #e1e5e9;
            padding-top: 20px;
            display: none;
        }
        
        .multiview-btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.3s ease;
            margin-bottom: 20px;
        }
        
        .multiview-btn:hover {
            transform: translateY(-2px);
        }
        
        .multiview-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .multiview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .multiview-item {
            text-align: center;
        }
        
        .multiview-item h4 {
            margin-bottom: 10px;
            color: #333;
            font-size: 16px;
        }
        
        .multiview-loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .multiview-loading .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #28a745;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
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
        
        .concatenation-options {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
        }
        
        .radio-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 FLUX.1-Kontext Multi-Image Generator</h1>
        
        <form id="generateForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="prompt">✨ Prompt</label>
                <textarea 
                    id="prompt" 
                    name="prompt" 
                    placeholder="Describe how you want to transform the images..."
                    rows="4"
                    required
                >Apply the brown and beige horizontal striped fabric texture from the left side as upholstery to the sectional sofa, then show only the reupholstered sofa in the final image. The striped pattern should wrap naturally around all cushions, backrest, armrests, and chaise section. Remove the texture sample from the final result - display only the finished sectional sofa with the new striped upholstery against a clean background.</textarea>
                <p style="font-size: 12px; color: #666; margin-top: 5px;">Describe the transformation you want to apply to the images</p>
            </div>
            
            <div class="image-uploads">
                <div class="image-upload">
                    <label for="texture_image">🎨 Texture Image (Required)</label>
                    <input type="file" id="texture_image" name="texture_image" accept="image/*,.webp" required>
                    <p style="font-size: 12px; color: #666; margin-top: 5px;">Upload the image containing the texture/pattern to apply</p>
                </div>
                <div class="image-upload">
                    <label for="furniture_image">🪑 Furniture Image (Required)</label>
                    <input type="file" id="furniture_image" name="furniture_image" accept="image/*,.webp" required>
                    <p style="font-size: 12px; color: #666; margin-top: 5px;">Upload the image containing the furniture/sofa to modify</p>
                </div>
            </div>
            
            <div class="advanced">
                <h3>⚙️ Advanced Settings</h3>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="num_inference_steps">Inference Steps</label>
                        <input type="number" id="num_inference_steps" name="num_inference_steps" value="28" min="10" max="100">
                    </div>
                    <div class="form-group">
                        <label for="guidance_scale">Guidance Scale</label>
                        <input type="number" id="guidance_scale" name="guidance_scale" value="3.5" min="1.0" max="20.0" step="0.1">
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="seed">Seed (-1 for random)</label>
                    <input type="number" id="seed" name="seed" value="-1" min="-1" max="2147483647">
                </div>
            </div>
            
            <button type="submit" class="generate-btn">🚀 Generate Images</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Generating your images... This may take a minute.</p>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="results" id="results"></div>
        
        <div class="multiview-section" id="multiviewSection">
            <h3>📐 Generate Multiple Views</h3>
            <p style="color: #666; margin-bottom: 15px;">Generate different angle views of your furniture design</p>
            <button type="button" class="multiview-btn" id="generateMultiviewBtn">🔄 Generate Top, Back & Side Views</button>
            
            <div class="multiview-loading" id="multiviewLoading">
                <div class="spinner"></div>
                <p>Generating multiple views... This may take a few minutes.</p>
            </div>
            
            <div class="multiview-grid" id="multiviewGrid"></div>
        </div>
    </div>

    <script>
        // Function to download image as PNG
        function downloadImage(base64Data, filename) {
            try {
                // Create a link element
                const link = document.createElement('a');
                link.href = `data:image/png;base64,${base64Data}`;
                link.download = filename;
                
                // Append to body, click, and remove
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } catch (error) {
                console.error('Download failed:', error);
                alert('Download failed. Please try again.');
            }
        }
        
        
        // Global variable to store the generated image
        let generatedImageBase64 = null;
        
        // Function to generate multi-view images
        async function generateMultiViews() {
            if (!generatedImageBase64) {
                alert('Please generate an image first');
                return;
            }
            
            const button = document.getElementById('generateMultiviewBtn');
            const loading = document.getElementById('multiviewLoading');
            const grid = document.getElementById('multiviewGrid');
            
            button.disabled = true;
            button.textContent = '⏳ Generating Views...';
            loading.style.display = 'block';
            grid.innerHTML = '';
            
            const views = [
                { name: 'Top View', prompt: `Show this exact same sofa from a top-down aerial view, maintaining the identical fabric pattern, colors, and design. Only change the camera angle to look down from above. Keep all textures, patterns, and sofa details exactly the same. Professional furniture photography, clean white background.` },
                { name: 'Back View', prompt: `Show this exact same sofa from the rear back view, maintaining the identical fabric pattern, colors, and design. Only change the camera angle to look from behind. Keep all textures, patterns, and sofa details exactly the same. Professional furniture photography, clean white background.` },
                { name: 'Side View', prompt: `Show this exact same sofa from a side profile view, maintaining the identical fabric pattern, colors, and design. Only change the camera angle to look from the side. Keep all textures, patterns, and sofa details exactly the same. Professional furniture photography, clean white background.` }
            ];
            
            for (let i = 0; i < views.length; i++) {
                const view = views[i];
                try {
                    // Create container for this view
                    const viewContainer = document.createElement('div');
                    viewContainer.className = 'multiview-item';
                    
                    const title = document.createElement('h4');
                    title.textContent = view.name;
                    viewContainer.appendChild(title);
                    
                    const loadingSpinner = document.createElement('div');
                    loadingSpinner.innerHTML = '<div class="spinner" style="width: 20px; height: 20px; margin: 10px auto;"></div><p style="font-size: 12px;">Generating...</p>';
                    viewContainer.appendChild(loadingSpinner);
                    
                    grid.appendChild(viewContainer);
                    
                    // Generate this view
                    const formData = new FormData();
                    formData.append('prompt', view.prompt);
                    formData.append('num_inference_steps', '28');
                    formData.append('guidance_scale', '3.5');
                    formData.append('seed', '-1');
                    
                    // Convert base64 to blob and append as file
                    const blob = await fetch(`data:image/png;base64,${generatedImageBase64}`).then(r => r.blob());
                    formData.append('texture_image', blob, 'generated.png');
                    formData.append('furniture_image', blob, 'generated.png');
                    
                    const response = await fetch('/api/generate-web', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    // Remove loading spinner
                    viewContainer.removeChild(loadingSpinner);
                    
                    if (result.success && result.images && result.images.length > 0) {
                        // Create image container
                        const imgContainer = document.createElement('div');
                        imgContainer.className = 'image-container';
                        
                        const img = document.createElement('img');
                        img.src = `data:image/png;base64,${result.images[0]}`;
                        img.className = 'result-image';
                        img.alt = view.name;
                        
                        const downloadBtn = document.createElement('button');
                        downloadBtn.className = 'download-btn';
                        downloadBtn.innerHTML = '📥 Download';
                        downloadBtn.onclick = () => downloadImage(result.images[0], `furniture-${view.name.replace(' ', '-').toLowerCase()}-${Date.now()}.png`);
                        
                        imgContainer.appendChild(img);
                        imgContainer.appendChild(downloadBtn);
                        viewContainer.appendChild(imgContainer);
                    } else {
                        const errorDiv = document.createElement('div');
                        errorDiv.textContent = `Failed to generate ${view.name}`;
                        errorDiv.style.color = 'red';
                        viewContainer.appendChild(errorDiv);
                    }
                } catch (error) {
                    console.error(`Error generating ${view.name}:`, error);
                    const errorDiv = document.createElement('div');
                    errorDiv.textContent = `Error: ${view.name}`;
                    errorDiv.style.color = 'red';
                    grid.appendChild(errorDiv);
                }
            }
            
            button.disabled = false;
            button.textContent = '🔄 Generate Top, Back & Side Views';
            loading.style.display = 'none';
        }
        
        // Add event listener for multi-view button
        document.getElementById('generateMultiviewBtn').addEventListener('click', generateMultiViews);
        
        document.getElementById('generateForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const button = document.querySelector('.generate-btn');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const results = document.getElementById('results');
            
            // Clear previous results
            error.style.display = 'none';
            results.innerHTML = '';
            document.getElementById('multiviewSection').style.display = 'none';
            document.getElementById('multiviewGrid').innerHTML = '';
            
            // Show loading state
            button.disabled = true;
            button.textContent = '⏳ Generating...';
            loading.style.display = 'block';
            
            try {
                // Create FormData from form
                const formData = new FormData(this);
                
                // Debug: Log form data
                console.log('Form data entries:');
                for (let [key, value] of formData.entries()) {
                    console.log(key, typeof value === 'object' ? value.name : value);
                }
                
                // Send request
                const response = await fetch('/api/generate-web', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // Debug: Log response
                console.log('Response status:', response.status);
                console.log('Response result:', result);
                
                if (result.success && result.images && result.images.length > 0) {
                    // Store the generated image for multi-view generation
                    generatedImageBase64 = result.images[0];
                    
                    // Display generated images with download buttons
                    result.images.forEach((imageBase64, index) => {
                        // Create container for image and download button
                        const container = document.createElement('div');
                        container.className = 'image-container';
                        
                        // Create image
                        const img = document.createElement('img');
                        img.src = `data:image/png;base64,${imageBase64}`;
                        img.className = 'result-image';
                        img.alt = `Generated image ${index + 1}`;
                        
                        // Create download button
                        const downloadBtn = document.createElement('button');
                        downloadBtn.className = 'download-btn';
                        downloadBtn.innerHTML = '📥 Download PNG';
                        downloadBtn.onclick = () => downloadImage(imageBase64, `generated-image-${Date.now()}-${index + 1}.png`);
                        
                        container.appendChild(img);
                        container.appendChild(downloadBtn);
                        results.appendChild(container);
                    });
                    
                    // Show the multi-view section
                    document.getElementById('multiviewSection').style.display = 'block';
                    
                    // Show generation info
                    const info = document.createElement('div');
                    info.innerHTML = `
                        <p style="text-align: center; color: #666; margin-top: 10px;">
                            Seed: ${result.seed_used} | 
                            Processing time: ${result.processing_time?.toFixed(2)}s
                        </p>
                    `;
                    results.appendChild(info);
                } else {
                    error.textContent = result.error || 'Generation failed';
                    error.style.display = 'block';
                }
            } catch (err) {
                error.textContent = 'Network error: ' + err.message;
                error.style.display = 'block';
            } finally {
                button.disabled = false;
                button.textContent = '🚀 Generate Images';
                loading.style.display = 'none';
            }
        });
    </script>
</body>
</html>
'''

@web_app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Web UI for FLUX.1-Kontext Multi-Image"""
    return HTMLResponse(content=HTML_TEMPLATE)

@web_app.get("/api")
async def api_root():
    return {"message": "FLUX.1-Kontext Multi-Image API is running", "status": "healthy"}

@web_app.post("/api/generate-web")
async def generate_web(
    request: Request,
    prompt: str = Form(...),
    num_inference_steps: int = Form(28),
    guidance_scale: float = Form(3.5),
    seed: int = Form(-1),
    texture_image: UploadFile = File(...),
    furniture_image: UploadFile = File(...),
):
    """Generate images via web form"""
    try:
        # Process uploaded files
        images_b64 = []
        
        # Process texture image (first)
        texture_content = await texture_image.read()
        texture_b64 = base64.b64encode(texture_content).decode()
        images_b64.append(texture_b64)
        
        # Process furniture image (second)
        furniture_content = await furniture_image.read()
        furniture_b64 = base64.b64encode(furniture_content).decode()
        images_b64.append(furniture_b64)
        
        # Create generation request
        gen_request = FluxKontextRequest(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            images=images_b64,
            concatenate_direction="horizontal"
        )
        
        # Generate images
        model = FluxKontextModel()
        result = model.generate.remote(gen_request)
        
        return JSONResponse(result.model_dump())
        
    except Exception as e:
        print(f"Generate web error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "success": False, 
            "error": str(e),
            "images": [],
            "seed_used": -1
        }, status_code=500)

@web_app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        model = FluxKontextModel()
        health_status = model.health_check.remote()
        return health_status
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@web_app.post("/api/generate", response_model=FluxKontextResponse)
async def generate_image(request: FluxKontextRequest):
    """Generate images using FLUX.1-Kontext model (API endpoint)"""
    try:
        model = FluxKontextModel()
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

# Mount the FastAPI app
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    min_containers=0,
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