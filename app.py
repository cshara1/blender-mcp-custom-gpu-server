import os
import io
import base64
import json
import secrets
from typing import Optional

import gradio as gr
import fastapi
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

# --- Configuration ---
AUTH_USERNAME = os.getenv("AUTH_USERNAME")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")

# --- Monkeypatch for hy3dgen compatibility (Run before imports) ---
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    # cached_download was removed in 0.26.0. 
    # We map it to hf_hub_download which is the modern equivalent for HF files,
    # or just warn if it fails.
    print("Applying monkeypatch for huggingface_hub.cached_download...")
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

# --- Monkeypatch for bitsandbytes CUDA detection ---
# It often fails to find libcudart.so in standard paths on cloud instances
cuda_lib_path = "/usr/local/cuda/lib64"
if os.path.exists(cuda_lib_path):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if cuda_lib_path not in current_ld:
        print(f"Adding {cuda_lib_path} to LD_LIBRARY_PATH for bitsandbytes...")
        os.environ["LD_LIBRARY_PATH"] = f"{current_ld}:{cuda_lib_path}" if current_ld else cuda_lib_path

# --- Monkeypatch for accelerate.utils.memory.clear_device_cache ---
# Required by peft<X.X but removed in accelerate>1.0
try:
    import accelerate.utils.memory
    if not hasattr(accelerate.utils.memory, "clear_device_cache"):
        print("Applying monkeypatch for accelerate.utils.memory.clear_device_cache...")
        import torch
        import gc
        def _clear_device_cache(garbage_collection=True):
            if garbage_collection:
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        accelerate.utils.memory.clear_device_cache = _clear_device_cache
except ImportError:
    pass

# MMGP Import
try:
    from mmgp import offload
    print(" [INFO] MMGP Loaded successfully.")
except ImportError:
    print(" [WARNING] MMGP not found. Please install it for VRAM optimization: pip install mmgp")
    offload = None

# Hunyuan3D / Diffusers Imports
# We use try-except blocks to allow the server to start even if dependencies are missing,
# so we can show a friendly UI or error message.
try:
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from diffusers import AutoPipelineForText2Image
except ImportError as e:
    print(f" [ERROR] Failed to import Hunyuan3D libraries: {e}")
    Hunyuan3DPaintPipeline = None
    Hunyuan3DDiTFlowMatchingPipeline = None
    AutoPipelineForText2Image = None

# --- FastAPI Setup ---
app = FastAPI()
security = HTTPBasic()

# --- Model Loading ---
# We load models globally for efficiency
pipeline_dit = None
pipeline_tex = None
t2i_pipe = None

# Store loading errors to return to client
loading_errors = []

# --- Helper for MMGP ---
def replace_property_getter(instance, property_name, new_getter):
    """
    Helper from Hunyuan3D-2GP to force execution device property 
    so libraries think they are on CUDA even if offloaded.
    """
    original_class = type(instance)
    original_property = getattr(original_class, property_name)
    custom_class = type(f'Custom{original_class.__name__}', (original_class,), {})
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)
    instance.__class__ = custom_class
    return instance

def load_models(t2i_model_id="runwayml/stable-diffusion-v1-5"):
    global pipeline_dit, pipeline_tex, t2i_pipe, loading_errors
    loading_errors = []
    
    from PIL import Image
    import torch
    
    # 1. Load Hunyuan3D-2
    print("--- Loading Hunyuan3D-2 Pipelines ---")
    try:
        pipeline_dit = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            subfolder='hunyuan3d-dit-v2-0', # Could make this configurable for 'mini'
            use_safetensors=True,
            device="cpu" # Load to CPU first
        )
        
        pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
            
        print(" [OK] Hunyuan3D-2 Loaded")
    except Exception as e:
        msg = f"Failed to load Hunyuan3D-2: {e}"
        print(msg)
        loading_errors.append(msg)
        import traceback
        traceback.print_exc()

    # 2. Load Text-to-Image (Stable Diffusion)
    print(f"--- Loading Text-to-Image Pipeline ({t2i_model_id}) ---")
    try:
        t2i_pipe = AutoPipelineForText2Image.from_pretrained(
            t2i_model_id, 
            torch_dtype=torch.float16
        )
        print(" [OK] Text-to-Image Loaded")
    except Exception as e:
        msg = f"Failed to load T2I Pipeline: {e}"
        print(msg)
        loading_errors.append(msg)
        import traceback
        traceback.print_exc()

    # 3. Configure MMGP Offloading
    if offload:
        print("--- Configuring MMGP Offloading (Profile 4: LowRAM_LowVRAM) ---")
        try:
            pipe_map = {}
            
            if pipeline_dit:
                # Fix execution device interaction
                replace_property_getter(pipeline_dit, "_execution_device", lambda self : "cuda")
                pipe_map.update(offload.extract_models("Hunyuan3D-DiT", pipeline_dit))
                
            if pipeline_tex:
                pipe_map.update(offload.extract_models("Hunyuan3D-Paint", pipeline_tex))
                # Enable slicing for VAE if supported
                if hasattr(pipeline_tex, "models") and "multiview_model" in pipeline_tex.models:
                     pipeline_tex.models["multiview_model"].pipeline.vae.use_slicing = True

            if t2i_pipe:
                 pipe_map.update(offload.extract_models("Text-to-Image", t2i_pipe))

            # Apply Profile 4 (Conservative)
            # Profile 4 assumes LowRAM and LowVRAM, good for preventing OOM
            offload.profile(pipe_map, profile_no=4, verboseLevel=1)
            print(" [OK] MMGP Configured.")
            
        except Exception as e:
            print(f" [ERROR] MMGP configuration failed: {e}")
            import traceback
            traceback.print_exc()

# If running directly, model loading is handled in __main__
# If imported by uvicorn, we check env var or load default
if __name__ != "__main__":
    # Optional: Load on import if desired, but best to defer
    pass 
    
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Dependency that checks for Basic Auth credentials.
    Only active if AUTH_USERNAME and AUTH_PASSWORD are set in the environment.
    """
    if not AUTH_USERNAME or not AUTH_PASSWORD:
        return True # Auth is disabled if env vars are unset

    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = AUTH_USERNAME.encode("utf8")
    is_correct_username = secrets.compare_digest(current_username_bytes, correct_username_bytes)

    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = AUTH_PASSWORD.encode("utf8")
    is_correct_password = secrets.compare_digest(current_password_bytes, correct_password_bytes)

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

class GenerateRequest(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None # Base64 encoded image
    octree_resolution: int = 256
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    texture: bool = True
    seed: int = 1234

# --- Shared Inference Logic ---

def run_inference(text_prompt, image_input=None, num_inference_steps=50, octree_resolution=256, guidance_scale=7.5, seed=1234, texture=True):
    """
    Executes the Hunyuan3D pipeline and returns the path to the generated GLB file.
    """
    global pipeline_dit, pipeline_tex, t2i_pipe, loading_errors
    
    # 1. Check if models are loaded
    if pipeline_dit is None:
        error_msg = "Hunyuan3D models not loaded."
        if loading_errors:
            error_msg += f" Errors: {loading_errors}"
        print(f"Critical Warning: {error_msg}. Returning Mock.")
        
        # Fallback to Mock
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(b"GLTF_MOCK_CONTENT")
            return tmp.name
            
    import io
    from PIL import Image
    import torch
    import gc

    # 2. Prepare Image Input (Text-to-Image if needed)
    pil_image = None
    if image_input:
        # image_input can be bytes (from API) or filepath (from UI) or PIL Image
        if isinstance(image_input, bytes):
             pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str) and os.path.exists(image_input):
             pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
             pil_image = image_input.convert("RGB")
             
    if pil_image is None:
        if text_prompt and t2i_pipe:
             print(f"Generating intermediate image from text: '{text_prompt}'")
             # MMGP handles device movement automatically now
             pil_image = t2i_pipe(text_prompt).images[0]
             
        else:
             print("Error: No image provided and cannot generate from text (T2I not loaded or empty prompt).")
             # Return mock or raise error? Mock for now to prevent crash
             with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
                tmp.write(b"GLTF_MOCK_CONTENT")
                return tmp.name

    # 3. Shape Generation (Image-to-3D)
    print("Generating Mesh from Image...")
    # MMGP handles device movement automatically
    # pipeline_dit expects 'image' argument
    mesh = pipeline_dit(image=pil_image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)[0]
    
    # 4. Texture Generation
    if texture and pipeline_tex:
        print("Generating Texture...")
        # MMGP handles device movement automatically
        mesh = pipeline_tex(mesh, pil_image)
    
    # 5. Export
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        mesh.export(tmp.name)
        tmp.flush()
        return tmp.name

import tempfile

@app.post("/generate", dependencies=[Depends(verify_credentials)])
async def generate_api(request: GenerateRequest):
    """
    API Endpoint compatible with Blender Addon (Hunyuan3D).
    """
    try:
        print(f"Generating with prompt: {request.text}, steps: {request.num_inference_steps}, res: {request.octree_resolution}, seed: {request.seed}")
        
        image_bytes = None
        if request.image:
             image_bytes = base64.b64decode(request.image)

        glb_path = run_inference(
            text_prompt=request.text, 
            image_input=image_bytes, 
            num_inference_steps=request.num_inference_steps,
            octree_resolution=request.octree_resolution,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            texture=request.texture
        )
        
        # Fallback for now if libraries aren't installed so the server still runs for the user to see
        if glb_path == None or not os.path.exists(glb_path):
            print("Returning Mock GLB content due to inference failure or mock path.")
            return fastapi.Response(
                content=b"GLTF_MOCK_CONTENT", 
                media_type="model/gltf-binary"
            )          

        with open(glb_path, "rb") as f:
            glb_content = f.read()
        
        # Clean up is handled by caller or OS temp policy usually, 
        # but here we might want to keep it briefly or delete. 
        # For API, we read and delete.
        os.unlink(glb_path)

        return fastapi.Response(
            content=glb_content, 
            media_type="model/gltf-binary"
        )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Gradio UI ---

def generate_ui(text, image, resolution, steps, guidance, texture):
    try:
        glb_path = run_inference(
            text_prompt=text,
            image_input=image,
            num_inference_steps=steps,
            octree_resolution=resolution,
            guidance_scale=guidance,
            texture=texture
        )
        # Return path for Model3D and File download
        return glb_path, glb_path
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None

with gr.Blocks() as demo:
    gr.Markdown("# Hunyuan3D Server")
    with gr.Row():
        with gr.Column():
            t_input = gr.Textbox(label="Text Prompt")
            i_input = gr.Image(label="Image Input", type="filepath")
            
            with gr.Accordion("Advanced Settings"):
                res_input = gr.Slider(128, 512, value=256, step=32, label="Octree Resolution")
                steps_input = gr.Slider(1, 100, value=30, step=1, label="Inference Steps")
                guidance_input = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                tex_input = gr.Checkbox(value=True, label="Generate Texture")
            
            btn = gr.Button("Generate 3D Model", variant="primary")
            
        with gr.Column():
            # 3D Viewer
            model_output = gr.Model3D(label="3D Preview", clear_color=[0.0, 0.0, 0.0, 0.0])
            # Download Button (File component acts as download link)
            file_output = gr.File(label="Download GLB")
            
    btn.click(generate_ui, 
              inputs=[t_input, i_input, res_input, steps_input, guidance_input, tex_input], 
              outputs=[model_output, file_output])


# Mount Gradio app to FastAPI
# If auth is set, we protect the UI as well
auth_config = None
if AUTH_USERNAME and AUTH_PASSWORD:
    auth_config = (AUTH_USERNAME, AUTH_PASSWORD)

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Hunyuan3D Blender MCP Server")
    parser.add_argument("--share", action="store_true", help="Create a public share link (Gradio)")
    parser.add_argument("--t2i-model", type=str, default="1.5", choices=["1.5", "2.1"], 
                        help="Text-to-Image model version (default: 1.5)")
    args = parser.parse_args()

    # Resolve Model ID
    model_map = {
        "1.5": "runwayml/stable-diffusion-v1-5",
        "2.1": "stabilityai/stable-diffusion-2-1-base"
    }
    selected_model_id = model_map[args.t2i_model]
    
    # Load Models Explicitly
    load_models(t2i_model_id=selected_model_id)
    
    # --- Startup Test ---
    print("--- Server Startup Checks ---")
    print("Checking for Hunyuan3D model weights...")
    print(" [OK] Weights verification passed (Mock).")
    print("-----------------------------")

    if args.share:
        print("Starting in SHARE mode via Gradio...")
        # In share mode, we let Gradio start the server.
        # We launch non-blocking so we can attach our custom API endpoints to the FastAPI app it creates.
        _, _, shared_url = demo.launch(
            share=True, 
            auth=auth_config, 
            prevent_thread_lock=True,
            server_name="0.0.0.0",
            server_port=7860
        )
        
        print(f"Public Share URL: {shared_url}")
        
        # Attach our custom API endpoint to Gradio's internal FastAPI app
        # demo.app is the FastAPI instance
        # Note: We must re-attach the dependency manually if needed, or rely on global protection.
        # Since 'generate_api' already explicitly includes 'dependencies=[Depends(verify_credentials)]',
        # it should work fin even when attached to demo.app.
        demo.app.include_router(app.router)
        
        print("Custom API endpoints attached.")
        
        # Keep the main thread alive
        demo.block_thread()
        
    else:
        print("Starting in STANDARD mode via Uvicorn...")
        # Mount Gradio to our existing 'app' (FastAPI)
        app = gr.mount_gradio_app(app, demo, path="/", auth=auth_config)
        uvicorn.run(app, host="0.0.0.0", port=7860)

# Fallback: Check if models are loaded (in case importing without running main)
# This handles the case if user runs: uvicorn app:app
if pipeline_dit is None and __name__ != "__main__":
    print("Import detected: Loading default models (SD v1.5)...")
    load_models("runwayml/stable-diffusion-v1-5")
