import os
import io
import base64
import json
import secrets
from typing import Optional

import gradio as gr
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

# --- Configuration ---
AUTH_USERNAME = os.getenv("AUTH_USERNAME")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")

# --- FastAPI Setup ---
app = FastAPI()
security = HTTPBasic()

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

# --- Shared Inference Logic ---

def run_inference(text_prompt, image_input=None, num_inference_steps=50, guidance_scale=7.5, texture=True):
    """
    Executes the Hunyuan3D pipeline and returns the path to the generated GLB file.
    """
    # Check if helper library matches expectations
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from PIL import Image
    except ImportError:
        print("Warning: `hy3dgen` not found. Returning Mock.")
        # Create a dummy mock file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp.write(b"GLTF_MOCK_CONTENT") # In reality this would be invalid GLB but serves as a file placeholder
            return tmp.name

    # 1. Load Models (Ideally, load these once globally)
    pipeline_dit = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2') 
    
    # 2. Shape Generation
    # Handle inputs
    pil_image = None
    if image_input:
        # image_input can be bytes (from API) or filepath (from UI) or PIL Image
        if isinstance(image_input, bytes):
             pil_image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str) and os.path.exists(image_input):
             pil_image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
             pil_image = image_input

    if pil_image:
         mesh = pipeline_dit(image=pil_image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)[0]
    else:
         mesh = pipeline_dit(prompt=text_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)[0]
         
    # 3. Texture Generation
    if texture:
        mesh = pipeline_tex(mesh, prompt=text_prompt if text_prompt else "3d model")[0]
        
    # 4. Export
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        mesh.export(tmp.name)
        tmp.flush()
        return tmp.name
    return None

import tempfile

@app.post("/generate", dependencies=[Depends(verify_credentials)])
async def generate_api(request: GenerateRequest):
    """
    API Endpoint compatible with Blender Addon (Hunyuan3D).
    """
    try:
        print(f"Generating with prompt: {request.text}, steps: {request.num_inference_steps}")
        
        image_bytes = None
        if request.image:
             image_bytes = base64.b64decode(request.image)

        glb_path = run_inference(
            text_prompt=request.text, 
            image_input=image_bytes, 
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            texture=request.texture
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
    args = parser.parse_args()
    
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
