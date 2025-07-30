import os
import sys
import torch
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import json
import uuid
from datetime import datetime

# Add paths for imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), './submodules/LAVIS'))
sys.path.append(os.path.join(os.getcwd(), './submodules/k-diffusion'))

from lavis.models import load_model_and_preprocess
from src.datasets.dataset import Hairstyle
from src.upsampling.upsampler import HairstyleUpsampler
from src.sampler import sample_euler_ancestral
from src.utils.config import load_config
from src.utils.text_utils import obtain_description_embedding, obtain_blip_features
from src.utils.model_utils import setup_model
import k_diffusion as K
import trimesh

app = FastAPI(
    title="HAAR API",
    description="API for HAAR (Hair Appearance and Rendering) - Generate 3D hairstyles from text descriptions",
    version="1.0.0"
)

# Global variables for model caching
model_ema = None
accelerator = None
model_feature_extractor = None
txt_processors = None
hairstyle_dataset = None
config = None
device = None

class InferenceRequest(BaseModel):
    hairstyle_description: str
    cfg_scale: float = 1.2
    n_samples: int = 1
    step: int = 50
    seed: Optional[int] = None
    save_guiding_strands: bool = True
    save_upsampled_hairstyle: bool = False
    save_latent_textures: bool = False
    upsample_resolution: int = 64

class TextInterpolationRequest(BaseModel):
    hairstyle_1: str
    hairstyle_2: str
    n_interpolation_states: int = 5
    cfg_scale: float = 1.5
    step: int = 50
    seed: int = 32
    degree: float = 80.0
    save_guiding_strands: bool = True
    save_upsampled_hairstyle: bool = False
    save_latent_textures: bool = False
    upsample_resolution: int = 128

class ImageToHairstyleRequest(BaseModel):
    cfg_scale: float = 1.2
    n_samples: int = 1
    step: int = 50
    seed: Optional[int] = None
    save_guiding_strands: bool = True
    save_upsampled_hairstyle: bool = False
    save_latent_textures: bool = False
    upsample_resolution: int = 64

def initialize_models():
    """Initialize and cache models globally"""
    global model_ema, accelerator, model_feature_extractor, txt_processors, hairstyle_dataset, config, device

    if model_ema is None:
        print("Initializing models...")
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        # Load config
        config = load_config("./configs/infer.json")

        # Setup model
        model_ema, accelerator = setup_model(
            config=config,
            ckpt_path="./pretrained_models/haar_prior/haar_diffusion.pth",
            device=device
        )

        # Setup text embedder model
        model_feature_extractor, _, txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=device
        )

        # Setup dataset
        hairstyle_dataset = Hairstyle(**config['dataset'])

        print("Models initialized successfully!")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HAAR API - Hair Appearance and Rendering",
        "version": "1.0.0",
        "endpoints": {
            "/infer": "Generate hairstyles from text description",
            "/interpolate": "Interpolate between two hairstyle descriptions",
            "/image2hairstyle": "Generate hairstyle from image",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": model_ema is not None}

@app.post("/infer")
async def infer_hairstyle(request: InferenceRequest):
    """Generate hairstyles from text description"""
    try:
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)

        # Create unique experiment name
        exp_name = f"api_infer_{uuid.uuid4().hex[:8]}"
        save_path = "./api_results"
        os.makedirs(os.path.join(save_path, exp_name, 'upsampled_hairstyle'), exist_ok=True)
        os.makedirs(os.path.join(save_path, exp_name, 'guiding'), exist_ok=True)

        # Setup upsampler if needed
        upsampler = None
        if request.save_upsampled_hairstyle:
            upsampler = HairstyleUpsampler(
                **config['upsampler'],
                resolution=request.upsample_resolution,
                hairstyle_dataset=hairstyle_dataset
            )

        # Generate text embedding
        cross_cond = obtain_description_embedding(
            hairstyle_description=request.hairstyle_description,
            average_descriptions=True,
            model_feature_extractor=model_feature_extractor,
            txt_processors=txt_processors
        )

        # Setup sampling parameters
        noise = torch.randn(
            request.n_samples,
            config['dataset']['desc_size'],
            config['dataset']['patch_size'],
            config['dataset']['patch_size']
        ).cuda()

        sigma = torch.tensor([config['model']['sigma_max']], device=device)
        noised_input = noise * sigma

        sigmas = K.sampling.get_sigmas_karras(
            request.step,
            config['model']['sigma_min'],
            config['model']['sigma_max'],
            rho=7.,
            device=device
        )

        extra_args = {
            'cross_cond': cross_cond,
            'cross_cond_zero': torch.zeros(
                cross_cond.shape[0], 1, config['model']['context_dim'],
                device=cross_cond.device
            )
        }

        # Generate samples
        latent_textures = []
        generated_files = []

        for sample in range(request.n_samples):
            # Denoise texture
            x_0 = sample_euler_ancestral(
                model_ema,
                noised_input[sample:sample+1],
                sigmas,
                extra_args=extra_args,
                cfg_scale=request.cfg_scale,
                disable=not accelerator.is_main_process,
                seed=request.seed
            )
            x_0 = accelerator.gather(x_0)
            latent_textures.append(x_0)

            # Decode texture into strands
            strands = hairstyle_dataset.texture2strands(x_0)

            # Save guiding strands
            if request.save_guiding_strands:
                cols = torch.cat((
                    torch.rand(strands[0].shape[0], 3).unsqueeze(1).repeat(1, 100, 1),
                    torch.ones(strands[0].shape[0], 100, 1)
                ), dim=-1).reshape(-1, 4).cpu()

                pc_path = os.path.join(save_path, exp_name, 'guiding', f'pc_{sample}.ply')
                _ = trimesh.PointCloud(
                    strands[0].reshape(-1, 3).detach().cpu(),
                    colors=cols
                ).export(pc_path)
                generated_files.append(pc_path)

            # Save upsampled hairstyle
            if request.save_upsampled_hairstyle and upsampler is not None:
                upsampled_pc = upsampler(x_0)

                cols = torch.cat((
                    torch.rand(upsampled_pc.shape[0], 3).unsqueeze(1).repeat(1, 100, 1),
                    torch.ones(upsampled_pc.shape[0], 100, 1)
                ), dim=-1).reshape(-1, 4).cpu()

                pc_path = os.path.join(save_path, exp_name, 'upsampled_hairstyle', f'pc_{sample}.ply')
                _ = trimesh.PointCloud(
                    upsampled_pc.reshape(-1, 3).detach().cpu(),
                    colors=cols
                ).export(pc_path)
                generated_files.append(pc_path)

        # Save latent textures
        if request.save_latent_textures:
            texture_path = os.path.join(save_path, exp_name, 'exp_textures.pt')
            torch.save(torch.stack(latent_textures), texture_path)
            generated_files.append(texture_path)

        # Create zip file with results
        zip_path = os.path.join(save_path, f"{exp_name}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in generated_files:
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))

        return {
            "status": "success",
            "exp_name": exp_name,
            "generated_files": len(generated_files),
            "download_url": f"/download/{exp_name}_results.zip"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/interpolate")
async def interpolate_hairstyles(request: TextInterpolationRequest):
    """Interpolate between two hairstyle descriptions"""
    try:
        torch.manual_seed(request.seed)

        # Create unique experiment name
        exp_name = f"api_interpolate_{uuid.uuid4().hex[:8]}"
        save_path = "./api_results"
        os.makedirs(os.path.join(save_path, exp_name, 'upsampled_hairstyle'), exist_ok=True)
        os.makedirs(os.path.join(save_path, exp_name, 'guiding'), exist_ok=True)

        # Setup upsampler if needed
        upsampler = None
        if request.save_upsampled_hairstyle:
            upsampler = HairstyleUpsampler(
                **config['upsampler'],
                resolution=request.upsample_resolution,
                hairstyle_dataset=hairstyle_dataset
            )

        # Obtain text embeddings
        emb_hairstyle_1 = obtain_description_embedding(
            hairstyle_description=request.hairstyle_1,
            average_descriptions=False,
            model_feature_extractor=model_feature_extractor,
            txt_processors=txt_processors
        )

        emb_hairstyle_2 = obtain_description_embedding(
            hairstyle_description=request.hairstyle_2,
            average_descriptions=False,
            model_feature_extractor=model_feature_extractor,
            txt_processors=txt_processors
        )

        # Setup interpolation weights
        weights = torch.linspace(0, 1, request.n_interpolation_states).cuda()

        # Create noise
        noise = torch.randn(
            1, config['dataset']['desc_size'],
            config['dataset']['patch_size'],
            config['dataset']['patch_size']
        ).cuda()

        sigma = torch.tensor([request.degree], device=device)
        noised_input = noise * sigma

        sigmas = K.sampling.get_sigmas_karras(
            request.step,
            config['model']['sigma_min'],
            request.degree,
            rho=7.,
            device=device
        )

        # Generate interpolated samples
        latent_textures = []
        generated_files = []

        for sample in range(weights.shape[0]):
            # Interpolate embeddings
            cross_cond = (1-weights[sample]) * emb_hairstyle_1 + weights[sample] * emb_hairstyle_2

            extra_args = {
                'cross_cond': cross_cond,
                'cross_cond_zero': torch.zeros(
                    cross_cond.shape[0], 1, config['model']['context_dim'],
                    device=cross_cond.device
                )
            }

            # Denoise texture
            x_0 = sample_euler_ancestral(
                model_ema,
                noised_input,
                sigmas,
                extra_args=extra_args,
                cfg_scale=request.cfg_scale,
                disable=not accelerator.is_main_process,
                seed=request.seed
            )
            x_0 = accelerator.gather(x_0)
            latent_textures.append(x_0)

            # Decode texture into strands
            strands = hairstyle_dataset.texture2strands(x_0)

            # Save guiding strands
            if request.save_guiding_strands:
                cols = torch.cat((
                    torch.rand(strands[0].shape[0], 3).unsqueeze(1).repeat(1, 100, 1),
                    torch.ones(strands[0].shape[0], 100, 1)
                ), dim=-1).reshape(-1, 4).cpu()

                pc_path = os.path.join(save_path, exp_name, 'guiding', f'pc_{sample}.ply')
                _ = trimesh.PointCloud(
                    strands[0].reshape(-1, 3).detach().cpu(),
                    colors=cols
                ).export(pc_path)
                generated_files.append(pc_path)

            # Save upsampled hairstyle
            if request.save_upsampled_hairstyle and upsampler is not None:
                upsampled_pc = upsampler(x_0)

                cols = torch.cat((
                    torch.rand(upsampled_pc.shape[0], 3).unsqueeze(1).repeat(1, 100, 1),
                    torch.ones(upsampled_pc.shape[0], 100, 1)
                ), dim=-1).reshape(-1, 4).cpu()

                pc_path = os.path.join(save_path, exp_name, 'upsampled_hairstyle', f'pc_{sample}.ply')
                _ = trimesh.PointCloud(
                    upsampled_pc.reshape(-1, 3).detach().cpu(),
                    colors=cols
                ).export(pc_path)
                generated_files.append(pc_path)

        # Save latent textures
        if request.save_latent_textures:
            texture_path = os.path.join(save_path, exp_name, 'interpolation_textures.pt')
            torch.save(torch.stack(latent_textures), texture_path)
            generated_files.append(texture_path)

        # Create zip file with results
        zip_path = os.path.join(save_path, f"{exp_name}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in generated_files:
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))

        return {
            "status": "success",
            "exp_name": exp_name,
            "generated_files": len(generated_files),
            "download_url": f"/download/{exp_name}_results.zip"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpolation failed: {str(e)}")

@app.post("/image2hairstyle")
async def image_to_hairstyle(
    image: UploadFile = File(...),
    request: ImageToHairstyleRequest = None
):
    """Generate hairstyle from uploaded image"""
    if request is None:
        request = ImageToHairstyleRequest()

    try:
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)

        # Save uploaded image temporarily
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, image.filename)

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Create unique experiment name
        exp_name = f"api_image2hairstyle_{uuid.uuid4().hex[:8]}"
        save_path = "./api_results"
        os.makedirs(os.path.join(save_path, exp_name, 'upsampled_hairstyle'), exist_ok=True)
        os.makedirs(os.path.join(save_path, exp_name, 'guiding'), exist_ok=True)

        # Setup upsampler if needed
        upsampler = None
        if request.save_upsampled_hairstyle:
            upsampler = HairstyleUpsampler(
                **config['upsampler'],
                resolution=request.upsample_resolution,
                hairstyle_dataset=hairstyle_dataset
            )

        # Get description from image using LLaVA
        # This would require implementing the image2hairstyle functionality
        # For now, we'll use a placeholder
        hairstyle_description = "a woman with medium length hairstyle"  # Placeholder

        # Generate text embedding
        cross_cond = obtain_description_embedding(
            hairstyle_description=hairstyle_description,
            average_descriptions=True,
            model_feature_extractor=model_feature_extractor,
            txt_processors=txt_processors
        )

        # Setup sampling parameters
        noise = torch.randn(
            request.n_samples,
            config['dataset']['desc_size'],
            config['dataset']['patch_size'],
            config['dataset']['patch_size']
        ).cuda()

        sigma = torch.tensor([config['model']['sigma_max']], device=device)
        noised_input = noise * sigma

        sigmas = K.sampling.get_sigmas_karras(
            request.step,
            config['model']['sigma_min'],
            config['model']['sigma_max'],
            rho=7.,
            device=device
        )

        extra_args = {
            'cross_cond': cross_cond,
            'cross_cond_zero': torch.zeros(
                cross_cond.shape[0], 1, config['model']['context_dim'],
                device=cross_cond.device
            )
        }

        # Generate samples
        latent_textures = []
        generated_files = []

        for sample in range(request.n_samples):
            # Denoise texture
            x_0 = sample_euler_ancestral(
                model_ema,
                noised_input[sample:sample+1],
                sigmas,
                extra_args=extra_args,
                cfg_scale=request.cfg_scale,
                disable=not accelerator.is_main_process,
                seed=request.seed
            )
            x_0 = accelerator.gather(x_0)
            latent_textures.append(x_0)

            # Decode texture into strands
            strands = hairstyle_dataset.texture2strands(x_0)

            # Save guiding strands
            if request.save_guiding_strands:
                cols = torch.cat((
                    torch.rand(strands[0].shape[0], 3).unsqueeze(1).repeat(1, 100, 1),
                    torch.ones(strands[0].shape[0], 100, 1)
                ), dim=-1).reshape(-1, 4).cpu()

                pc_path = os.path.join(save_path, exp_name, 'guiding', f'pc_{sample}.ply')
                _ = trimesh.PointCloud(
                    strands[0].reshape(-1, 3).detach().cpu(),
                    colors=cols
                ).export(pc_path)
                generated_files.append(pc_path)

            # Save upsampled hairstyle
            if request.save_upsampled_hairstyle and upsampler is not None:
                upsampled_pc = upsampler(x_0)

                cols = torch.cat((
                    torch.rand(upsampled_pc.shape[0], 3).unsqueeze(1).repeat(1, 100, 1),
                    torch.ones(upsampled_pc.shape[0], 100, 1)
                ), dim=-1).reshape(-1, 4).cpu()

                pc_path = os.path.join(save_path, exp_name, 'upsampled_hairstyle', f'pc_{sample}.ply')
                _ = trimesh.PointCloud(
                    upsampled_pc.reshape(-1, 3).detach().cpu(),
                    colors=cols
                ).export(pc_path)
                generated_files.append(pc_path)

        # Save latent textures
        if request.save_latent_textures:
            texture_path = os.path.join(save_path, exp_name, 'exp_textures.pt')
            torch.save(torch.stack(latent_textures), texture_path)
            generated_files.append(texture_path)

        # Create zip file with results
        zip_path = os.path.join(save_path, f"{exp_name}_results.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in generated_files:
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))

        # Clean up temporary files
        shutil.rmtree(temp_dir)

        return {
            "status": "success",
            "exp_name": exp_name,
            "generated_files": len(generated_files),
            "download_url": f"/download/{exp_name}_results.zip",
            "note": "Image-to-hairstyle functionality is partially implemented"
        }

    except Exception as e:
        # Clean up temporary files on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Image-to-hairstyle failed: {str(e)}")

@app.get("/download/{filename}")
async def download_results(filename: str):
    """Download generated results"""
    file_path = os.path.join("./api_results", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/zip'
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)