# HAAR FastAPI

A FastAPI wrapper for HAAR (Hair Appearance and Rendering) that provides REST API endpoints for generating 3D hairstyles from text descriptions.

## Features

- **Text-to-Hairstyle Generation**: Generate 3D hairstyles from text descriptions
- **Text Interpolation**: Interpolate between two hairstyle descriptions
- **Image-to-Hairstyle**: Generate hairstyles from uploaded images (partially implemented)
- **Model Caching**: Models are loaded once and cached for faster inference
- **File Downloads**: Results are packaged as ZIP files for easy download

## Installation

1. **Setup HAAR Environment**: Follow the main HAAR installation instructions first
2. **Install API Dependencies**:
   ```bash
   pip install -r requirements_api.txt
   ```
3. **Download Pretrained Models**: Ensure you have the required model files:
   - `./pretrained_models/haar_prior/haar_diffusion.pth`
   - `./pretrained_models/strand_prior/strand_ckpt.pth`

## Quick Start

### Start the API Server

```bash
# Option 1: Use the startup script
./start_api.sh

# Option 2: Manual startup
conda activate haar
python api.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

```bash
GET /health
```

Returns the health status and whether models are loaded.

### 2. Text-to-Hairstyle Generation

```bash
POST /infer
```

**Request Body:**
```json
{
  "hairstyle_description": "a woman with long straight hairstyle",
  "cfg_scale": 1.2,
  "n_samples": 1,
  "step": 50,
  "seed": null,
  "save_guiding_strands": true,
  "save_upsampled_hairstyle": false,
  "save_latent_textures": false,
  "upsample_resolution": 64
}
```

**Parameters:**
- `hairstyle_description` (required): Text description of the desired hairstyle
- `cfg_scale` (default: 1.2): Classifier-free guidance weight
- `n_samples` (default: 1): Number of variations to generate
- `step` (default: 50): Number of denoising steps
- `seed` (optional): Random seed for reproducible results
- `save_guiding_strands` (default: true): Save guiding strand point clouds
- `save_upsampled_hairstyle` (default: false): Save full upsampled hairstyles
- `save_latent_textures` (default: false): Save latent texture representations
- `upsample_resolution` (default: 64): Resolution for upsampling (64, 128, or 256)

**Response:**
```json
{
  "status": "success",
  "exp_name": "api_infer_abc12345",
  "generated_files": 1,
  "download_url": "/download/api_infer_abc12345_results.zip"
}
```

### 3. Text Interpolation

```bash
POST /interpolate
```

**Request Body:**
```json
{
  "hairstyle_1": "straight woman hairstyle",
  "hairstyle_2": "long wavy haircut",
  "n_interpolation_states": 5,
  "cfg_scale": 1.5,
  "step": 50,
  "seed": 32,
  "degree": 80.0,
  "save_guiding_strands": true,
  "save_upsampled_hairstyle": false,
  "save_latent_textures": false,
  "upsample_resolution": 128
}
```

**Parameters:**
- `hairstyle_1` (required): First hairstyle description
- `hairstyle_2` (required): Second hairstyle description
- `n_interpolation_states` (default: 5): Number of interpolation steps
- `degree` (default: 80.0): Noise level for generation
- Other parameters same as `/infer` endpoint

### 4. Image-to-Hairstyle

```bash
POST /image2hairstyle
```

**Request:**
- Upload an image file
- Optional JSON body with parameters (same as `/infer`)

**Note:** This endpoint is partially implemented and currently uses a placeholder description.

### 5. Download Results

```bash
GET /download/{filename}
```

Download the generated results as a ZIP file.

## Usage Examples

### Python Client Example

```python
import requests
import json

# Generate hairstyle from text
url = "http://localhost:8000/infer"
data = {
    "hairstyle_description": "a woman with short curly hairstyle",
    "cfg_scale": 1.5,
    "n_samples": 3,
    "save_upsampled_hairstyle": True,
    "upsample_resolution": 128
}

response = requests.post(url, json=data)
result = response.json()

if result["status"] == "success":
    # Download the results
    download_url = f"http://localhost:8000{result['download_url']}"
    download_response = requests.get(download_url)

    with open("hairstyle_results.zip", "wb") as f:
        f.write(download_response.content)
    print("Results downloaded successfully!")
```

### cURL Examples

**Basic Inference:**
```bash
curl -X POST "http://localhost:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "hairstyle_description": "a woman with long straight hairstyle",
       "cfg_scale": 1.2,
       "n_samples": 1
     }'
```

**Text Interpolation:**
```bash
curl -X POST "http://localhost:8000/interpolate" \
     -H "Content-Type: application/json" \
     -d '{
       "hairstyle_1": "straight woman hairstyle",
       "hairstyle_2": "long wavy haircut",
       "n_interpolation_states": 5,
       "cfg_scale": 1.5
     }'
```

**Image Upload:**
```bash
curl -X POST "http://localhost:8000/image2hairstyle" \
     -F "image=@path/to/your/image.jpg" \
     -F 'request={"cfg_scale": 1.2, "n_samples": 1}'
```

## Output Files

The API generates the following file types:

1. **Guiding Strands** (`.ply`): Point cloud files containing the guiding strands
2. **Upsampled Hairstyles** (`.ply`): Full upsampled hairstyle point clouds
3. **Latent Textures** (`.pt`): PyTorch tensors containing latent representations

All files are packaged in a ZIP archive for easy download.

## Configuration

The API uses the same configuration as the command-line interface:
- Model configuration: `./configs/infer.json`
- Pretrained model: `./pretrained_models/haar_prior/haar_diffusion.pth`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `n_samples` or `upsample_resolution`
2. **Model Loading Errors**: Ensure all pretrained models are downloaded
3. **Import Errors**: Make sure the HAAR environment is properly set up

### Logs

The API logs are printed to the console. Check for:
- Model initialization messages
- Inference progress
- Error messages

### Performance Tips

- Use smaller `upsample_resolution` for faster generation
- Reduce `n_samples` for quicker results
- The API caches models, so subsequent requests are faster

## Development

### Adding New Endpoints

1. Define a Pydantic model for the request
2. Create the endpoint function
3. Add proper error handling
4. Update this documentation

### Model Caching

Models are loaded once on startup and cached globally. To reload models, restart the API server.

## License

This API follows the same license as the main HAAR project.