node = hou.pwd()
geo = node.geometry()

"""
HAAR Image-to-Hairstyle Generator
A clean script for generating 3D hairstyles from uploaded images using the HAAR API.
"""

import requests
import json
import os
import zipfile
import shutil
import tempfile
from pathlib import Path

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# API Configuration
API_BASE_URL = node.parm('api_url').eval()
API_ENDPOINT = node.parm('api_endpoint').eval()

# Input Image Configuration
INPUT_IMAGE_PATH = node.parm('input_image_path').eval()

# Generation Parameters
CFG_SCALE = node.parm('cfg_scale').eval()
N_SAMPLES = node.parm('n_samples').eval()
STEP = node.parm('steps').eval()
SEED = node.parm('seed').eval() if node.parm('use_seed') else None

# Output Configuration
SAVE_GUIDING_STRANDS = True if node.parm('save_guiding_strands').eval() else False
SAVE_UPSAMPLED_HAIRSTYLE = True if node.parm('save_upsampled_hairstyle').eval() else False
SAVE_LATENT_TEXTURES = True if node.parm('save_latent_textures').eval() else False
UPSAMPLE_RESOLUTION = node.parm('upsample_res').menuLabels()[node.parm('upsample_res').eval()]

# File Output Configuration
OUTPUT_DIR = node.parm('output_dir').eval()
DOWNLOAD_FILENAME = node.parm('output_filename').eval() + '.zip'

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_hairstyle_from_image():
    """Generate hairstyle from uploaded image using HAAR API."""

    # Check if input image exists
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"✗ Input image not found: {INPUT_IMAGE_PATH}")
        return None

    # Prepare request data
    files = {
        'image': open(INPUT_IMAGE_PATH, 'rb')
    }

    data = {
        'request': json.dumps({
            "cfg_scale": CFG_SCALE,
            "n_samples": N_SAMPLES,
            "step": STEP,
            "seed": SEED,
            "save_guiding_strands": SAVE_GUIDING_STRANDS,
            "save_upsampled_hairstyle": SAVE_UPSAMPLED_HAIRSTYLE,
            "save_latent_textures": SAVE_LATENT_TEXTURES,
            "upsample_resolution": UPSAMPLE_RESOLUTION
        })
    }

    # Make API request
    try:
        print(f"Generating hairstyle from image: {INPUT_IMAGE_PATH}")
        print(f"Parameters: CFG={CFG_SCALE}, Samples={N_SAMPLES}, Steps={STEP}")

        response = requests.post(f"{API_BASE_URL}{API_ENDPOINT}", files=files, data=data)
        response.raise_for_status()

        result = response.json()

        if result["status"] == "success":
            print(f"✓ Generation successful! Experiment: {result['exp_name']}")
            print(f"✓ Generated {result['generated_files']} files")

            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()

            # Download results to temp directory
            download_url = f"{API_BASE_URL}{result['download_url']}"
            download_response = requests.get(download_url)
            download_response.raise_for_status()

            # Save zip file to temp directory
            temp_zip_path = os.path.join(temp_dir, DOWNLOAD_FILENAME)
            with open(temp_zip_path, "wb") as f:
                f.write(download_response.content)

            print(f"✓ Downloaded zip to temp directory")

            # Extract zip file in temp directory
            print("📦 Extracting results...")
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find the extracted folder in temp directory
            extracted_items = [item for item in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, item))]
            if extracted_items:
                extracted_folder = os.path.join(temp_dir, extracted_items[0])
                new_folder_name = DOWNLOAD_FILENAME.replace('.zip', '')

                # Ensure output directory exists
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                new_folder_path = os.path.join(OUTPUT_DIR, new_folder_name)

                # Remove existing folder if it exists
                if os.path.exists(new_folder_path):
                    shutil.rmtree(new_folder_path)
                    print(f"🗑️  Removed existing folder: {new_folder_path}")

                # Move the folder to final location
                shutil.move(extracted_folder, new_folder_path)
                print(f"📁 Moved folder to: {new_folder_path}")

                # Clean up temp directory (includes the zip file)
                shutil.rmtree(temp_dir)
                return new_folder_path
            else:
                print("⚠️  No folder found in extracted contents")
                shutil.rmtree(temp_dir)
                return OUTPUT_DIR

        else:
            print(f"✗ Generation failed: {result}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None
    finally:
        # Close the file
        files['image'].close()

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    result_path = generate_hairstyle_from_image()
    if result_path:
        print(f"\n🎉 Image-to-hairstyle generation completed successfully!")
        print(f"📁 Results available at: {result_path}")
        print(f"⚠️  Note: This feature is partially implemented and uses placeholder descriptions")
    else:
        print(f"\n❌ Image-to-hairstyle generation failed!")