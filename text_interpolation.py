node = hou.pwd()
geo = node.geometry()

"""
HAAR Text Interpolation Generator
A clean script for interpolating between two hairstyle descriptions using the HAAR API.
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

# Hairstyle Descriptions
HAIRSTYLE_1 = node.parm('hairstyle_1').eval()
HAIRSTYLE_2 = node.parm('hairstyle_2').eval()

# Interpolation Parameters
N_INTERPOLATION_STATES = node.parm('n_samples').eval()
CFG_SCALE = node.parm('cfg_scale').eval()
STEP = node.parm('steps').eval()
DEGREE = node.parm('degree').eval()
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

def interpolate_hairstyles():
    """Interpolate between two hairstyle descriptions using HAAR API."""

    # Prepare request payload
    payload = {
        "hairstyle_1": HAIRSTYLE_1,
        "hairstyle_2": HAIRSTYLE_2,
        "n_interpolation_states": N_INTERPOLATION_STATES,
        "cfg_scale": CFG_SCALE,
        "step": STEP,
        "seed": SEED,
        "degree": DEGREE,
        "save_guiding_strands": SAVE_GUIDING_STRANDS,
        "save_upsampled_hairstyle": SAVE_UPSAMPLED_HAIRSTYLE,
        "save_latent_textures": SAVE_LATENT_TEXTURES,
        "upsample_resolution": UPSAMPLE_RESOLUTION
    }

    # Make API request
    try:
        print(f"Interpolating between hairstyles:")
        print(f"  From: '{HAIRSTYLE_1}'")
        print(f"  To:   '{HAIRSTYLE_2}'")
        print(f"  Steps: {N_INTERPOLATION_STATES}")

        response = requests.post(f"{API_BASE_URL}{API_ENDPOINT}", json=payload)
        response.raise_for_status()

        result = response.json()

        if result["status"] == "success":
            print(f"✓ Interpolation successful! Experiment: {result['exp_name']}")
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
            print(f"✗ Interpolation failed: {result}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"✗ API request failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    result_path = interpolate_hairstyles()
    if result_path:
        print(f"\n🎉 Hairstyle interpolation completed successfully!")
        print(f"📁 Results available at: {result_path}")
    else:
        print(f"\n❌ Hairstyle interpolation failed!")