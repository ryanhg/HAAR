#!/bin/bash
# test_haar_api.sh

echo "🧪 Testing HAAR API..."

# Make the API request
echo "�� Making API request..."
RESPONSE=$(curl -s -X POST "https://ftunvuix1abd5v-8000.proxy.runpod.net/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "hairstyle_description": "a woman with medium length wavy brown hair",
    "cfg_scale": 1.2,
    "n_samples": 1,
    "step": 50,
    "save_guiding_strands": true
  }')

echo "📊 Response: $RESPONSE"

# Extract download URL using jq (if available) or grep
if command -v jq &> /dev/null; then
    DOWNLOAD_URL=$(echo $RESPONSE | jq -r '.download_url')
    EXP_NAME=$(echo $RESPONSE | jq -r '.exp_name')
else
    DOWNLOAD_URL=$(echo $RESPONSE | grep -o '"/download/[^"]*"' | tr -d '"')
    EXP_NAME=$(echo $RESPONSE | grep -o '"exp_name":"[^"]*"' | cut -d'"' -f4)
fi

if [ ! -z "$DOWNLOAD_URL" ]; then
    echo "⬇️  Downloading results..."
    curl -O "https://oo0wzzt1rikhev-8000.proxy.runpod.net$DOWNLOAD_URL"
    echo "✅ Download completed!"
    echo "�� File: ${EXP_NAME}_results.zip"
else
    echo "❌ No download URL found in response"
    exit 1
fi
