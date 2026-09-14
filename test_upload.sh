#!/usr/bin/env zsh
set -euo pipefail

FILE="./paper.pdf"
BASE="http://localhost:8000"

# Initialize session and save cookies
# curl -s -c cookies.txt \
#   -X POST "$BASE/api/v1/session/init"

# echo

# Generate presigned URL
curl -s -b cookies.txt \
  -X POST "$BASE/api/v1/upload/generate-presigned-url" \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "paper.pdf",
    "content_type": "application/pdf"
  }' \
  | tee presign.json

# Extract upload URL and corpus ID
read -r UPLOAD_URL CORPUS_ID < <(
  python3 -c '
import json

d = json.load(open("presign.json"))

upload_url = d["upload_url"]
corpus_id = d["s3_key"].split("/")[1]

print(upload_url, corpus_id)
'
)

echo "Corpus ID: $CORPUS_ID"
echo "Uploading: $FILE"

# Upload file
curl -v -X PUT "$UPLOAD_URL" \
  -H "Content-Type: application/pdf" \
  -H "x-amz-meta-corpus_id: $CORPUS_ID" \
  -H "x-amz-meta-original_filename: $(basename "$FILE")" \
  --upload-file "$FILE"

echo
echo "Upload successful."