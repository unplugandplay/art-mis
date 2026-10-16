#!/bin/sh
# Download and unpack FB15k-237 (Toutanova & Chen, 2015) into data/fb15k237/.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
URL="https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip"
echo "downloading $URL"
curl -fL --retry 3 -o data/fb15k237.zip "$URL"
unzip -o data/fb15k237.zip -d data/fb15k237
# unpacks into Release/ — normalize layout
if [ -d data/fb15k237/Release ]; then
  mv data/fb15k237/Release/* data/fb15k237/
  rmdir data/fb15k237/Release
fi
echo "files:"; ls -la data/fb15k237
