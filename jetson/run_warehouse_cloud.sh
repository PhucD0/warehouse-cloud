#!/bin/bash
cd ~
export WAREHOUSE_CLOUD_URL=https://warehouse-cloud.onrender.com
export WAREHOUSE_CLOUD_TIMEOUT=8

echo "[RUN] WAREHOUSE_CLOUD_URL=$WAREHOUSE_CLOUD_URL"
curl -4 --tlsv1.2 -s https://warehouse-cloud.onrender.com/api/health
echo

python3 warehouse.py
