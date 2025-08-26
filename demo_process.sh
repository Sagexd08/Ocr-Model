#!/bin/bash

# This script provides a simple end-to-end demonstration of the CurioScan system.

# 1. Start the services
docker-compose up -d

# 2. Wait for the services to be ready
./wait-for-it.sh localhost:8000 -t 60
./wait-for-it.sh localhost:8501 -t 60

# 3. Upload a document for processing
JOB_ID=$(curl -s -X POST http://localhost:8000/upload | jq -r .job_id)

# 4. Check the status of the job until it is complete
STATUS=""
while [ "$STATUS" != "SUCCESS" ]
do
  STATUS=$(curl -s http://localhost:8000/status/$JOB_ID | jq -r .status)
  sleep 1
done

# 5. Get the result
curl -s http://localhost:8000/result/$JOB_ID

# 6. Stop the services
docker-compose down