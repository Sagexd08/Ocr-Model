#!/bin/bash

# CurioScan Demo Processing Script
# This script demonstrates end-to-end OCR processing with sample documents

set -e

echo "🚀 CurioScan Demo Processing Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="http://localhost:8000"
DEMO_DATA_DIR="data/demo"
OUTPUT_DIR="data/output"
SAMPLE_FILES=(
    "sample_invoice.pdf"
    "sample_table.png"
    "sample_form.jpg"
    "sample_document.docx"
)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if service is running
check_service() {
    local service_url=$1
    local service_name=$2
    
    print_status "Checking $service_name..."
    if curl -s -f "$service_url" > /dev/null; then
        print_success "$service_name is running"
        return 0
    else
        print_error "$service_name is not running"
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local service_url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$service_url" > /dev/null; then
            print_success "$service_name is ready"
            return 0
        fi
        
        print_status "Attempt $attempt/$max_attempts - waiting for $service_name..."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Function to upload and process file
process_file() {
    local file_path=$1
    local file_name=$(basename "$file_path")
    
    print_status "Processing $file_name..."
    
    # Upload file
    local upload_response=$(curl -s -X POST \
        -F "file=@$file_path" \
        "$API_BASE_URL/upload")
    
    if [ $? -ne 0 ]; then
        print_error "Failed to upload $file_name"
        return 1
    fi
    
    # Extract job ID
    local job_id=$(echo "$upload_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('job_id', ''))
except:
    print('')
")
    
    if [ -z "$job_id" ]; then
        print_error "Failed to get job ID for $file_name"
        echo "Response: $upload_response"
        return 1
    fi
    
    print_success "Uploaded $file_name with job ID: $job_id"
    
    # Poll for completion
    local max_wait=120  # 2 minutes
    local wait_time=0
    local status="pending"
    
    while [ "$status" != "completed" ] && [ "$status" != "failed" ] && [ $wait_time -lt $max_wait ]; do
        sleep 5
        wait_time=$((wait_time + 5))
        
        local status_response=$(curl -s "$API_BASE_URL/status/$job_id")
        status=$(echo "$status_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('status', 'unknown'))
except:
    print('unknown')
")
        
        print_status "Status: $status (${wait_time}s elapsed)"
    done
    
    if [ "$status" = "completed" ]; then
        print_success "Processing completed for $file_name"
        
        # Download results
        local output_file="$OUTPUT_DIR/${file_name%.*}_result.json"
        curl -s "$API_BASE_URL/result/$job_id" -o "$output_file"
        
        if [ -f "$output_file" ]; then
            print_success "Results saved to $output_file"
            
            # Show summary
            local row_count=$(python3 -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
        print(len(data.get('rows', [])))
except:
    print('0')
")
            print_status "Extracted $row_count rows from $file_name"
        else
            print_warning "Failed to download results for $file_name"
        fi
        
        return 0
    else
        print_error "Processing failed or timed out for $file_name (status: $status)"
        return 1
    fi
}

# Function to generate sample data if not exists
generate_sample_data() {
    print_status "Checking for sample data..."
    
    if [ ! -d "$DEMO_DATA_DIR" ]; then
        mkdir -p "$DEMO_DATA_DIR"
    fi
    
    # Create sample files if they don't exist
    if [ ! -f "$DEMO_DATA_DIR/sample_invoice.pdf" ]; then
        print_status "Generating sample invoice PDF..."
        python3 -c "
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Create a simple invoice PDF
c = canvas.Canvas('$DEMO_DATA_DIR/sample_invoice.pdf', pagesize=letter)
width, height = letter

# Header
c.setFont('Helvetica-Bold', 16)
c.drawString(50, height - 50, 'INVOICE')
c.setFont('Helvetica', 12)
c.drawString(50, height - 80, 'Invoice #: INV-001')
c.drawString(50, height - 100, 'Date: 2024-01-15')

# Company info
c.drawString(50, height - 140, 'From:')
c.drawString(50, height - 160, 'CurioScan Demo Company')
c.drawString(50, height - 180, '123 Demo Street')
c.drawString(50, height - 200, 'Demo City, DC 12345')

# Customer info
c.drawString(300, height - 140, 'To:')
c.drawString(300, height - 160, 'Demo Customer')
c.drawString(300, height - 180, '456 Customer Ave')
c.drawString(300, height - 200, 'Customer City, CC 67890')

# Table header
y = height - 250
c.drawString(50, y, 'Description')
c.drawString(300, y, 'Quantity')
c.drawString(400, y, 'Price')
c.drawString(500, y, 'Total')

# Table rows
items = [
    ('OCR Processing Service', '10', '$50.00', '$500.00'),
    ('Document Analysis', '5', '$75.00', '$375.00'),
    ('Table Extraction', '3', '$100.00', '$300.00')
]

y -= 30
for item in items:
    c.drawString(50, y, item[0])
    c.drawString(300, y, item[1])
    c.drawString(400, y, item[2])
    c.drawString(500, y, item[3])
    y -= 20

# Total
y -= 20
c.setFont('Helvetica-Bold', 12)
c.drawString(400, y, 'TOTAL: $1,175.00')

c.save()
print('Generated sample invoice PDF')
" 2>/dev/null || print_warning "Could not generate sample PDF (reportlab not installed)"
    fi
    
    # Create a simple text file as fallback
    if [ ! -f "$DEMO_DATA_DIR/sample_document.txt" ]; then
        cat > "$DEMO_DATA_DIR/sample_document.txt" << EOF
CurioScan Demo Document

This is a sample document for testing OCR capabilities.

Table Example:
Name        Age    City
John Doe    30     New York
Jane Smith  25     Los Angeles
Bob Johnson 35     Chicago

Key-Value Pairs:
Company: CurioScan Demo
Date: 2024-01-15
Status: Active
Revenue: $1,175.00

This document contains various text elements including:
- Headers and titles
- Structured tables
- Key-value pairs
- Regular paragraphs
- Numbers and dates

The OCR system should be able to extract all this information
with proper structure and provenance tracking.
EOF
        print_success "Generated sample text document"
    fi
}

# Main execution
main() {
    print_status "Starting CurioScan demo processing..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if services are running
    print_status "Checking required services..."
    
    if ! check_service "$API_BASE_URL/health" "CurioScan API"; then
        print_error "CurioScan API is not running. Please start it with:"
        print_error "  docker-compose up api"
        print_error "  OR"
        print_error "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
        exit 1
    fi
    
    # Generate sample data
    generate_sample_data
    
    # Process available files
    local processed=0
    local failed=0
    
    for file_name in "${SAMPLE_FILES[@]}"; do
        local file_path="$DEMO_DATA_DIR/$file_name"
        
        # Try alternative extensions if file doesn't exist
        if [ ! -f "$file_path" ]; then
            file_path="$DEMO_DATA_DIR/sample_document.txt"
        fi
        
        if [ -f "$file_path" ]; then
            if process_file "$file_path"; then
                ((processed++))
            else
                ((failed++))
            fi
        else
            print_warning "Sample file not found: $file_path"
        fi
    done
    
    # Summary
    echo ""
    print_status "Demo processing completed!"
    print_success "Successfully processed: $processed files"
    if [ $failed -gt 0 ]; then
        print_warning "Failed to process: $failed files"
    fi
    
    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR)" ]; then
        print_status "Results available in: $OUTPUT_DIR"
        print_status "Files:"
        ls -la "$OUTPUT_DIR"
    fi
    
    echo ""
    print_status "Next steps:"
    echo "  1. View results in the output directory"
    echo "  2. Open Streamlit demo: http://localhost:8501"
    echo "  3. Check API documentation: http://localhost:8000/docs"
    echo "  4. Monitor with Flower: http://localhost:5555"
}

# Run main function
main "$@"
