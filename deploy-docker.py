#!/usr/bin/env python3
"""
Docker Deployment Script for Enterprise OCR Processing System
Automates the complete Docker deployment process
"""

import subprocess
import time
import sys
import os
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command with proper error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def check_docker():
    """Check if Docker is installed and running"""
    print("🐳 Checking Docker installation...")
    
    # Check Docker version
    if not run_command("docker --version", "Checking Docker version"):
        print("❌ Docker is not installed or not in PATH")
        return False
    
    # Check Docker Compose
    if not run_command("docker-compose --version", "Checking Docker Compose version"):
        print("❌ Docker Compose is not installed or not in PATH")
        return False
    
    # Check if Docker daemon is running
    if not run_command("docker info", "Checking Docker daemon", check=False):
        print("❌ Docker daemon is not running. Please start Docker Desktop.")
        return False
    
    print("✅ Docker environment is ready")
    return True

def cleanup_existing():
    """Clean up existing containers and images"""
    print("🧹 Cleaning up existing containers...")
    
    # Stop and remove containers
    run_command("docker-compose down", "Stopping existing containers", check=False)
    
    # Remove old images (optional)
    run_command("docker system prune -f", "Cleaning up Docker system", check=False)
    
    print("✅ Cleanup completed")

def build_images():
    """Build Docker images"""
    print("🏗️ Building Docker images...")
    
    # Build with no cache for fresh build
    success = run_command(
        "docker-compose build --no-cache --parallel",
        "Building all Docker images"
    )
    
    if success:
        print("✅ All Docker images built successfully")
        return True
    else:
        print("❌ Docker build failed")
        return False

def start_services():
    """Start all Docker services"""
    print("🚀 Starting Docker services...")
    
    success = run_command(
        "docker-compose up -d",
        "Starting all services in detached mode"
    )
    
    if success:
        print("✅ All services started successfully")
        return True
    else:
        print("❌ Failed to start services")
        return False

def wait_for_services():
    """Wait for services to be ready"""
    print("⏳ Waiting for services to be ready...")
    
    max_wait = 120  # 2 minutes
    wait_interval = 10
    waited = 0
    
    while waited < max_wait:
        print(f"⏳ Waiting... ({waited}/{max_wait}s)")
        time.sleep(wait_interval)
        waited += wait_interval
        
        # Check if main service is responding
        if run_command(
            "curl -f http://localhost:8505 > /dev/null 2>&1",
            "Checking Streamlit service",
            check=False
        ):
            print("✅ Services are ready!")
            return True
    
    print("⚠️ Services may still be starting. Check manually.")
    return False

def show_service_status():
    """Show status of all services"""
    print("📊 Service Status:")
    print("=" * 50)
    
    # Show running containers
    run_command("docker-compose ps", "Showing container status")
    
    print("\n🌐 Access URLs:")
    print("- Streamlit UI: http://localhost:8505")
    print("- FastAPI Docs: http://localhost:8001/docs")
    print("- Nginx Proxy: http://localhost:80")
    
    print("\n📋 Useful Commands:")
    print("- View logs: docker-compose logs -f")
    print("- Stop services: docker-compose down")
    print("- Restart services: docker-compose restart")

def main():
    """Main deployment function"""
    print("🚀 Enterprise OCR Docker Deployment")
    print("=" * 50)
    
    # Check prerequisites
    if not check_docker():
        print("❌ Docker environment check failed")
        sys.exit(1)
    
    # Cleanup existing deployment
    cleanup_existing()
    
    # Build images
    if not build_images():
        print("❌ Build failed")
        sys.exit(1)
    
    # Start services
    if not start_services():
        print("❌ Service startup failed")
        sys.exit(1)
    
    # Wait for services
    wait_for_services()
    
    # Show status
    show_service_status()
    
    print("\n🎉 Docker deployment completed successfully!")
    print("🌐 Open http://localhost:8505 to access the Enterprise OCR System")
    
    # Ask if user wants to run tests
    try:
        response = input("\n🧪 Would you like to run the test suite? (y/n): ")
        if response.lower() in ['y', 'yes']:
            print("🧪 Running test suite...")
            subprocess.run([sys.executable, "docker-test.py"])
    except KeyboardInterrupt:
        print("\n👋 Deployment completed. Tests skipped.")

if __name__ == "__main__":
    main()
