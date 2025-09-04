#!/usr/bin/env python3
"""
Docker Test Script for Enterprise OCR Processing System
Tests the containerized application functionality
"""

import requests
import time
import sys
import subprocess
import json
from pathlib import Path

def test_docker_services():
    """Test if Docker services are running"""
    print("🐳 Testing Docker Services...")
    
    services = [
        ("Streamlit App", "http://localhost:8505"),
        ("FastAPI Backend", "http://localhost:8001/docs"),
        ("Nginx Proxy", "http://localhost:80")
    ]
    
    results = {}
    
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                results[service_name] = "✅ RUNNING"
                print(f"✅ {service_name}: RUNNING")
            else:
                results[service_name] = f"❌ ERROR ({response.status_code})"
                print(f"❌ {service_name}: ERROR ({response.status_code})")
        except requests.exceptions.RequestException as e:
            results[service_name] = f"❌ FAILED ({str(e)})"
            print(f"❌ {service_name}: FAILED ({str(e)})")
    
    return results

def test_docker_containers():
    """Test Docker container status"""
    print("\n🔍 Checking Docker Container Status...")
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check containers: {e}")
        return False

def test_streamlit_ui():
    """Test Streamlit UI functionality"""
    print("\n🎨 Testing Streamlit Dark Mode UI...")
    
    try:
        response = requests.get("http://localhost:8505", timeout=15)
        if response.status_code == 200:
            content = response.text
            
            # Check for dark mode elements
            dark_mode_indicators = [
                "#0E1117",  # Background color
                "#00D4FF",  # Primary color
                "dark",     # Theme
                "Enterprise OCR"  # Title
            ]
            
            found_indicators = []
            for indicator in dark_mode_indicators:
                if indicator in content:
                    found_indicators.append(indicator)
            
            print(f"✅ Streamlit UI loaded successfully")
            print(f"✅ Found {len(found_indicators)}/{len(dark_mode_indicators)} dark mode indicators")
            
            return True
        else:
            print(f"❌ Streamlit UI failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Streamlit UI test failed: {e}")
        return False

def test_api_endpoints():
    """Test FastAPI endpoints"""
    print("\n🔌 Testing API Endpoints...")
    
    endpoints = [
        "/health",
        "/docs",
        "/openapi.json"
    ]
    
    base_url = "http://localhost:8001"
    results = {}
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                results[endpoint] = "✅ OK"
                print(f"✅ {endpoint}: OK")
            else:
                results[endpoint] = f"❌ ERROR ({response.status_code})"
                print(f"❌ {endpoint}: ERROR ({response.status_code})")
        except requests.exceptions.RequestException as e:
            results[endpoint] = f"❌ FAILED ({str(e)})"
            print(f"❌ {endpoint}: FAILED ({str(e)})")
    
    return results

def generate_test_report(service_results, api_results):
    """Generate comprehensive test report"""
    print("\n📊 Test Report Summary")
    print("=" * 50)
    
    total_tests = len(service_results) + len(api_results)
    passed_tests = sum(1 for result in list(service_results.values()) + list(api_results.values()) 
                      if "✅" in result)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n🔍 Detailed Results:")
    print("\nServices:")
    for service, status in service_results.items():
        print(f"  {service}: {status}")
    
    print("\nAPI Endpoints:")
    for endpoint, status in api_results.items():
        print(f"  {endpoint}: {status}")
    
    # Save report to file
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
        "services": service_results,
        "api_endpoints": api_results
    }
    
    with open("docker-test-report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: docker-test-report.json")
    
    return passed_tests == total_tests

def main():
    """Main test function"""
    print("🚀 Enterprise OCR Docker Test Suite")
    print("=" * 50)
    
    # Wait for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Test Docker containers
    container_status = test_docker_containers()
    
    # Test services
    service_results = test_docker_services()
    
    # Test Streamlit UI
    ui_success = test_streamlit_ui()
    
    # Test API endpoints
    api_results = test_api_endpoints()
    
    # Generate report
    all_passed = generate_test_report(service_results, api_results)
    
    if all_passed and ui_success:
        print("\n🎉 All tests passed! Docker deployment successful!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
