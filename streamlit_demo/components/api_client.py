"""
API client for CurioScan Streamlit demo.
"""

import requests
import json
import time
from typing import Dict, Any, Optional, BinaryIO
import streamlit as st


class CurioScanAPIClient:
    """Client for interacting with CurioScan API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "CurioScan-Streamlit-Demo/2.0.0"
        })
    
    def upload_file(self, file, confidence_threshold: float = 0.8) -> str:
        """Upload a file for processing."""
        
        try:
            # Prepare file for upload
            files = {
                'file': (file.name, file.getvalue(), file.type)
            }
            
            params = {
                'confidence_threshold': confidence_threshold
            }
            
            # Remove Content-Type header for file upload
            headers = {k: v for k, v in self.session.headers.items() if k != "Content-Type"}
            
            response = requests.post(
                f"{self.base_url}/api/v1/upload",
                files=files,
                params=params,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get('job_id')
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Upload failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job processing status."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/status/{job_id}",
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Status check failed: {str(e)}")
    
    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get job processing results."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/jobs/{job_id}",
                timeout=30
            )
            
            response.raise_for_status()
            job_data = response.json()
            
            return job_data.get('result', {})
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Results retrieval failed: {str(e)}")
    
    def get_download_url(self, job_id: str, format_type: str = "json", 
                        include_provenance: bool = True) -> str:
        """Get download URL for results."""
        
        params = {
            'format': format_type,
            'include_provenance': include_provenance
        }
        
        return f"{self.base_url}/api/v1/result/{job_id}?" + "&".join([
            f"{k}={v}" for k, v in params.items()
        ])
    
    def trigger_retraining(self, model_type: str, dataset_path: str, 
                          config_overrides: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """Trigger model retraining."""
        
        try:
            payload = {
                'model_type': model_type,
                'dataset_path': dataset_path,
                'config_overrides': config_overrides,
                'dry_run': dry_run
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/retrain-trigger",
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Retraining trigger failed: {str(e)}")
    
    def get_review_items(self, job_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """Get items requiring review."""
        
        try:
            params = {
                'page': page,
                'page_size': page_size
            }
            
            response = self.session.get(
                f"{self.base_url}/api/v1/review/{job_id}",
                params=params,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Review items retrieval failed: {str(e)}")
    
    def update_review_items(self, updates: list) -> Dict[str, Any]:
        """Update review items with corrections."""
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/review/update",
                json=updates,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Review update failed: {str(e)}")
    
    def register_webhook(self, url: str, events: list, secret: str = None) -> Dict[str, Any]:
        """Register a webhook."""
        
        try:
            payload = {
                'url': url,
                'events': events,
                'secret': secret
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/webhooks/register",
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Webhook registration failed: {str(e)}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def list_jobs(self, status: str = None, limit: int = 50) -> Dict[str, Any]:
        """List processing jobs."""
        
        try:
            params = {'limit': limit}
            if status:
                params['status'] = status
            
            response = self.session.get(
                f"{self.base_url}/api/v1/jobs",
                params=params,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Jobs listing failed: {str(e)}")
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a processing job."""
        
        try:
            response = self.session.delete(
                f"{self.base_url}/api/v1/jobs/{job_id}",
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Job cancellation failed: {str(e)}")
    
    def get_model_checkpoints(self) -> Dict[str, Any]:
        """Get available model checkpoints."""
        
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/models/checkpoints",
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Checkpoints retrieval failed: {str(e)}")
    
    def activate_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Activate a model checkpoint."""
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/models/activate/{checkpoint_id}",
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Checkpoint activation failed: {str(e)}")


# Utility functions for Streamlit integration

@st.cache_data(ttl=60)
def get_cached_job_status(api_client: CurioScanAPIClient, job_id: str) -> Dict[str, Any]:
    """Get cached job status (refreshes every 60 seconds)."""
    return api_client.get_job_status(job_id)


@st.cache_data(ttl=300)
def get_cached_system_health(api_client: CurioScanAPIClient) -> Dict[str, Any]:
    """Get cached system health (refreshes every 5 minutes)."""
    return api_client.get_system_health()


def poll_job_completion(api_client: CurioScanAPIClient, job_id: str, 
                       progress_callback=None, max_wait_time: int = 600) -> Dict[str, Any]:
    """
    Poll for job completion with progress updates.
    
    Args:
        api_client: API client instance
        job_id: Job ID to poll
        progress_callback: Optional callback for progress updates
        max_wait_time: Maximum time to wait in seconds
        
    Returns:
        Final job status
    """
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            status = api_client.get_job_status(job_id)
            
            if progress_callback:
                progress_callback(status)
            
            job_status = status.get('status', 'unknown')
            
            if job_status in ['completed', 'failed', 'cancelled']:
                return status
            
            time.sleep(2)  # Poll every 2 seconds
            
        except Exception as e:
            st.error(f"Error polling job status: {str(e)}")
            break
    
    raise TimeoutError(f"Job {job_id} did not complete within {max_wait_time} seconds")
