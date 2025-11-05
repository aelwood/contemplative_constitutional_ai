"""
SageMaker-specific utilities for Contemplative Constitutional AI.
Provides S3 sync, device detection, and path management for SageMaker environments.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import torch

logger = logging.getLogger(__name__)


def is_sagemaker_environment() -> bool:
    """
    Check if running in a SageMaker environment.
    
    Returns:
        True if in SageMaker, False otherwise
    """
    # SageMaker sets specific environment variables
    return (
        os.environ.get('SM_TRAINING_ENV') is not None or
        os.path.exists('/opt/ml') or
        'SageMaker' in os.getcwd()
    )


def detect_sagemaker_device() -> str:
    """
    Detect the best available device in SageMaker environment.
    Prioritizes CUDA over MPS (which won't be available in SageMaker anyway).
    
    Returns:
        Device string: 'cuda', 'cpu'
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA detected with {device_count} GPU(s): {device_name}")
        return 'cuda'
    else:
        logger.warning("No GPU acceleration available in SageMaker, using CPU")
        return 'cpu'


def get_s3_client():
    """
    Get boto3 S3 client with proper configuration.
    
    Returns:
        boto3 S3 client
    """
    try:
        return boto3.client('s3')
    except NoCredentialsError:
        logger.error("AWS credentials not found. Configure with AWS CLI or IAM role.")
        raise


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse S3 path into bucket and key.
    
    Args:
        s3_path: S3 path like 's3://bucket/path/to/file'
        
    Returns:
        Tuple of (bucket, key)
    """
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'")
    
    path_parts = s3_path[5:].split('/', 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ''
    
    return bucket, key


def sync_to_s3(
    local_path: Union[str, Path],
    s3_path: str,
    overwrite: bool = True
) -> bool:
    """
    Upload a local file or directory to S3.
    
    Args:
        local_path: Local file or directory path
        s3_path: S3 destination path (s3://bucket/path)
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if successful, False otherwise
    """
    local_path = Path(local_path)
    
    if not local_path.exists():
        logger.error(f"Local path does not exist: {local_path}")
        return False
    
    try:
        s3_client = get_s3_client()
        bucket, key_prefix = parse_s3_path(s3_path)
        
        if local_path.is_file():
            # Upload single file
            if not overwrite:
                # Check if file exists
                try:
                    s3_client.head_object(Bucket=bucket, Key=key_prefix)
                    logger.info(f"File exists in S3, skipping: {s3_path}")
                    return True
                except ClientError:
                    pass  # File doesn't exist, proceed with upload
            
            logger.info(f"Uploading {local_path} to {s3_path}")
            s3_client.upload_file(str(local_path), bucket, key_prefix)
            logger.info(f"✅ Successfully uploaded to S3: {s3_path}")
            return True
            
        elif local_path.is_dir():
            # Upload directory recursively
            uploaded_count = 0
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Compute relative path for S3 key
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{key_prefix}/{relative_path}".replace('\\', '/')
                    
                    if not overwrite:
                        try:
                            s3_client.head_object(Bucket=bucket, Key=s3_key)
                            continue  # Skip existing file
                        except ClientError:
                            pass
                    
                    s3_client.upload_file(str(file_path), bucket, s3_key)
                    uploaded_count += 1
            
            logger.info(f"✅ Successfully uploaded {uploaded_count} files to S3: {s3_path}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def sync_from_s3(
    s3_path: str,
    local_path: Union[str, Path],
    overwrite: bool = True
) -> bool:
    """
    Download a file or directory from S3 to local.
    
    Args:
        s3_path: S3 source path (s3://bucket/path)
        local_path: Local destination path
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if successful, False otherwise
    """
    local_path = Path(local_path)
    
    try:
        s3_client = get_s3_client()
        bucket, key_prefix = parse_s3_path(s3_path)
        
        # Check if it's a single file or directory
        try:
            # Try to get object metadata (single file)
            s3_client.head_object(Bucket=bucket, Key=key_prefix)
            is_single_file = True
        except ClientError:
            # Not a single file, assume directory
            is_single_file = False
        
        if is_single_file:
            # Download single file
            if not overwrite and local_path.exists():
                logger.info(f"File exists locally, skipping: {local_path}")
                return True
            
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {s3_path} to {local_path}")
            s3_client.download_file(bucket, key_prefix, str(local_path))
            logger.info(f"✅ Successfully downloaded from S3: {s3_path}")
            return True
            
        else:
            # Download directory
            local_path.mkdir(parents=True, exist_ok=True)
            
            # List all objects with prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            downloaded_count = 0
            
            for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Compute local path
                    relative_path = Path(s3_key).relative_to(key_prefix)
                    file_local_path = local_path / relative_path
                    
                    if not overwrite and file_local_path.exists():
                        continue
                    
                    file_local_path.parent.mkdir(parents=True, exist_ok=True)
                    s3_client.download_file(bucket, s3_key, str(file_local_path))
                    downloaded_count += 1
            
            logger.info(f"✅ Successfully downloaded {downloaded_count} files from S3: {s3_path}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return False


def list_s3_files(s3_path: str, max_keys: int = 1000) -> list[str]:
    """
    List files in an S3 path.
    
    Args:
        s3_path: S3 path (s3://bucket/path)
        max_keys: Maximum number of keys to return
        
    Returns:
        List of S3 paths
    """
    try:
        s3_client = get_s3_client()
        bucket, key_prefix = parse_s3_path(s3_path)
        
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=key_prefix,
            MaxKeys=max_keys
        )
        
        if 'Contents' not in response:
            return []
        
        return [f"s3://{bucket}/{obj['Key']}" for obj in response['Contents']]
        
    except Exception as e:
        logger.error(f"Failed to list S3 files: {e}")
        return []


def get_sagemaker_paths() -> dict:
    """
    Get standard SageMaker paths.
    
    Returns:
        Dictionary with standard paths
    """
    base_path = Path('/home/ec2-user/SageMaker')
    
    return {
        'base': base_path,
        'data': base_path / 'data',
        'models': base_path / 'models',
        'results': base_path / 'results',
        'notebooks': base_path / 'notebooks',
    }


def ensure_local_directories():
    """
    Ensure standard local directories exist.
    """
    paths = get_sagemaker_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Local directories initialized")


class S3PathManager:
    """
    Manages paths with automatic S3 synchronization.
    """
    
    def __init__(self, s3_bucket: str, local_base: Optional[Path] = None):
        """
        Initialize S3 path manager.
        
        Args:
            s3_bucket: S3 bucket name (without s3:// prefix)
            local_base: Local base directory (defaults to SageMaker base)
        """
        self.s3_bucket = s3_bucket
        self.local_base = local_base or Path('/home/ec2-user/SageMaker')
        
    def get_local_path(self, relative_path: str) -> Path:
        """Get local path for a relative path."""
        return self.local_base / relative_path
    
    def get_s3_path(self, relative_path: str) -> str:
        """Get S3 path for a relative path."""
        return f"s3://{self.s3_bucket}/{relative_path}"
    
    def sync_to_s3(self, relative_path: str) -> bool:
        """Sync a local path to S3."""
        local = self.get_local_path(relative_path)
        s3 = self.get_s3_path(relative_path)
        return sync_to_s3(local, s3)
    
    def sync_from_s3(self, relative_path: str) -> bool:
        """Sync a path from S3 to local."""
        s3 = self.get_s3_path(relative_path)
        local = self.get_local_path(relative_path)
        return sync_from_s3(s3, local)


if __name__ == "__main__":
    # Basic tests
    print(f"Is SageMaker environment: {is_sagemaker_environment()}")
    print(f"Detected device: {detect_sagemaker_device()}")
    
    if is_sagemaker_environment():
        print("SageMaker paths:")
        for name, path in get_sagemaker_paths().items():
            print(f"  {name}: {path}")

