import os

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError

from forecasting import (
    AWS_ACCESS_KEY_ID,
    AWS_REGION,
    AWS_SECRET_ACCESS_KEY,
    RAW_DATA_BUCKET,
)
from forecasting.configs import logger


class AWSStorage:
    """Class for interacting with AWS S3 storage."""

    def __init__(self, client: boto3.client, bucket: str):
        """
        Initialize the AWSStorage instance.

        Parameters:
        - client: The AWS S3 client.
        - bucket: The name of the S3 bucket.
        """
        self._client = client
        self._bucket = bucket

    def upload(self, local_path: str, remote_path: str):
        """
        Upload a file to the specified location in the S3 bucket.

        Parameters:
        - local_path: The local path of the file to be uploaded.
        - remote_path: The remote path in the S3 bucket.

        Returns:
            None

        Raises:
            S3UploadFailedError: If the upload fails.
        """
        try:
            logger.info(f"Uploading {local_path} to s3://{self._bucket}/{remote_path}")
            self._client.upload_file(local_path, self._bucket, remote_path)
        except S3UploadFailedError as e:
            logger.error("Upload aws failed.")
            raise e
        logger.info("Upload file complete")

    def upload_folder(self, local_folder_path: str, s3_folder_path: str):
        """
        Upload all files from a specified local folder to a folder in the S3 bucket.

        Parameters:
        - local_folder_path: The local path of the folder to be uploaded.
        - s3_folder_path: The remote path in the S3 bucket where files should be uploaded.

        Returns:
            None

        Raises:
            S3UploadFailedError: If the upload fails.
        """
        logger.info(
            f"Uploading all files from {local_folder_path} to s3://{self._bucket}/{s3_folder_path}"
        )
        for root, _, files in os.walk(local_folder_path):
            for filename in files:
                # construct the full local path
                local_path = os.path.join(root, filename)

                # construct the full s3 path
                relative_path = os.path.relpath(local_path, local_folder_path)
                s3_path = os.path.join(s3_folder_path, relative_path)
                # Upload the file
                self.upload(local_path, s3_path)
        logger.info("Upload folder complete")

    def download_file(self, key: str, local_file_path: str, version_id: str = None):
        """
        Download a specific version of a file from the S3 bucket.

        Parameters:
        - key: The key of the file in the S3 bucket.
        - local_file_path: The local path to save the downloaded file.
        - version_id: The version ID of the file to download.

        Returns:
            None

        Raises:
            ClientError: If the download fails.
        """
        try:
            extra_args = {"VersionId": version_id} if version_id else {}
            self._client.download_file(
                self._bucket, key, local_file_path, ExtraArgs=extra_args
            )
        except ClientError as e:
            logger.error(
                f"Failed to download file {key} with version {version_id}: {str(e)}"
            )
            raise e

    def download_folder(self, s3_folder_path: str, local_folder_path: str) -> bool:
        """
        Download all files from a specified folder in the S3 bucket to a local folder.

        Parameters:
        - s3_folder_path: The folder path in the S3 bucket.
        - local_folder_path: The local path to save the downloaded files.

        Returns:
            None

        Raises:
            ClientError: If the download fails.
        """
        logger.info(
            f"Downloading all files from s3://{self._bucket}/{s3_folder_path} to {local_folder_path}"
        )
        # Ensure the local directory exists
        os.makedirs(local_folder_path, exist_ok=True)
        try:
            # List all objects within the specified folder
            response = self._client.list_objects_v2(
                Bucket=self._bucket, Prefix=s3_folder_path
            )
            if "Contents" in response:
                for obj in response["Contents"]:
                    # Ensure the object key starts with the specified folder path
                    if obj["Key"].startswith(s3_folder_path):
                        # Construct the full local path for the file
                        local_file_path = os.path.join(
                            local_folder_path,
                            os.path.relpath(obj["Key"], s3_folder_path),
                        )

                        # Ensure the local directory structure exists
                        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                        # Download the file
                        self._client.download_file(
                            self._bucket, obj["Key"], local_file_path
                        )
            else:
                logger.info("No files found to download.")
        except ClientError as e:
            logger.error(f"An error occurred: {e}")

    def getFullURL(self, remote_path: str) -> str:
        """
        Get the full URL of a file in the S3 bucket.

        Parameters:
        - remote_path: The remote path in the S3 bucket.

        Returns:
        - The full URL of the file in the S3 bucket.
        """
        base_url = "https://s3.console.aws.amazon.com/s3/buckets/"
        params = f"?region={AWS_REGION}&prefix={remote_path}/&showversions=false"
        return f"{base_url}{self._bucket}{params}"

    def list_objects(self, s3_path):
        """
        List objects in the S3 bucket with a given prefix.

        Parameters:
        - s3_path: The prefix to filter objects.

        Returns:
        - List of list_objects in the S3 bucket with the specified prefix.
        """
        return self._client.list_objects_v2(
            Bucket=self._bucket, Prefix=s3_path, Delimiter="/"
        )

    def list_object_versions(self, s3_path: str):
        """
        List all versions of objects in the S3 bucket with a given prefix.

        Parameters:
        - s3_path: The prefix to filter objects.

        Returns:
        - List of versions of objects in the S3 bucket with the specified prefix.
        """
        response = self._client.list_object_versions(
            Bucket=self._bucket, Prefix=s3_path
        )
        versions = response.get("Versions", [])
        if not versions:
            print(response)  # Print the response to debug
            logger.info(f"No versions found for {s3_path}")
        return versions

    def file_exists(self, remote_path: str) -> bool:
        """
        Check if a file exists in the S3 bucket.

        Parameters:
        - remote_path: The remote path in the S3 bucket.

        Returns:
        - True if the file exists, False otherwise.
        """
        try:
            self._client.head_object(Bucket=self._bucket, Key=remote_path)
            logger.info(f"File exists: s3://{self._bucket}/{remote_path}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info(f"File does not exist: s3://{self._bucket}/{remote_path}")
                return False
            else:
                logger.error(f"An error occurred: {e}")
                raise e


def create_client(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
) -> boto3.client:
    """
    Create an S3 client.

    Parameters:
    - aws_access_key_id: The AWS access key ID.
    - aws_secret_access_key: The AWS secret access key.
    - region_name: The AWS region name.

    Returns:
    - The S3 client.
    """
    client = None
    if not client:
        client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
    return client


aws_client = create_client()
s3_data_service = AWSStorage(client=aws_client, bucket=RAW_DATA_BUCKET)
