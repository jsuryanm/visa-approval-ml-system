import boto3
from us_visa.constants.constant import REGION_NAME


class S3Client:
    s3_client = None
    s3_resource = None

    def __init__(self, region_name=REGION_NAME):
        """
        Create S3 client/resource using boto3's default credential chain.
        Works with:
        - IAM Role (EC2 / ECS / EKS)
        - Environment variables (local dev)
        - ~/.aws/credentials
        """

        if S3Client.s3_client is None or S3Client.s3_resource is None:
            # DO NOT pass access keys explicitly
            S3Client.s3_resource = boto3.resource(
                "s3",
                region_name=region_name
            )
            S3Client.s3_client = boto3.client(
                "s3",
                region_name=region_name
            )

        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client
