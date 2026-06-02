locals {
  name_prefix = "${var.project_name}-${var.stage}"

  deployment_bucket_name = (
    var.deployment_bucket_name != null && var.deployment_bucket_name != ""
  ) ? var.deployment_bucket_name : "${local.name_prefix}-deployment-bucket"
}

resource "aws_s3_bucket" "deployment_bucket" {
  bucket        = local.deployment_bucket_name
  force_destroy = true

  tags = {
    Name = local.deployment_bucket_name
  }
}
