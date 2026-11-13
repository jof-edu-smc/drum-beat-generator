"""Submit the containerized GAN training job to Vertex AI."""

from __future__ import annotations

import argparse

from google.cloud import aiplatform # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="drumbeat-generator")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--staging-bucket", default="gs://drumbeat-studio-training-data")
    parser.add_argument(
        "--image-uri",
        default="us-central1-docker.pkg.dev/drumbeat-generator/drumbeat/training:latest",
    )
    parser.add_argument(
        "--metadata-uri",
        default="gs://drumbeat-studio-training-data/metadata.json",
    )
    parser.add_argument(
        "--dataset-uri",
        default="gs://drumbeat-studio-training-data/Apple",
    )
    parser.add_argument("--display-name", default="drumbeat-gan-training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--genre", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aiplatform.init(project=args.project, location=args.region, staging_bucket=args.staging_bucket)
    job = aiplatform.CustomJob(
        display_name=args.display_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4",
                    "accelerator_type": "NVIDIA_TESLA_T4",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": args.image_uri,
                    "command": ["python", "vertex_train.py"],
                    "args": [
                        "--metadata-uri",
                        args.metadata_uri,
                        "--dataset-uri",
                        args.dataset_uri,
                        "--genre",
                        args.genre,
                        "--epochs",
                        str(args.epochs),
                    ],
                },
            }
        ],
    )
    job.run(
        base_output_dir=f"{args.staging_bucket.rstrip('/')}/models/{args.display_name}",
        sync=True,
    )


if __name__ == "__main__":
    main()
