"""
Kaggle Automation Script for Phase A
Handles authentication, dataset creation, kernel submission, status polling, and artifact downloading.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
EMBEDDINGS_DIR = ROOT_DIR / "data" / "embeddings"
MODELS_DIR = ROOT_DIR / "models"
KAGGLE_OUTPUT_DIR = EXPERIMENTS_DIR / "kaggle_output"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def configure_kaggle_credentials():
    """
    Loads credentials from .env and configures Kaggle API authentication.
    Handles JSON token, username:key pair, or raw API token.
    """
    load_dotenv(ROOT_DIR / ".env")

    # Check various possible environment variables
    token = os.getenv("kaggle_token") or os.getenv("KAGGLE_TOKEN")
    username = os.getenv("KAGGLE_USERNAME") or os.getenv("kaggle_username")
    key = os.getenv("KAGGLE_KEY") or os.getenv("kaggle_key")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    if token:
        token = token.strip()
        # Case 1: token is raw JSON string
        if token.startswith("{") and token.endswith("}"):
            try:
                data = json.loads(token)
                username = data.get("username")
                key = data.get("key")
            except Exception:
                pass
        # Case 2: username:key format
        elif ":" in token and not username:
            parts = token.split(":", 1)
            username, key = parts[0].strip(), parts[1].strip()
        # Case 3: API token string (Kaggle >= 1.7)
        elif not username:
            os.environ["KAGGLE_API_TOKEN"] = token

    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        # Also write ~/.kaggle/kaggle.json for CLI compatibility
        with open(kaggle_json_path, "w", encoding="utf-8") as f:
            json.dump({"username": username, "key": key}, f)
        try:
            os.chmod(kaggle_json_path, 0o600)
        except Exception:
            pass

    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username") or username or "user"
    print(f"Kaggle API authenticated successfully for user: {user}")
    return api, user


def upload_dataset_splits(api, username: str):
    """
    Creates or updates the private Kaggle dataset 'productmatching-splits'.
    """
    print("\n[Step 1/4] Preparing dataset splits for Kaggle...")
    staging_dir = DATA_PROCESSED_DIR / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Copy split files into staging
    for csv_file in ["train_split.csv", "val_split.csv", "test_demo_split.csv"]:
        src_path = DATA_PROCESSED_DIR / csv_file
        if src_path.exists():
            shutil.copy2(src_path, staging_dir / csv_file)
        else:
            print(f"Warning: {src_path} not found.")

    dataset_slug = "productmatching-splits"
    metadata = {
        "title": "Shopee Product Matching Zero Leakage Splits",
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    try:
        # Check if dataset already exists
        api.dataset_status(f"{username}/{dataset_slug}")
        print(f"Dataset {username}/{dataset_slug} exists. Uploading new version...")
        api.dataset_create_version(
            str(staging_dir),
            version_notes="Automated splits update for product matching pipeline",
            dir_mode="zip"
        )
    except Exception as e:
        print(f"Creating new dataset {username}/{dataset_slug}...")
        try:
            api.dataset_create_new(str(staging_dir), dir_mode="zip")
        except Exception as create_err:
            print(f"Dataset create note: {create_err}")

    print(f"Dataset {username}/{dataset_slug} is ready.")
    return f"{username}/{dataset_slug}"


def push_kernel(api, username: str, dataset_id: str):
    """
    Configures kernel-metadata.json and pushes the pipeline to Kaggle GPU.
    """
    print("\n[Step 2/4] Pushing GPU pipeline kernel to Kaggle...")
    kernel_slug = "shopee-multimodal-pipeline"

    kernel_metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": "Shopee Multimodal Pipeline",
        "code_file": "kaggle_pipeline.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [dataset_id],
        "competition_sources": ["shopee-product-matching"],
        "kernel_sources": [],
    }

    metadata_path = NOTEBOOKS_DIR / "kernel-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_metadata, f, indent=2)

    api.kernels_push(str(NOTEBOOKS_DIR))
    print(f"Kernel {username}/{kernel_slug} pushed successfully!")
    return f"{username}/{kernel_slug}"


def get_status_str(status_info) -> str:
    """Safely extracts lowercase status string across different kaggle SDK versions."""
    if hasattr(status_info, "status"):
        raw = str(status_info.status)
        return raw.split(".")[-1].lower()
    if isinstance(status_info, dict):
        return str(status_info.get("status", "unknown")).lower()
    return str(status_info).lower()


def get_failure_message(status_info) -> str:
    """Safely extracts failure message string across different kaggle SDK versions."""
    if hasattr(status_info, "failure_message") and status_info.failure_message:
        return str(status_info.failure_message)
    if hasattr(status_info, "failureMessage") and status_info.failureMessage:
        return str(status_info.failureMessage)
    if isinstance(status_info, dict):
        return str(status_info.get("failureMessage") or status_info.get("failure_message") or "")
    return ""


def poll_kernel_status(api, kernel_id: str, poll_interval_sec: int = 60):
    """
    Polls the running Kaggle kernel until execution completes or errors.
    """
    print(f"\n[Step 3/4] Monitoring execution of {kernel_id} on Kaggle GPU...")
    while True:
        status_info = api.kernels_status(kernel_id)
        status = get_status_str(status_info)
        print(f"[{time.strftime('%H:%M:%S')}] Kernel Status: {status.upper()}")

        if status == "complete":
            print("GPU Pipeline finished successfully!")
            break
        elif status in ["error", "failed", "cancelled"]:
            failure_message = get_failure_message(status_info) or "Execution error on Kaggle (inspect kernel logs)"
            raise RuntimeError(f"Kaggle kernel run failed with status: {status}. Error: {failure_message}")

        time.sleep(poll_interval_sec)


def download_and_sync_artifacts(api, kernel_id: str):
    """
    Downloads outputs from Kaggle working directory and syncs to local paths.
    """
    print("\n[Step 4/4] Downloading generated artifacts from Kaggle...")
    api.kernels_output(kernel_id, path=str(KAGGLE_OUTPUT_DIR))
    print(f"Downloaded files to {KAGGLE_OUTPUT_DIR}")

    # 1. Sync JSON benchmark and configuration files to experiments/
    for json_file in ["results.json", "final_config.json", "sweeps.json", "latency.json", "error_cases.json"]:
        src = KAGGLE_OUTPUT_DIR / json_file
        if src.exists():
            shutil.copy2(src, EXPERIMENTS_DIR / json_file)
            print(f"  [OK] Synced {json_file} -> experiments/")

    # 2. Sync generated numpy embeddings to data/embeddings/
    for npy_file in KAGGLE_OUTPUT_DIR.glob("*.npy"):
        shutil.copy2(npy_file, EMBEDDINGS_DIR / npy_file.name)
        print(f"  [OK] Synced {npy_file.name} -> data/embeddings/")

    # 3. Sync fine-tuned model checkpoint if generated
    src_models = KAGGLE_OUTPUT_DIR / "models" / "text_ft"
    dst_models = MODELS_DIR / "text_ft"
    if src_models.exists():
        if dst_models.exists():
            shutil.rmtree(dst_models)
        shutil.copytree(src_models, dst_models)
        print("  [OK] Synced fine-tuned text encoder -> models/text_ft/")

    print("\nAll Phase A artifacts have been successfully integrated into the project!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Automate Kaggle GPU pipeline execution and artifact synchronization")
    parser.add_argument(
        "--async", "--push-only",
        dest="push_only",
        action="store_true",
        help="Push kernel to Kaggle and exit immediately without waiting (allows shutting down PC)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check the current execution status of the Kaggle GPU kernel",
    )
    parser.add_argument(
        "--download-only", "--sync-only",
        dest="download_only",
        action="store_true",
        help="Download artifacts from a completed Kaggle kernel without re-running",
    )
    args = parser.parse_args()

    try:
        api, username = configure_kaggle_credentials()
    except Exception as e:
        print(f"Kaggle authentication error: {e}")
        print("Please ensure kaggle_token is populated in .env")
        sys.exit(1)

    kernel_id = f"{username}/shopee-multimodal-pipeline"

    if args.status:
        try:
            status_info = api.kernels_status(kernel_id)
            status = get_status_str(status_info)
            print(f"Kernel '{kernel_id}' Status: {status.upper()}")
            if status == "error":
                msg = get_failure_message(status_info)
                if msg:
                    print(f"Failure Details: {msg}")
        except Exception as e:
            print(f"Error checking status for {kernel_id}: {e}")
        return

    if args.download_only:
        try:
            status_info = api.kernels_status(kernel_id)
            status = get_status_str(status_info)
            print(f"Kernel '{kernel_id}' Status: {status.upper()}")
            if status != "complete":
                print(f"Kernel is not complete yet (status: {status}). Please wait until it completes.")
                sys.exit(1)
            download_and_sync_artifacts(api, kernel_id)
        except Exception as e:
            print(f"Error downloading artifacts for {kernel_id}: {e}")
            sys.exit(1)
        return

    try:
        dataset_id = upload_dataset_splits(api, username)
        kernel_id = push_kernel(api, username, dataset_id)

        if args.push_only:
            print("\n" + "=" * 65)
            print("[OK] PUSHED ASYNCHRONOUSLY TO KAGGLE!")
            print(f"Kernel is running remotely on Kaggle cloud servers:")
            print(f"  https://www.kaggle.com/code/{kernel_id}")
            print("\nYou can safely shut down or disconnect your computer!")
            print("When you turn on your computer later, run:")
            print(f"  python scripts/kaggle_run.py --status")
            print(f"  python scripts/kaggle_run.py --download-only")
            print("=" * 65)
            return

        poll_kernel_status(api, kernel_id)
        download_and_sync_artifacts(api, kernel_id)
    except Exception as err:
        print(f"\nExecution encountered an error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
