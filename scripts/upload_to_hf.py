"""Upload all staged PolyVL packages to Hugging Face under personal account Pushkarrrr15151."""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

WORKSPACE = Path("c:/Users/Admin/Desktop/final").resolve()
RELEASE_DIR = WORKSPACE / "release"

REPOS = [
    {
        "name": "PolyVL-router-laya",
        "repo_id": "Pushkarrrr15151/PolyVL-router-laya",
        "type": "model",
        "folder": RELEASE_DIR / "router-laya",
        "private": False
    },
    {
        "name": "PolyVL-router-requests",
        "repo_id": "Pushkarrrr15151/PolyVL-router-requests",
        "type": "dataset",
        "folder": RELEASE_DIR / "router-requests",
        "private": False
    },
    {
        "name": "PolyVL-tep-report-lora",
        "repo_id": "Pushkarrrr15151/PolyVL-tep-report-lora",
        "type": "model",
        "folder": RELEASE_DIR / "tep-report-lora",
        "private": False
    },
    {
        "name": "PolyVL-tep-report-distill",
        "repo_id": "Pushkarrrr15151/PolyVL-tep-report-distill",
        "type": "dataset",
        "folder": RELEASE_DIR / "tep-report-distill",
        "private": False
    },
    {
        "name": "PolyVL-logic-lora",
        "repo_id": "Pushkarrrr15151/PolyVL-logic-lora",
        "type": "model",
        "folder": RELEASE_DIR / "logic-lora",
        "private": False
    },
    {
        "name": "PolyVL-finance-rotation-lora",
        "repo_id": "Pushkarrrr15151/PolyVL-finance-rotation-lora",
        "type": "model",
        "folder": RELEASE_DIR / "finance-rotation-lora",
        "private": False
    },
    {
        "name": "PolyVL-deepfake-lora",
        "repo_id": "Pushkarrrr15151/PolyVL-deepfake-lora",
        "type": "model",
        "folder": RELEASE_DIR / "deepfake-lora",
        "private": False
    },
    {
        "name": "PolyVL-tep-protocol-a-lora",
        "repo_id": "Pushkarrrr15151/PolyVL-tep-protocol-a-lora",
        "type": "model",
        "folder": RELEASE_DIR / "tep-protocol-a-lora",
        "private": False
    }
]

def main():
    api = HfApi()
    try:
        user = api.whoami()
        username = user.get("name")
        print(f"Logged in user: {username}")
        token_info = user.get("auth", {}).get("accessToken", {})
        print(f"Token: {token_info.get('displayName')} (role: {token_info.get('role')})")
        scopes = token_info.get("fineGrained", {}).get("scoped", [])
        if scopes:
            print(f"Fine-grained scopes: {scopes}")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate: {e}")
        sys.exit(1)

    print("\nStarting repository creation and uploads:")
    print("=" * 60)

    for item in REPOS:
        repo_id = item["repo_id"]
        repo_type = item["type"]
        folder = item["folder"]
        print(f"\n--> Processing [{repo_type}] {repo_id} from {folder.name}...")
        
        if not folder.exists():
            print(f"    [SKIP] Folder {folder} does not exist!")
            continue

        try:
            print(f"    Creating {repo_type} repository: {repo_id}...")
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                exist_ok=True,
                private=item["private"]
            )
            print(f"    Repository verified/created.")
            
            print(f"    Uploading folder contents...")
            url = api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"Initial release of {repo_id}"
            )
            print(f"    [SUCCESS] Uploaded to {url}")
        except Exception as e:
            print(f"    [FAILED] {type(e).__name__}: {e}")
            print("    Halting execution due to error.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All PolyVL repositories successfully created and uploaded!")

if __name__ == "__main__":
    main()
