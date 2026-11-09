"""Run this locally whenever you want module_5 to use Doc Sans's latest
reviewed masks. Pulls everything from Drive's 'approved_masks' folder down
to a local directory, which you then publish as a Kaggle Dataset (same
pattern as the model checkpoints) and attach to the session running
module_5_integration/pipeline.py.

Needs GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET / GDRIVE_REFRESH_TOKEN /
GDRIVE_ROOT_ID set as environment variables (printed by setup_drive.py).

Usage:
    python download_approved_masks.py ./approved_masks_local
"""
import os
import sys

from drive_client import DriveClient

MIME_EXT = {
    "image/png": ".png",
}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python download_approved_masks.py /local/output/dir")
    local_dir = sys.argv[1]
    os.makedirs(local_dir, exist_ok=True)

    client = DriveClient.from_env()
    root_id = os.environ["GDRIVE_ROOT_ID"]
    approved_folder_id = client.get_or_create_folder("approved_masks", root_id)

    files = client.list_children(approved_folder_id)
    print(f"{len(files)} approved masks in Drive. Downloading to {local_dir} ...")

    # Skip masks already downloaded - safe to re-run as Doc Sans reviews
    # more images, without re-fetching everything from scratch each time.
    already_local = set(os.listdir(local_dir))
    new_count = 0

    for i, file_entry in enumerate(files, 1):
        if file_entry["name"] in already_local:
            continue
        data = client.download_bytes(file_entry["id"])
        with open(os.path.join(local_dir, file_entry["name"]), "wb") as f:
            f.write(data)
        new_count += 1
        if i % 50 == 0 or i == len(files):
            print(f"  checked {i} / {len(files)}")

    print(f"Done. {new_count} new masks downloaded, {len(files) - new_count} already up to date.")


if __name__ == "__main__":
    main()
