"""Run this ONCE, locally, after setup_drive.py, to populate the Drive
'images' folder with the dataset app.py will pull review images from.

Needs GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET / GDRIVE_REFRESH_TOKEN /
GDRIVE_ROOT_ID set as environment variables (printed by setup_drive.py).

Usage:
    python upload_dataset_to_drive.py /path/to/local/images/folder
"""
import os
import sys

from drive_client import DriveClient

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python upload_dataset_to_drive.py /path/to/local/images/folder")
    local_dir = sys.argv[1]

    client = DriveClient.from_env()
    root_id = os.environ["GDRIVE_ROOT_ID"]
    images_folder_id = client.get_or_create_folder("images", root_id)

    # Skip files already uploaded - lets this script be re-run safely if it
    # gets interrupted partway through 1000+ images, rather than duplicating
    # everything already there.
    already_uploaded = {f["name"] for f in client.list_children(images_folder_id)}

    local_files = sorted(f for f in os.listdir(local_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
    to_upload = [f for f in local_files if f not in already_uploaded]
    print(f"{len(local_files)} local images, {len(already_uploaded)} already in Drive, "
          f"uploading {len(to_upload)} new ones.")

    mime_by_ext = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

    for i, filename in enumerate(to_upload, 1):
        path = os.path.join(local_dir, filename)
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(filename)[1].lower()
        client.upload_bytes(filename, images_folder_id, data, mime_by_ext[ext])
        if i % 25 == 0 or i == len(to_upload):
            print(f"  uploaded {i} / {len(to_upload)}")

    print("Done.")


if __name__ == "__main__":
    main()
