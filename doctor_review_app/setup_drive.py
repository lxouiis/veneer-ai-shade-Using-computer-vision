"""Run this ONCE, locally, to connect this app to your Google Drive.

Prerequisite (one-time, in Google Cloud Console - console.cloud.google.com):
  1. Create a project (or reuse one), enable the "Google Drive API".
  2. Create an OAuth 2.0 Client ID, type "Desktop app". Download its JSON -
     save it next to this script as client_secret.json.
  3. Run: python setup_drive.py
     A browser window opens - sign in with the Google account whose Drive
     will hold the dataset/review data, approve access.
  4. This prints GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET / GDRIVE_REFRESH_TOKEN
     and creates (or reuses) a "tooth_shade_review" root folder, printing its
     ID as GDRIVE_ROOT_ID.

Paste all four into Streamlit Community Cloud's app Secrets (for the hosted
version), or a local .env, as plain KEY=VALUE pairs. They are not printed
to any file here - copy them from the terminal output.

Scope is drive.file: the app can only see files/folders IT creates, never
your whole Drive - same scope the existing CVD_Annotator tool uses.
"""
import json
import os
from google_auth_oauthlib.flow import InstalledAppFlow

from drive_client import DriveClient, SCOPES

CLIENT_SECRET_PATH = "client_secret.json"
ROOT_FOLDER_NAME = "tooth_shade_review"


def main():
    if not os.path.exists(CLIENT_SECRET_PATH):
        raise SystemExit(
            f"{CLIENT_SECRET_PATH} not found - see this file's docstring for "
            f"how to create one in Google Cloud Console first."
        )

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_PATH, SCOPES)
    creds = flow.run_local_server(port=0)

    with open(CLIENT_SECRET_PATH) as f:
        client_config = json.load(f)["installed"]

    os.environ["GDRIVE_CLIENT_ID"] = client_config["client_id"]
    os.environ["GDRIVE_CLIENT_SECRET"] = client_config["client_secret"]
    os.environ["GDRIVE_REFRESH_TOKEN"] = creds.refresh_token

    client = DriveClient.from_env()
    root_id = client.get_or_create_folder(ROOT_FOLDER_NAME, parent_id=None)
    client.get_or_create_folder("images", parent_id=root_id)
    client.get_or_create_folder("approved_masks", parent_id=root_id)
    client.get_or_create_folder("needs_review", parent_id=root_id)

    print("\nSetup complete. Paste these into Streamlit Secrets (or a local .env):\n")
    print(f'GDRIVE_CLIENT_ID = "{client_config["client_id"]}"')
    print(f'GDRIVE_CLIENT_SECRET = "{client_config["client_secret"]}"')
    print(f'GDRIVE_REFRESH_TOKEN = "{creds.refresh_token}"')
    print(f'GDRIVE_ROOT_ID = "{root_id}"')
    print(f"\nUpload the dataset images to the '{ROOT_FOLDER_NAME}/images' "
          f"folder in Drive before running app.py.")


if __name__ == "__main__":
    main()
