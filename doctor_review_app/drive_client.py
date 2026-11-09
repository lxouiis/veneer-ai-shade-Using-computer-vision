"""Thin wrapper around the Google Drive v3 API - only the calls this app
needs. Same pattern (OAuth refresh token, drive.file scope - sees only
files it created itself) as the existing CVD_Annotator tool, reused here
rather than inventing a second auth approach for the same problem.
"""
from __future__ import annotations
import io
import os

FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class DriveClient:
    def __init__(self, service):
        self.svc = service

    @classmethod
    def from_env(cls) -> "DriveClient":
        """Needs GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN
        (see setup_drive.py for how these are minted, once, locally)."""
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        missing = [k for k in ("GDRIVE_CLIENT_ID", "GDRIVE_CLIENT_SECRET", "GDRIVE_REFRESH_TOKEN")
                   if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"Missing secrets: {', '.join(missing)}")
        creds = Credentials(
            None,
            refresh_token=os.environ["GDRIVE_REFRESH_TOKEN"],
            client_id=os.environ["GDRIVE_CLIENT_ID"],
            client_secret=os.environ["GDRIVE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=SCOPES,
        )
        return cls(build("drive", "v3", credentials=creds, cache_discovery=False))

    # ---------- queries ----------
    def list_children(self, parent_id: str) -> list[dict]:
        out, token = [], None
        while True:
            r = self.svc.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType)",
                pageSize=1000, pageToken=token,
            ).execute()
            out += r.get("files", [])
            token = r.get("nextPageToken")
            if not token:
                return out

    def find(self, name: str, parent_id: str) -> dict | None:
        safe = name.replace("\\", "\\\\").replace("'", "\\'")
        r = self.svc.files().list(
            q=f"name = '{safe}' and '{parent_id}' in parents and trashed = false",
            fields="files(id, name, mimeType)", pageSize=10,
        ).execute()
        files = r.get("files", [])
        return files[0] if files else None

    # ---------- writes ----------
    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        body = {"name": name, "mimeType": FOLDER_MIME}
        if parent_id:
            body["parents"] = [parent_id]
        return self.svc.files().create(body=body, fields="id").execute()["id"]

    def get_or_create_folder(self, name: str, parent_id: str | None) -> str:
        if parent_id:
            existing = self.find(name, parent_id)
            if existing:
                return existing["id"]
        return self.create_folder(name, parent_id)

    def upload_bytes(self, name: str, parent_id: str, data: bytes, mime: str,
                      file_id: str | None = None) -> str:
        """Create a file, or replace the content of an existing file_id -
        used both for images/masks and for overwriting reviews.json."""
        from googleapiclient.http import MediaIoBaseUpload
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=len(data) > 5_000_000)
        if file_id:
            return self.svc.files().update(fileId=file_id, media_body=media, fields="id").execute()["id"]
        return self.svc.files().create(
            body={"name": name, "parents": [parent_id]}, media_body=media, fields="id"
        ).execute()["id"]

    # ---------- reads ----------
    def download_bytes(self, file_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, self.svc.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()
