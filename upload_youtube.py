import os
import time
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

class UploadYoutube:
    def __init__(self, client_secrets_file, scopes):
        self.client_secrets_file_ = client_secrets_file
        self.scopes_ = scopes
        self.creds = None

    def authenticate(self):
        flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_file_, self.scopes_)
        self.creds = flow.run_local_server(port=0)

    def do(self, music_file, image_file):
        youtube = build('youtube', 'v3', credentials=self.creds)

        # Combine image and music into a video file (You need ffmpeg installed)
        video_file = "output_video.mp4"
        os.system(f"ffmpeg -loop 1 -i {image_file} -i {music_file} -c:v libx264 -c:a aac -strict experimental -b:a 192k -shortest {video_file}")

        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": "Generated Music and Image",
                    "description": "This is a video with generated music and image",
                    "tags": ["music", "image", "generated"],
                    "categoryId": "10"  # Music category
                },
                "status": {
                    "privacyStatus": "public"
                }
            },
            media_body=MediaFileUpload(video_file)
        )
        response = request.execute()
        print(f"Uploaded video with ID: {response['id']}")