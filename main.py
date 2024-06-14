# pip install -r requirements.txt 
# 갱신시 pip freeze > requirements.txt

from gen_music import GenMusic
from gen_image import GenImage
#from upload_youtube import UploadYoutube

if __name__ == '__main__':
    prompt = "Music used to drive autonomously on the highway in a Tesla vehicle"

    gen_m = GenMusic()
    gen_i = GenImage()

    music_file = gen_m.do(prompt, 600)
    image_file = gen_i.do(prompt)

    # client_secrets_file = "path/to/client_secret.json"  # Replace with the path to your client_secret.json file
    # scopes = ["https://www.googleapis.com/auth/youtube.upload"]

    # uploader = UploadYoutube(client_secrets_file, scopes)
    # uploader.authenticate()
    # uploader.do(music_file, image_file)
