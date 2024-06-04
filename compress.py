import os
import subprocess

def compress_videos(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4"):

            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, f"compressed_{filename}")

            ffmpeg_cmd=f'ffmpeg -i {input_path} -c:v libx264 -c:a copy -crf 20 {output_path}'
            subprocess.run(ffmpeg_cmd, shell=True)

folder_path = 'output_videos'
compress_videos(folder_path)