import cv2
import os, ffmpeg

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()


# def compress_video(video_full_path, output_file_name, target_size):
#     # Reference: https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
#     min_audio_bitrate = 32000
#     max_audio_bitrate = 256000

#     probe = ffmpeg.probe(video_full_path)
#     # Video duration, in s.
#     duration = float(probe['format']['duration'])
#     # Audio bitrate, in bps.
#     audio_bitrate = float(next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)['bit_rate'])
#     # Target total bitrate, in bps.
#     target_total_bitrate = (target_size * 1024 * 8) / (1.073741824 * duration)

#     # Target audio bitrate, in bps
#     if 10 * audio_bitrate > target_total_bitrate:
#         audio_bitrate = target_total_bitrate / 10
#         if audio_bitrate < min_audio_bitrate < target_total_bitrate:
#             audio_bitrate = min_audio_bitrate
#         elif audio_bitrate > max_audio_bitrate:
#             audio_bitrate = max_audio_bitrate
#     # Target video bitrate, in bps.
#     video_bitrate = target_total_bitrate - audio_bitrate

#     i = ffmpeg.input(video_full_path)
#     ffmpeg.output(i, os.devnull,
#                   **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
#                   ).overwrite_output().run()
#     ffmpeg.output(i, output_file_name,
#                   **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 2, 'c:a': 'aac', 'b:a': audio_bitrate}
#                   ).overwrite_output().run()

# def compress_video(input_path, output_path, bitrate="1M"):
#     """
#     Compresses a video to a specified bitrate.
    
#     :param input_path: Path to the input video file.
#     :param output_path: Path to save the compressed video file.
#     :param bitrate: Target bitrate (e.g., '1M' for 1 Mbps).
#     """
#     ffmpeg.input(input_path).output(output_path, video_bitrate=bitrate).run()

