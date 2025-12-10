import os
import subprocess

# ✅ Configuration
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ROOT_FOLDERS = [r"C:/Users/Lenovo/DeepFake_New/DeepFake_New/dataset2_raw/Celeb-DF & FF++"]



def convert_to_h264(input_path):
    """
    Converts uploaded video to MP4 (H.264/AAC) format for browser playback.
    """
    import subprocess, os
    ffmpeg_path = r"C:/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe"  # <-- ✅ Update this to your actual path

    base, ext = os.path.splitext(input_path)
    output_path = base + "_converted.mp4"

    try:
        cmd = [
            ffmpeg_path, "-y", "-i", input_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-strict", "experimental", "-movflags", "+faststart",
            output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if os.path.exists(output_path):
            print("✅ Video converted for web playback:", output_path)
            return output_path
        else:
            return input_path
    except Exception as e:
        print("⚠️ FFmpeg conversion failed:", e)
        return input_path



def convert_all_videos(root_folders):
    """
    Recursively converts all videos inside given dataset root folders.
    Works for nested 'real' and 'fake' directories.
    """
    total_videos = 0
    converted_videos = 0

    for root_dir in root_folders:
        print(f"\n📂 Scanning dataset: {root_dir}")

        for subdir, _, files in os.walk(root_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in VIDEO_EXTENSIONS:
                    total_videos += 1
                    video_path = os.path.join(subdir, file)
                    print(f"🎞 Processing: {video_path}")
                    out = convert_to_h264(video_path)
                    if out.endswith("_converted.mp4"):
                        converted_videos += 1

    print("\n✅ Conversion Completed!")
    print(f"Total videos found: {total_videos}")
    print(f"Converted successfully: {converted_videos}")
    print(f"Skipped/failed: {total_videos - converted_videos}")


if __name__ == "__main__":
    convert_all_videos(ROOT_FOLDERS)
