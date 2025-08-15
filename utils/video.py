# utils/video.py
import os
import shutil
import subprocess
import tempfile

def _find_ffmpeg_exe():
    """
    Prefer the system 'ffmpeg' if available; otherwise fall back to the
    Python package 'imageio-ffmpeg' (bundled binary).
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

def mp4_to_gif(mp4_path: str, gif_path: str, fps: int = 15, width: int = 640) -> bool:
    """
    Convert MP4 to GIF using ffmpeg with palette generation for good quality.
    Returns True on success, False if ffmpeg is not found or conversion fails.
    """
    ffmpeg = _find_ffmpeg_exe()
    if ffmpeg is None:
        return False

    # Use a portable temp location for the palette (works on Windows too)
    tmp_dir = tempfile.gettempdir()
    palette_path = os.path.join(tmp_dir, "ffmpeg_palette.png")

    # 1) Generate an optimized palette from the MP4
    palette_cmd = [
        ffmpeg, "-y",
        "-i", mp4_path,
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff",
        "-hide_banner",
        palette_path
    ]

    # 2) Apply the palette to create the GIF
    gif_cmd = [
        ffmpeg, "-y",
        "-i", mp4_path, "-i", palette_path,
        "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos [x]; [x][1:v] paletteuse=new=1",
        "-loop", "0",
        "-hide_banner",
        gif_path
    ]

    try:
        subprocess.run(palette_cmd, check=True)
        subprocess.run(gif_cmd, check=True)
        # Clean up palette if possible
        try:
            os.remove(palette_path) # Remove the temporary palette file, after conversion is no longer needed
        except OSError:
            pass
        return True
    except Exception:
        return False
