"""Video frame extraction for FieldWorkArena tasks."""

import base64
import io
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_frames_from_video(video_bytes: bytes, num_frames: int = 4) -> list[str]:
    """Extract key frames from video bytes and return as base64 strings.

    Args:
        video_bytes: Raw video file bytes
        num_frames: Number of frames to extract (default 4)

    Returns:
        List of base64-encoded JPEG frame images
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not available for frame extraction")
        return []

    frames = []

    try:
        # Write video to temp file for processing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            try:
                import cv2

                cap = cv2.VideoCapture(tmp.name)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if total_frames <= 0:
                    logger.warning("Could not determine video frame count")
                    return []

                # Sample evenly across the video
                frame_indices = [
                    int(i * total_frames / num_frames) for i in range(num_frames)
                ]

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)

                        # Resize for cost efficiency
                        img.thumbnail((800, 800))

                        # Convert to base64
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=75)
                        frames.append(base64.b64encode(buffer.getvalue()).decode())

                cap.release()

            except ImportError:
                logger.warning("OpenCV not available, using fallback frame extraction")
                # Fallback: just return empty - caller can use video thumbnail
                return []

    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")

    return frames


def extract_single_frame(video_bytes: bytes, timestamp_sec: float = 0) -> str | None:
    """Extract a single frame at a specific timestamp.

    Args:
        video_bytes: Raw video file bytes
        timestamp_sec: Timestamp in seconds to extract frame from

    Returns:
        Base64-encoded JPEG image or None if extraction fails
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            cap = cv2.VideoCapture(tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps > 0:
                frame_num = int(timestamp_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img.thumbnail((800, 800))

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=75)
                return base64.b64encode(buffer.getvalue()).decode()

    except Exception as e:
        logger.error(f"Error extracting frame: {e}")

    return None
