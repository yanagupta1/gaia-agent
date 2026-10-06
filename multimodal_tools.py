import os
from dataclasses import dataclass
from pathlib import Path

from file_tools import AttachmentContext, AttachmentTool
from web_tools import WebSearchTool


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


@dataclass(frozen=True)
class MediaInspection:
    kind: str
    path: Path
    details: str


class MultimodalTool:
    def __init__(
        self,
        attachment_tool: AttachmentTool | None = None,
        web_search: WebSearchTool | None = None,
    ):
        self.attachment_tool = attachment_tool or AttachmentTool()
        self.web_search = web_search or WebSearchTool()
        self.vision_backend = os.getenv("VISION_BACKEND", "").strip().lower()
        self.audio_backend = os.getenv("AUDIO_BACKEND", "").strip().lower()
        self.video_backend = os.getenv("VIDEO_BACKEND", "").strip().lower()

    def inspect_image(self, context: AttachmentContext) -> str:
        path = self.attachment_tool.download(context).resolve()
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            return f"Tool error: {context.file_name} is not a supported image file."

        details = [self._file_metadata(path)]
        try:
            from PIL import Image

            with Image.open(path) as image:
                details.append(
                    "Image metadata:\n"
                    f"Format: {image.format}\n"
                    f"Size: {image.width}x{image.height}\n"
                    f"Mode: {image.mode}"
                )
        except Exception as error:
            details.append(f"Could not inspect image dimensions: {error}")

        if not self.vision_backend:
            details.append(
                "Vision backend is not configured. This tool can provide image "
                "metadata only; it cannot identify objects, read diagrams, or solve "
                "chess positions yet."
            )
        return "\n\n".join(details)

    def transcribe_audio(self, context: AttachmentContext) -> str:
        path = self.attachment_tool.download(context).resolve()
        suffix = path.suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            return f"Tool error: {context.file_name} is not a supported audio file."

        details = [self._file_metadata(path)]
        if not self.audio_backend:
            details.append(
                "Audio backend is not configured. Set AUDIO_BACKEND to a real "
                "transcription backend before expecting audio answers."
            )
        return "\n\n".join(details)

    def inspect_video(self, context: AttachmentContext) -> str:
        path = self.attachment_tool.download(context).resolve()
        suffix = path.suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            return f"Tool error: {context.file_name} is not a supported video file."

        details = [self._file_metadata(path)]
        if not self.video_backend:
            details.append(
                "Video backend is not configured. This tool can provide file metadata "
                "only; it cannot inspect frames or motion yet."
            )
        return "\n\n".join(details)

    def youtube_research(self, url: str) -> str:
        if not url.strip():
            return "Tool error: missing YouTube URL."
        query = f"{url} transcript captions"
        results = self.web_search.search(query)
        if not results:
            return "No transcript/caption search results found for the video."
        lines = []
        for index, result in enumerate(results, start=1):
            snippet = f"\nSnippet: {result.snippet}" if result.snippet else ""
            lines.append(f"[{index}] {result.title}\nURL: {result.url}{snippet}")
        lines.append(
            "Use visit_webpage on relevant transcript/caption results, or continue "
            "searching with the video title if needed."
        )
        return "\n\n".join(lines)

    def _file_metadata(self, path: Path) -> str:
        return (
            f"File metadata:\n"
            f"Path: {path}\n"
            f"Extension: {path.suffix.lower() or '(none)'}\n"
            f"Size bytes: {path.stat().st_size}"
        )
