"""
Camera Plugin

Plugin for imaging and video capture in lab automation.
"""

import re
from enum import Enum
from typing import Any, Dict, List
from ...core.plugin_base import PluginBase, ParserBase, OperationDef


class CameraOperation(Enum):
    """Camera operation types."""
    START_STREAM = "start_stream"
    STOP_STREAM = "stop_stream"
    CAPTURE_IMAGE = "capture_image"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"


CAMERA_OPERATIONS: Dict[CameraOperation, OperationDef] = {
    CameraOperation.START_STREAM: OperationDef(
        name="start_stream",
        action="camera.start_stream",
        keywords={
            "en": ["start stream", "start video", "begin streaming", "start camera"],
            "zh": ["开始录像", "开始视频", "启动摄像头"],
        },
        params_schema={
            "filename_prefix": {"type": "string", "required": False},
        },
        description="Start video streaming",
    ),

    CameraOperation.STOP_STREAM: OperationDef(
        name="stop_stream",
        action="camera.stop_stream",
        keywords={
            "en": ["stop stream", "stop video", "end streaming", "stop camera"],
            "zh": ["停止录像", "停止视频", "关闭摄像头"],
        },
        params_schema={},
        description="Stop video streaming",
    ),

    CameraOperation.CAPTURE_IMAGE: OperationDef(
        name="capture_image",
        action="camera.capture_image",
        keywords={
            "en": ["capture image", "capture", "take photo", "snapshot", "take picture", "photograph"],
            "zh": ["拍照", "截图", "抓拍"],
        },
        params_schema={
            "filename": {"type": "string", "required": False},
        },
        description="Capture a still image",
    ),

    CameraOperation.START_RECORDING: OperationDef(
        name="start_recording",
        action="camera.start_recording",
        keywords={
            "en": ["start recording", "record video", "begin recording"],
            "zh": ["开始录制", "录制视频"],
        },
        params_schema={
            "filename": {"type": "string", "required": False},
            "duration_s": {"type": "float", "required": False},
        },
        description="Start recording video",
    ),

    CameraOperation.STOP_RECORDING: OperationDef(
        name="stop_recording",
        action="camera.stop_recording",
        keywords={
            "en": ["stop recording", "end recording"],
            "zh": ["停止录制", "结束录制"],
        },
        params_schema={},
        description="Stop recording video",
    ),
}


class CameraParser(ParserBase):
    """Parser for camera instructions."""

    def __init__(self):
        self._operations = CAMERA_OPERATIONS

    def parse(self, instruction: str) -> Dict[str, Any]:
        """Parse camera instruction."""
        language = self.detect_language(instruction)

        best_match = None
        best_confidence = 0.0

        for op_type, op_def in self._operations.items():
            matches, confidence = op_def.matches(instruction, language)
            if matches and confidence > best_confidence:
                best_match = op_type
                best_confidence = confidence

        if not best_match:
            return {
                "operation": None,
                "action": None,
                "params": {},
                "confidence": 0.0,
                "language": language,
                "description": instruction,
            }

        # Extract parameters
        params = {}

        # Filename
        fn_match = re.search(r'(?:as|to|named?)\s+["\']?(\w+)["\']?', instruction, re.IGNORECASE)
        if fn_match:
            params['filename'] = fn_match.group(1)

        # Duration
        dur_match = re.search(r'for\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)', instruction, re.IGNORECASE)
        if dur_match:
            params['duration_s'] = float(dur_match.group(1))

        op_def = self._operations[best_match]
        return {
            "operation": best_match.value,
            "action": op_def.action,
            "params": params,
            "confidence": best_confidence,
            "language": language,
            "description": instruction,
        }


class CameraPlugin(PluginBase):
    """
    Plugin for camera/imaging systems.

    Supports:
    - USB cameras
    - SSH video streams
    - IP cameras
    - Microscope cameras
    """

    name = "camera"
    device_type = "camera"
    version = "1.0.0"
    description = "Imaging systems (USB, SSH, IP cameras)"

    def _register_operations(self):
        """Register camera operations."""
        for op_type, op_def in CAMERA_OPERATIONS.items():
            self.register_operation(op_def)

    def _create_parser(self) -> ParserBase:
        """Create camera parser."""
        return CameraParser()

    def get_supported_operations(self) -> List[str]:
        """Get list of supported operation names."""
        return [op.value for op in CameraOperation]
