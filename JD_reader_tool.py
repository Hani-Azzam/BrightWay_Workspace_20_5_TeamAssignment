### JD_reader_tool.py is an identical copy of CV_reader_tool.py. This is bad code design and should be changed.

from google.ai.generativelanguage_v1beta import Schema
from pathlib import Path

from base.tool_base import ToolBase, ToolResult, ToolSchema

_MAX_FILENAME_LEN = 90

class JDReaderTool(ToolBase):

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="JD_reader_tool",
            description=(
                "A tool that reads a plain-text job description file and returns its full content. "
                "Use this when you need to read the entire job description file. "
                "The job description (jd) file must be of type .txt file"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name, e.g. 'jd_01.txt'",
                        "maxLength": _MAX_FILENAME_LEN,
                    },
                },
                "required": ["filename"],
            },
        )

    # A tool that reads a plain-text jd file and returns its full content.
    # It must:
    # Accept a filename (not a full path)
    # Validate that the file exists in the data/ directory and is a .txt file
    # Return the file content as a string
    # Raise ValueError with a clear message if validation fails
    def run(self, filename: str) -> ToolResult:
        # 1. Enforce that the input is strictly a filename, not a path
        if Path(filename).name != filename:
            return ToolResult(
                error=f"Input must be a simple filename, not a path.",
                is_idempotent=True,
            )

        # 2. Strict extension check
        if not filename.lower().endswith(".txt"):
            return ToolResult(
                error=f"File must have a .txt extension.",
                is_idempotent=True,
            )

        # 3. Combine safely and verify physical existence
        target_file = Path("data") / filename
        if not target_file.is_file():
            return ToolResult(
                error=f"'{filename}' does not exist in the 'data' directory.",
                is_idempotent=True,
            )

        # Read the entire file as a string (specify encoding to prevent crashes)
        result = target_file.read_text(encoding="utf-8")

        return ToolResult(value=result, is_idempotent=True)
