from google.ai.generativelanguage_v1beta import Schema
from pathlib import Path

from base.tool_base import ToolBase, ToolResult, ToolSchema

_MAX_FILENAME_LEN = 90

class CVWriterTool(ToolBase):

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="CV_writer_tool",
            description=(
                "A tool that writes a new plain-text CV and saves it in data/cv as a .txt file. "
                "Use this when you want to write a new CV file. "
                "The CV sections must have titles with **double asterisks** around them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "File name, e.g. 'new_cv.txt'",
                        "maxLength": _MAX_FILENAME_LEN,
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content of the new CV file (contains stuff like candidate name, "
                                       "education, experience, skills etc.)",
                        "minLength": 1,
                    },
                },
                "required": ["filename", "content"],
            },
        )

    # A tool that saves a new CV version to data/cv/ as a .txt file.
    # It must:
    # Accept filename (output name, must end in .txt) and content (the full new CV text)
    # Validate that filename ends in .txt and that content is not empty
    # Write the file and return ToolResult(value="CV saved to data/cv/<filename>")
    # Be marked is_idempotent=False — writing a file is a side effect; the executor must not retry it automatically
    def run(self, filename: str, content: str) -> ToolResult:
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

        # 3. Enforce text 'content' is not empty
        if not content:
            return ToolResult(
                error=f"Content must not be empty.",
                is_idempotent=True, #it is fine to retry here since the file has not been created yet
            )

        # 4. Combine safely and verify physical existence - NEED CHANGE - TODO
        target_file = Path("data/cv") / filename
        # if not target_file.is_file():
        #     return ToolResult(
        #         error=f"'{filename}' does not exist in the 'data' directory.",
        #         is_idempotent=False,
        #     )

        # TODO: Check the name is valid file name, and doesnt contain illegal characters!
        # Open (or create new) file and write the 'content' to it (specify encoding to prevent crashes)
        with open(target_file, "w", encoding="utf-8") as file:
            file.write(content)
            print(f"    [Agent] CV written to {target_file}")

        return ToolResult(value="CV saved to data/cv/<filename>", is_idempotent=False)
