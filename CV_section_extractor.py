from google.ai.generativelanguage_v1beta import Schema
from pathlib import Path

from base.tool_base import ToolBase, ToolResult, ToolSchema

_MAX_FILENAME_LEN = 90

class CVSectionExtractor(ToolBase):

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="CV_section_extractor",
            description=(
                "A tool that returns the text of one named section from a CV file."
                "Use this when you need to read a specific section from a CV file. "
                "The CV file must be of type .txt file"
                "Note that section names are often wrapped in double number signs (e.g. ##Contact Information##)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename_and_section": {
                        "type": "string",
                        "description": "File name and section from it to extract. "
                                       "Input in the format filename|section_name (e.g. cv_alice.txt|WORK EXPERIENCE"
                                       "or cv_alice.txt|##SKILLS##)",
                        "maxLength": _MAX_FILENAME_LEN,
                    },
                },
                "required": ["filename_and_section"],
            },
        )

    # Accept input in the format filename|section_name (e.g. cv_alice.txt|WORK EXPERIENCE)
    # Validate the format and that the file exists
    # Search the CV text for the section heading and return everything between that heading and the next heading
    # (or end of file)
    # Return a clear message if the section is not found
    def run(self, filename_and_section: str) -> ToolResult:
        # Split the input string exactly at the '|' character
        if "|" not in filename_and_section:
            return ToolResult(
                error="Invalid input format. Must be exactly filename|section_name",
                is_idempotent=True,
            )
        else:
            filename, section = filename_and_section.split("|", maxsplit=1)

        # 1. Enforce that the input is strictly a filename, not a path
        if Path(filename).name != filename:
            return ToolResult(
                error=f"Filename part of input must be a simple filename, not a path.",
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
        file = target_file.read_text(encoding="utf-8")

        # Extract section
        section = section.strip()
        if section not in file:
            return ToolResult(
                error="Section not found in the CV file.",
                is_idempotent=True,
            )
        else:
            print(f"Extracting: {section=}")
            result = self.extract_cv_section(file, section)


        return ToolResult(value=result, is_idempotent=True)


    def extract_cv_section(self, cv_text: str, section_name: str) -> str:
        # Split the CV into individual lines
        lines = cv_text.splitlines()

        section_content = []
        inside_target_section = False

        # Normalize the target name to make matching reliable (e.g., "WORK EXPERIENCE")
        target_heading = section_name.strip().upper()

        for line in lines:
            cleaned_line = line.strip()

            # Skip empty lines when looking for headings
            if not cleaned_line:
                if inside_target_section:
                    section_content.append(line)  # Keep formatting inside the section
                continue

            # Check if the current line is a new section heading (fully uppercase)
            is_heading = cleaned_line.isupper() and len(cleaned_line) > 2

            if is_heading:
                # If we were already tracking our section, another heading means it's over
                if inside_target_section:
                    break

                # If this heading matches our target, start collecting lines
                if cleaned_line == target_heading:
                    inside_target_section = True
                    continue  # Skip printing the heading itself

            # If we are inside the correct section, collect the text
            if inside_target_section:
                section_content.append(line)

        # Join the collected lines back into a single string
        result = "\n".join(section_content).strip()

        return result if result else f"Section '{section_name}' not found or empty."
