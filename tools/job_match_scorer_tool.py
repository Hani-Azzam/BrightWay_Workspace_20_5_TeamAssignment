from pathlib import Path
import os
import sys

from google.ai.generativelanguage_v1beta import Schema
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from base.tool_base import ToolBase, ToolResult, ToolSchema
from agents.conversation_agent import ConversationAgent
from services.llm_client import LlmClient, LlmConfig

from dotenv import load_dotenv


load_dotenv()

config = LlmConfig(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name=os.getenv("GEMINI_MODEL_NAME"),
    temperature=float(os.getenv("GEMINI_TEMPERATURE")),
)


_MAX_FILENAME_LEN = 180


class JobMatchScorerTool(ToolBase):

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="job_match_scorer_tool",
            description=(
                "A tool that takes a CV filename and a job description filename in the format cv_file|jd_file "
                "and uses an LLM to produce a short assessment of how well the candidate fits the role. "
                "Use this when you need a short 3-5 sentence assessment of how well the candidate fits the role."
                "Both CV files must be of type .txt"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cv_filename_and_job_description_filename": {
                        "type": "string",
                        "description": "CV file name and job description file name of type .txt."
                                       "Input must be in the format cv_filename|job_description_filename "
                                       "(e.g. cv_alice.txt|job_requirements.txt)",
                        "maxLength": _MAX_FILENAME_LEN,
                    },
                },
                "required": ["cv_filename_and_job_description_filename"],
            },
        )

    def run(self, cv_filename_and_job_description_filename: str) -> ToolResult:
        # Split the input string exactly at the '|' character
        if "|" not in cv_filename_and_job_description_filename:
            return ToolResult(
                error="Invalid input format. Must be exactly format cv_filename|job_description_filename",
                is_idempotent=True,
            )
        else:
            cv_filename, jd_filename = cv_filename_and_job_description_filename.split("|", maxsplit=1)

        # 1. Enforce that the input is strictly a filename, not a path
        if Path(cv_filename).name != cv_filename or Path(jd_filename).name != jd_filename:
            return ToolResult(
                error=f"Both filenames must be a simple filename, not a paths.",
                is_idempotent=True,
            )

        # 2. Strict extension check
        if not cv_filename.lower().endswith(".txt") or not jd_filename.lower().endswith(".txt"):
            return ToolResult(
                error=f"Both files must have a .txt extension.",
                is_idempotent=True,
            )

        # 3. Combine safely and verify physical existence
        cv_target_file = Path("data") / cv_filename
        jd_target_file = Path("data") / jd_filename
        if not cv_target_file.is_file():
            return ToolResult(
                error=f"'{cv_filename}' does not exist in the 'data' directory.",
                is_idempotent=True,
            )
        if not jd_target_file.is_file():
            return ToolResult(
                error=f"'{jd_filename}' does not exist in the 'data' directory.",
                is_idempotent=True,
            )

        # Read both files as strings (specify encoding to prevent crashes)
        cv_file = cv_target_file.read_text(encoding="utf-8")
        jd_file = jd_target_file.read_text(encoding="utf-8")

        # Create llm instance and call it
        llm = LlmClient(config)
        system_message = ("You are a hiring assistant. You receive a candidates CV and a job description, and you must "
                          "produce a short assessment of how well the candidate fits the role. Write only 3-5 sentences.")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "question: {question}, cv: {CV}, jb: {JD}"),
        ])
        question = "I'm giving you the CV and job description. Asses if candidate fits the role."
        response: str = llm.build_chain(prompt).invoke({"question": question, "CV": cv_file, "JD": jd_file})


        return ToolResult(value=response, is_idempotent=True)
