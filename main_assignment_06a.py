'''
A simple REPL that creates both tools and lets the user test them directly — without an agent in the loop.

Each user input should go straight to the tool of the user's choice (add a command to pick which tool to call).
Print the raw tool output.

This is a tool test harness, not a conversational agent. Its purpose is to verify that validation and extraction work
correctly before the tools are handed to an LLM.
'''

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

# from agents.tool_agent import  ReActConfig, ToolAgent
# from services.llm_client import LlmClient, LlmConfig
from services.tool_executor import ToolExecutor
from tools.CV_reader_tool import CVReaderTool
from tools.CV_section_extractor import CVSectionExtractor


# Load environment variables from .env file
load_dotenv(override=True)


executor = ToolExecutor(max_retries=2, base_delay=0.5)
executor.register(CVReaderTool())
executor.register(CVSectionExtractor())

print("=== assigment_06a  - Tool Agent (ReAct) ====\n")
print("Registered tools:")
for schema in executor.tool_schemas():
    print(f" {schema['name']:<16} {schema['description']}")

cv_reader = CVReaderTool()
cv_extractor = CVSectionExtractor()

if __name__ == '__main__':
    # REPL loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl-C or piped input exhausted — clean exit
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)

        # call tools - assume user gives file name
        if "|" not in user_input:
            result = cv_reader.run(user_input)
        else:
            result = cv_extractor.run(user_input)

        if result.ok:
            # print cv or section
            print(result.value)
        else:
            # print error
            print(result.error)


