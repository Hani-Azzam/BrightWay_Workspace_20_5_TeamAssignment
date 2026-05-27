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

from agents.tool_agent import  ReActConfig, ToolAgent
from services.llm_client import LlmClient, LlmConfig
from services.tool_executor import ToolExecutor
from tools.CV_reader_tool import CVReaderTool
from tools.CV_section_extractor import CVSectionExtractor
from tools.job_match_scorer_tool import JobMatchScorerTool
from agents.AgentExecutor import AgentExecutor

# Load environment variables from .env file
load_dotenv(override=True)

llm = LlmClient(LlmConfig(
    api_key=os.getenv('GEMINI_API_KEY'),
    model_name=os.getenv('GEMINI_MODEL_NAME'),
    temperature=float(os.getenv('GEMINI_TEMPERATURE', 0.1)),
))

executor = ToolExecutor(max_retries=2, base_delay=0.5)
executor.register(CVReaderTool())
executor.register(CVSectionExtractor())
executor.register(JobMatchScorerTool())

print("=== assigment_06b  - Hiring scorer ====\n")
print("Registered tools:")
for schema in executor.tool_schemas():
    print(f" {schema['name']:<16} {schema['description']}")

cv_reader = CVReaderTool()
cv_extractor = CVSectionExtractor()
cv_scorer = JobMatchScorerTool()

agent = AgentExecutor(llm, executor)

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

        if user_input.lower() == "stam":
            # call tools - assume user gives file name
            if "|" not in user_input:
                result = cv_reader.run(user_input)
            else:
                # call the job match scorer
                result = cv_scorer.run(user_input)
            # else:
            #     result = cv_extractor.run(user_input)
            if result.ok:
                # print cv or section
                print(result.value)
            else:
                # print error
                print(result.error)

        # 'trace' command: show the step-by-step log for the last query
        if user_input.lower() == "trace":
            traces = executor.get_traces()
            if not traces:
                print("[No trace yet - ask a question first]\n")
                continue
            print("\n[Plan / Act/ Observe trace]")
            for t in traces:
                tool_col = f"[{t.tool_name}]" if t.tool_name else "     "
                ms_str   = f"  ({t.duration_ms:.0f} ms)" if t.duration_ms > 0 else ""
                print(
                    f"  step {t.step}  {t.phase:<8}  "
                    f"{tool_col:<16}  {t.details[:90]}{ms_str}"
                )
            print()
            continue

        answer = agent.chat(user_input)
        print(f"\nAgent: {answer}\n")
