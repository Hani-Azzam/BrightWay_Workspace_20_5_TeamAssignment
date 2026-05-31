'''
A simple REPL that creates both tools and lets the user test them directly — without an agent in the loop.

Each user input should go straight to the tool of the user's choice (add a command to pick which tool to call).
Print the raw tool output.

This is a tool test harness, not a conversational agent. Its purpose is to verify that validation and extraction work
correctly before the tools are handed to an LLM.
'''

import os
import sys
from pathlib import Path


sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

from agents.tool_agent import  ReActConfig, ToolAgent
from services.llm_client import LlmClient, LlmConfig
from services.tool_executor import ToolExecutor
from tools.CV_reader_tool import CVReaderTool
from tools.CV_section_extractor import CVSectionExtractor
from tools.job_match_scorer_tool import JobMatchScorerTool
from tools.CV_writer import CVWriterTool
from tools.JD_reader_tool import JDReaderTool
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
executor.register(CVWriterTool())
executor.register(JDReaderTool())

print("=== assigment_06c  - CV Job Fit Agent ====\n")
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

        # 'fit <jd_file> <cv_file>' command: suggest a fit for a given cv to a given job description, then ask user for
        # approval, if they approve, changes are saved in a new cv file, otherwise, no change is made.
        if user_input.startswith("fit ") or user_input == "fit":
            print("The string starts with the word 'fit'.")
            ### Phase 1 — Analysis ###
            jd_file, cv_file = "none", "none"
            # Split the input into words
            words = user_input.split()
            # command gets 'fit', file_list gets ['file_a', 'file_b', 'file_c']
            command, *file_list = words
            # We expect exactly two file arguments
            if len(file_list) == 2:
                jd_file, cv_file = file_list
            else:
                print(f"Error: 'fit' expected 2 arguments, but got {len(file_list)}")
            # The arguments must end with .txt and exist in /data, but that can be verified by the tools called in chat()
            # Build a prompt to instruct the agent to read cv and jb and create final_answer
            analysis_request = (f"Do the following: \n1. Read the job description file using JD_reader_tool\n"
                                f"2. Read the CV file using CV_reader_tool\n"
                                f"3. Produce a final_answer containing:\n"
                                f"  - A list of tech keywords present in the JD but missing or weak in the CV.\n"
                                f"  - Section-by-section suggestions (e.g. 'rewrite the summary to mention microservices').\n"
                                f"  - A list of tech terms to highlight in the new CV.\n\n"
                                f"The job description file name is {jd_file} and the CV file name is {cv_file}.\n"
                                f"DO NOT USE CV_writer_tool.")
            analysis = agent.chat(analysis_request)
            print(f"\nAgent: {analysis}")

            ### Phase 1.5 - Confirmation Gate ###
            print("\nAgent: Apply these changes and create a new CV? (yes/no):\n")
            user_input = input("You: ").strip()
            if user_input.lower() in ("yes", "y"):
                print(f"\nAgent: How would you like to name the output file? Make sure the name ends with .txt.\n"
                      f"(press enter for the default name {cv_file}_v2.txt)")
                user_input = input("You: ").strip()
            else:
                print("\nAgent: No changes made.")
                continue


            ### Phase 2 - Generation ###
            # Call agent.chat() a second time with a prompt that includes:
            # The JD filename and CV filename (so the agent re-reads both)
            # The analysis from Phase 1 (pass it as context in the prompt)
            # The output filename chosen by the user
            # Instructions to rewrite the CV, apply the suggestions, wrap every tech keyword in **double asterisks**,
            # and save it using cv_writer.
            # Rules the prompt must enforce for the rewritten CV:
            # Do not invent new roles, degrees, or companies
            # Add missing tech keywords only where the candidate's real experience supports it
            # Improve the summary to target the specific role
            # Wrap every tech keyword and tool name in **double asterisks**
            # After agent.chat() returns, print the agent's final message and the path to the saved file.
            if user_input.lower() == "": # you may want to check if was 'exit' todo
                new_cv_filename = cv_file.removesuffix(".txt")
                new_cv_filename = f"{new_cv_filename}_v2.txt"
            else:
                new_cv_filename = user_input
            generation_request = (f"Do the following: \n1. Read the job description file using JD_reader_tool\n"
                                  f"2. Read the CV file using CV_reader_tool\n"
                                  f"The job description file name is {jd_file} and the CV file name is {cv_file}.\n"
                                  f"3. Read the provided analysis of the CV.\n"
                                  f"4. Rewrite the CV and apply the suggestions from the analysis, and save it using"
                                  f" CV_writer_tool. The new CV file must be named {new_cv_filename}.\n"
                                  f"Follow these rules when writing the new CV:\n"
                                  f" - Do not invent new roles, degrees, or companies.\n"
                                  f" - Add missing tech keywords only where the candidate's real experience supports it.\n"
                                  f" - Improve the summary to target the specific role.\n"
                                  f" - Wrap every tech keyword in **double asterisks**.\n\n\n"
                                  f"The analysis is:\n {analysis}")
            answer = agent.chat(generation_request)
            print(f"\nAgent: {answer}")
            print("[File has been saved to data\cv\\" + f"{new_cv_filename}" + "]")
            continue



        answer = agent.chat(user_input)
        print(f"\nAgent: {answer}\n")
