"""
A minimal data-question agent.

The agent takes a natural-language question, lets the LLM decide whether to call
the `query_data` tool, runs the tool, feeds the result back to the LLM, and
returns a final natural-language answer.

The agent depends only on the `LLM` interface from llm.py -- it must not care
which provider is behind it.

Protocol (what the LLM is told to produce):
  - To query the data, the model replies with ONLY a JSON object:
        {"tool": "query_data", "sql": "SELECT ..."}
  - To answer, the model replies with ONLY:
        {"answer": "..."}

Your job is to implement the loop in `Agent.answer` (see TODOs).
"""

from __future__ import annotations

import json
from typing import List

from llm import LLM, Message, get_llm
from tools import load_programs_db, query_data

SYSTEM_PROMPT = """You are a data assistant. Answer questions about a SQLite table
named `programs` with columns:
  program_id (TEXT), program_name (TEXT), region (TEXT), sector (TEXT),
  year (INTEGER), budget_usd (INTEGER), people_served (INTEGER), status (TEXT).

sector is one of: health, education, energy.

To read the data, reply with ONLY a JSON object: {"tool": "query_data", "sql": "SELECT ..."}
When you can answer, reply with ONLY: {"answer": "..."}
If the data cannot answer the question, reply with {"answer": "..."} saying so.
Reply with JSON only -- no extra text.
"""

MAX_STEPS = 4


class Agent:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm or get_llm()
        self.con = load_programs_db()

    def answer(self, question: str) -> str:
        """
        Run the question through the agent loop and return a final answer string.

        TODO(candidate):
          1. Validate `question` (non-empty string).
          2. Build the message list (system + user) and loop up to MAX_STEPS:
               - call self.llm.complete(messages)
               - parse the JSON reply
               - if it's a tool call, run query_data(...) and append the result
                 as a {"role": "tool", "content": <json>} message, then continue
               - if it's an answer, return it
          3. Add basic reliability: handle malformed JSON / tool errors, and
             retry the LLM call once before giving up.
        """

        #step 1: validate question 
        if not isinstance(question,str) or not question.strip():
            raise ValueError("Error: Invalid Input")

        #step 2: build messages - with dictionary
        messages=[
            #system message-----
            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },
            #user message-----
            {
                "role":"user",
                "content":question
            }

        ]

        #step 3- self call llm and parse json

        #for loop used to go through all steps required 
        for step in range(MAX_STEPS):
            response=self.llm.complete(messages) # calling llm - what to do

            #convert llm json response to python dict
            try:
                parsed_response=json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON error:{e}")
            except Exception as e:
                raise ValueError(f"LLM error:{e}")

            #check whether llm requested a tool
            if "tool" in parsed_response:
              found_key=parsed_response.get("tool") #get name of tool requested

              if found_key=="query_data":
                  load_sql=parsed_response.get("sql") #extract sql query gen by llm
              else:
                  raise ValueError("Unknown tool requested")

              if not isinstance(load_sql, str) or not load_sql.strip():
                    raise ValueError("Tool call is missing a valid SQL query")

              result=query_data(load_sql,self.con)

              #add to database result so llm can get answer 
              messages.append({
                    "role":"tool",
                    "content":json.dumps(result)
                  })

            #returning answer to user 
            if "answer" in parsed_response:
                return parsed_response.get("answer")

        raise RuntimeError("Maximum steps reached no answer acquired")

            
                
            


if __name__ == "__main__":
    import sys

    agent = Agent()
    q = " ".join(sys.argv[1:]) or "How many programs are in the education sector?"
    print(agent.answer(q))
