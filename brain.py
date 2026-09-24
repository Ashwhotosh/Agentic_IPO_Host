import os
import json
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Literal
from tools_library import fetch_ipo_details, fetch_sentiment, query_rhp
from dotenv import load_dotenv

load_dotenv()


class ToolCall(BaseModel):
    tool_name: Literal["gmp_tool", "sentiment_tool", "rhp_tool"]
    arguments: str = Field(description="The specific question or argument")


class Plan(BaseModel):
    steps: List[ToolCall] = Field(description="List of tools to execute")


def execute_brain(user_query, ipo_name, vector_store):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        yield "❌ Error: GROQ_API_KEY missing."
        return

    llm = ChatGroq(api_key=api_key, model="qwen/qwen3.8-27b", temperature=0, max_tokens=400)

    # 1. Planning
    structured_llm = llm.with_structured_output(Plan)
    system_prompt = f"""
    You are an expert IPO Analyst Brain.
    User Query: "{user_query}"
    Current IPO: "{ipo_name}"

    Break the query into steps. Available Tools:
    1. 'gmp_tool': For Price, GMP, Open/Close Dates, Listing, Status. (Arg: 'details')
    2. 'sentiment_tool': For Market Mood. (Arg: 'reddit', 'news', or 'all')
    3. 'rhp_tool': Strictly for Document questions (Peers, Risks, Promoters).
       - IMPORTANT: Copy the user's specific document question verbatim.
    """

    try:
        plan = structured_llm.invoke(system_prompt)
    except Exception as e:
        yield f"Error in planning: {e}"
        return

    results = []

    # 2. Execution
    for step in plan.steps:
        yield f"⚙️ **Executing:** {step.tool_name}..."

        output = ""
        if step.tool_name == "gmp_tool":
            output = str(fetch_ipo_details(ipo_name))

        elif step.tool_name == "sentiment_tool":
            output = fetch_sentiment(ipo_name, source=step.arguments)

        elif step.tool_name == "rhp_tool":
            # --- FIX: RAW QUERY INJECTION ---
            search_query = step.arguments
            if len(plan.steps) == 1:
                search_query = user_query

            output = query_rhp(ipo_name, query=search_query, vector_store=vector_store)

        results.append(f"--- {step.tool_name.upper()} RESULT ---\n{output}\n")
        yield f"✅ {step.tool_name} Complete."

    # 3. Synthesis
    yield "🧠 **Synthesizing Final Answer...**"

    final_prompt = f"""
    User Query: {user_query}
    Data:
    {"".join(results)}

    Answer professionally using only the data above.
    """
    final_response = llm.invoke(final_prompt).content
    yield final_response