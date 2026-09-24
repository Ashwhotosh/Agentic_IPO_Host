import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools_library import fetch_ipo_details, fetch_sentiment, query_rhp


def execute_peer_comparison(target_ipo, selected_peers, vector_store):
    yield "🔄 **Phase 1: Analyzing Fundamentals (RHP)...**"

    rhp_fundamentals = "Target RHP not loaded."
    if vector_store:
        yield "📖 Reading 'Industry Comparison' from RHP..."
        q = "Extract the 'Comparison with Listed Industry Peers' table. List Peer Companies and P/E, EPS, RoNW."
        rhp_fundamentals = query_rhp(target_ipo, q, vector_store=vector_store)

    yield "📊 **Phase 2: Gathering Live Market Data...**"
    market_data = {}
    companies = [target_ipo] + selected_peers

    for co in companies:
        role = "TARGET" if co == target_ipo else "PEER"
        yield f"🕵️ Scouting: **{co}**..."
        market_data[co] = {
            "Details": fetch_ipo_details(co),
            "Sentiment": fetch_sentiment(co, source="all"),
            "Role": role
        }

    yield "⚖️ **Phase 3: Calculating Rankings...**"
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="qwen/qwen3.8-27b", temperature=0.1, max_tokens=400)

    data_str = json.dumps(market_data, indent=2, default=str)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Portfolio Manager. Compare Target vs Peers. Rank them #1 to Last based on Fundamentals (P/E) and Hype (GMP). Create a table and justify."),
        ("human", """
        **RHP Analysis:** {rhp_data}
        **Live Data:** {json_data}
        Generate the Report.
        """)
    ])

    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"rhp_data": rhp_fundamentals, "json_data": data_str})
    yield analysis