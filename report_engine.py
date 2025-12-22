import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools_library import fetch_ipo_details, fetch_sentiment, query_rhp


def generate_section(section_title, specific_questions, vector_store, ipo_name, llm):
    if not vector_store: return f"## {section_title}\n*RHP Not Loaded.*\n"

    raw_context = []
    for q in specific_questions:
        ans = query_rhp(ipo_name, q, vector_store=vector_store)
        raw_context.append(f"Q: {q}\nA: {ans}")

    context_str = "\n\n".join(raw_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Senior Equity Analyst. Write a detailed section based on the raw data. Use tables and bullets."),
        ("human", f"**Section:** {section_title}\n**Data:**\n{context_str}\n\nWrite the section content.")
    ])
    return (prompt | llm | StrOutputParser()).invoke({})


def generate_deep_dive_report(ipo_name, vector_store):
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.2)
    yield "📊 **Initializing Deep Dive Analysis...**"

    market_data = fetch_ipo_details(ipo_name)
    sentiment_data = fetch_sentiment(ipo_name, source="all")

    full_report = [f"# 📑 Investment Report: {ipo_name}\n---\n"]

    chapters = {
        "1. Executive Summary": {"type": "intro"},
        "2. Business Model": {"questions": ["Business model?", "Products?", "Revenue model?"], "type": "rhp"},
        "3. Financial Health": {
            "questions": ["Financial statements last 3 years?", "Revenue/Profit trends?", "EPS, RoNW?"], "type": "rhp"},
        "4. Objects & Promoters": {"questions": ["Objects of issue?", "Promoter profiles?", "OFS details?"],
                                   "type": "rhp"},
        "5. Risks & Litigation": {"questions": ["Internal risk factors?", "Litigations?", "Regulatory risks?"],
                                  "type": "rhp"},
        "6. Peer Comparison": {"questions": ["Who are listed peers?", "Compare financial metrics?"], "type": "rhp"}
    }

    for title, config in chapters.items():
        yield f"✍️ **Drafting: {title}...**"

        if config["type"] == "intro":
            intro_prompt = f"Write Executive Summary. Data: {str(market_data)} Sentiment: {sentiment_data}"
            response = llm.invoke(intro_prompt).content
            full_report.append(f"## {title}\n{response}\n")
        elif config["type"] == "rhp":
            content = generate_section(title, config["questions"], vector_store, ipo_name, llm)
            full_report.append(f"## {title}\n{content}\n")

        yield f"✅ {title} Complete."

    yield "⚖️ **Formulating Verdict...**"
    verdict = llm.invoke(f"Write Final Verdict based on: {''.join(full_report)}").content
    full_report.append(f"## 7. Final Verdict\n{verdict}")

    yield "\n".join(full_report)