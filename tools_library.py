import os
import shutil
import requests
import praw
import feedparser
import urllib.parse
import chromadb
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# --- CATEGORIZATION HELPERS ---
def get_all_ipo_names():
    categorized = {"Mainboard": [], "SME": []}
    try:
        r = requests.get("https://www.ipopremium.in/ipo", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json().get("data", [])
        for d in data:
            raw_name = d.get("name", "")
            clean_name = BeautifulSoup(raw_name, "html.parser").get_text(" ", strip=True)
            if "SME" in clean_name:
                categorized["SME"].append(clean_name)
            else:
                categorized["Mainboard"].append(clean_name)
        return categorized
    except Exception as e:
        return {"Mainboard": [], "SME": []}


def get_concurrent_ipos(target_name, category_filter="All"):
    """Returns active IPOs filtered by category."""
    peers = []
    try:
        r = requests.get("https://www.ipopremium.in/ipo", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        for d in r.json().get("data", []):
            name = BeautifulSoup(d.get("name", ""), "html.parser").get_text(" ", strip=True)
            status = d.get("status", "").lower()

            if name == target_name: continue
            if "listed" in status: continue

            is_sme = "SME" in name
            if category_filter == "Mainboard" and is_sme: continue
            if category_filter == "SME" and not is_sme: continue

            peers.append(name)
        return peers
    except:
        return []


# --- WORKER 1: IPO DETAILS ---
def fetch_ipo_details(ipo_name: str):
    """Fetches GMP, Dates, Price, and SLUG."""
    try:
        r = requests.get("https://www.ipopremium.in/ipo", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json().get("data", [])
        clean_names = [BeautifulSoup(d.get("name", ""), "html.parser").get_text(" ", strip=True) for d in data]
        match = process.extractOne(ipo_name, clean_names, scorer=fuzz.QRatio)

        if match and match[1] > 80:
            target = match[0]
            for d in data:
                if BeautifulSoup(d.get("name", ""), "html.parser").get_text(" ", strip=True) == target:
                    return {
                        "id": d.get("id"),
                        "slug": d.get("slug", ""),
                        "Company": target,
                        "GMP": d.get("premium", "N/A"),
                        "Price Band": d.get("price", "N/A"),
                        "Open Date": d.get("open", "N/A"),
                        "Close Date": d.get("close", "N/A"),
                        "Listing Date": d.get("listing", "N/A"),
                        "Status": d.get("status", "N/A"),
                        "Issue Size": d.get("size", "N/A")
                    }
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Not Found"}


# --- WORKER 2: SENTIMENT ---
def fetch_sentiment(ipo_name: str, source: str = "all"):
    texts = []
    if source in ["reddit", "all"]:
        try:
            reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT", "Bot/1.0")
            )
            for sub in reddit.subreddit("all").search(f"{ipo_name} IPO", limit=5):
                texts.append(f"[Reddit]: {sub.title}")
        except:
            pass

    if source in ["news", "all"]:
        try:
            q = urllib.parse.quote(f"{ipo_name} IPO")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en")
            texts.extend([f"[News]: {e.title}" for e in feed.entries[:5]])
        except:
            pass

    return "\n".join(texts) if texts else "No sentiment data found."


# --- WORKER 3: RHP DOCUMENT ---
def query_rhp(ipo_name, query, vector_store=None):
    if not vector_store:
        return "⚠️ RHP Document is not loaded."

    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    context_q_system_prompt = (
        "Given a chat history, formulate a standalone question. Do NOT answer it."
    )
    context_q_prompt = ChatPromptTemplate.from_messages(
        [("system", context_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    qa_system_prompt = (
        "You are an expert financial analyst reading an IPO RHP document. "
        "Answer the question based strictly on the context. "
        "If not found, strictly say 'I cannot find this information in the RHP document'.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )

    chain = create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))

    try:
        response = chain.invoke({"input": query, "chat_history": []})
        return f"[Source: RHP Document]\n{response['answer']}"
    except Exception as e:
        return f"Error querying RHP: {str(e)}"


# --- PDF & VECTOR STORE HELPERS ---
def download_pdf_logic(details):
    """Scrapes the page to find the best PDF link."""
    ipo_id = details.get('id')
    slug = details.get('slug')
    os.makedirs("pdfs", exist_ok=True)
    save_path = os.path.join("pdfs", f"{ipo_id}.pdf")

    if os.path.exists(save_path): return save_path

    page_url = f"https://www.ipopremium.in/view/ipo/{ipo_id}/{slug}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(page_url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")

        target_url = None
        candidates = []
        for a in soup.find_all("a", href=True):
            text = a.get_text().lower()
            if "rhp" in text or "drhp" in text or "anchor" in text:
                candidates.append({"link": a["href"], "text": text})

        # Prioritize RHP > DRHP > Anchor (Fixed Syntax Logic)
        for c in candidates:
            if "rhp" in c["text"] and "drhp" not in c["text"]:
                target_url = c["link"]
                break

        if not target_url:
            for c in candidates:
                if "drhp" in c["text"]:
                    target_url = c["link"]
                    break

        if not target_url:
            target_url = f"https://assets.ipopremium.in/images/ipo/{ipo_id}_rhp.pdf"  # Fallback

        if target_url:
            if not target_url.startswith("http"): target_url = "https://www.ipopremium.in" + target_url
            pdf_resp = requests.get(target_url, headers=headers, stream=True)
            if pdf_resp.status_code == 200:
                with open(save_path, "wb") as f: f.write(pdf_resp.content)
                return save_path
    except:
        pass
    return None


def build_vs_logic(pdf_path):
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

    db_path = "./chroma_db_storage"
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except:
            pass

    client = chromadb.PersistentClient(path=db_path)
    return Chroma.from_documents(documents=splits, embedding=emb, client=client, collection_name="ipo_collection")