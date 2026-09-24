import os
import shutil
import requests
import praw
import feedparser
import urllib.parse
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz
import cloudscraper
# --- SWITCH TO FAISS (RAM DB) ---
from langchain_community.vectorstores import FAISS 
# --------------------------------
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- CATEGORIZATION HELPERS ---
def _scrape_ipo_data():
    try:
        scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True})
        r = scraper.get("https://www.ipopremium.in/", timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        data = []
        rows = soup.select("table tbody tr")
        for row in rows:
            cols = row.select("td")
            if len(cols) < 7: continue
            
            a_tag = cols[0].find("a")
            if not a_tag: continue
            
            name = a_tag.get_text(strip=True)
            href = a_tag.get("href", "")
            
            parts = [p for p in href.split("/") if p]
            ipo_id = parts[-2] if len(parts) >= 2 else ""
            slug = parts[-1] if len(parts) >= 1 else ""
            
            data.append({
                "id": ipo_id,
                "slug": slug,
                "name": name,
                "premium": cols[2].get_text(strip=True),
                "price": cols[5].get_text(strip=True),
                "open": cols[3].get_text(strip=True),
                "close": cols[4].get_text(strip=True),
                "listing": cols[6].get_text(strip=True),
                "status": "upcoming",
                "size": "N/A"
            })
        return data
    except Exception as e:
        return [{"id": "", "slug": "", "name": f"Scrape Error: {str(e)}", "premium": "", "price": "", "open": "", "close": "", "listing": "", "status": "upcoming", "size": ""}]

def get_all_ipo_names():
    categorized = {"Mainboard": [], "SME": []}
    try:
        data = _scrape_ipo_data()
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
        for d in _scrape_ipo_data():
            name = BeautifulSoup(d.get("name", ""), "html.parser").get_text(" ", strip=True)
            status = d.get("status", "").lower()
            
            if name == target_name: continue
            if "listed" in status: continue
            
            is_sme = "SME" in name
            if category_filter == "Mainboard" and is_sme: continue
            if category_filter == "SME" and not is_sme: continue
            
            peers.append(name)
        return peers
    except: return []

# --- WORKER 1: IPO DETAILS ---
def fetch_ipo_details(ipo_name: str):
    """Fetches GMP, Dates, Price, and SLUG."""
    try:
        data = _scrape_ipo_data()
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
        except: pass

    if source in ["news", "all"]:
        try:
            q = urllib.parse.quote(f"{ipo_name} IPO")
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en")
            texts.extend([f"[News]: {e.title}" for e in feed.entries[:5]])
        except: pass

    return "\n".join(texts) if texts else "No sentiment data found."

# --- WORKER 3: RHP DOCUMENT ---
def query_rhp(ipo_name, query, vector_store=None):
    if not vector_store:
        return "⚠️ RHP Document is not loaded."

    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="qwen/qwen3.8-27b", max_tokens=60)
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
    ipo_id = details.get('id')
    slug = details.get('slug')
    os.makedirs("pdfs", exist_ok=True)
    save_path = os.path.join("pdfs", f"{ipo_id}.pdf")
    
    if os.path.exists(save_path): return save_path

    page_url = f"https://www.ipopremium.in/view/ipo/{ipo_id}/{slug}"
    try:
        scraper = cloudscraper.create_scraper()
        r = scraper.get(page_url, timeout=15)
        soup = BeautifulSoup(r.content, "html.parser")
        
        target_url = None
        candidates = []
        for a in soup.find_all("a", href=True):
            text = a.get_text().lower()
            if "rhp" in text or "drhp" in text or "anchor" in text:
                candidates.append({"link": a["href"], "text": text})
        
        # Prioritize RHP > DRHP > Anchor
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
            target_url = f"https://assets.ipopremium.in/images/ipo/{ipo_id}_rhp.pdf" 

        if target_url:
            if not target_url.startswith("http"): target_url = "https://www.ipopremium.in" + target_url
            pdf_resp = scraper.get(target_url, stream=True, timeout=15)
            if pdf_resp.status_code == 200:
                with open(save_path, "wb") as f: f.write(pdf_resp.content)
                return save_path
    except: pass
    return None

def build_vs_logic(pdf_path):
    """
    Builds Vector Store using FAISS (RAM-Only).
    This fixes the 'Tenant' and 'SQLite' errors on Streamlit Cloud.
    """
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
    
    # --- FIXED: USE FAISS INSTEAD OF CHROMA ---
    # FAISS runs in memory and doesn't care about SQLite versions.
    return FAISS.from_documents(splits, emb)
