# streamlit_app.py
import logging

# Silence Streamlit watcher warnings
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
import os
import faiss
import pickle
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json

st.set_page_config(
    page_title="Houston Faith Church Q&A",
    layout="wide",
    initial_sidebar_state="expanded"   # ← forces the sidebar open
)

# ─── Configuration ─────────────────────────────────────────────────────────
TRANSCRIPT_DIR = "sermon_transcripts"
# If you used 500‐word chunks when you indexed:
CHUNK_SIZE     = 500
# (If you used a different size in your embed script, match that here)
INDEX_DIR      = "sermon_faiss_free"
EMBED_MODEL    = "all-MiniLM-L6-v2"

# ─── Load videos to build map ─────────────────────────────────────────────────────────
with open("videos.json", "r", encoding="utf-8") as f:
    vids_data = json.load(f).get("entries", [])
video_title_map = {entry["id"]: entry.get("title", entry["id"]) for entry in vids_data}

# ─── Load environment ─────────────────────────────────────────────────────────
load_dotenv("info.env")
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# ─── Cached resource loading ──────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    index = faiss.read_index("sermon_faiss_free/sermon_index.faiss")
    with open("sermon_faiss_free/sermon_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return index, meta, embedder

index, meta, embedder = load_resources()

# ─── Sidebar settings ─────────────────────────────────────────────────────────
st.sidebar.title("Settings")

st.sidebar.markdown(
    """
    **Context window (K)**  
    This controls how many 500-word sermon chunks are pulled in to answer each question.  
    - **Lower K** (e.g. 2–4): Faster, more focused answers  
    - **Higher K** (e.g. 10–15): Broader context, but may be slower or include off-topic bits  
    """
)

K = st.sidebar.slider(
    "Number of chunks (K)",
    min_value=1,
    max_value=20,
    value=6,
    step=1
)


# ─── App UI ───────────────────────────────────────────────────────────────────
st.title("Houston Faith Church Sermon Q&A")
question = st.text_input("THIS IS STILL IN DEVELOPMENT, BEING UPDATED CONSTANTLY. Ask a question about our sermons:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            # 1) Embed & retrieve
            q_vec = embedder.encode([question])
            _, I = index.search(q_vec, K)

            # 2) Gather contexts & links
            contexts = []
            links    = []
            for idx in I[0]:
                video_id, chunk_i, start_time = meta[idx]
                # load chunk text…
                words = open(os.path.join(TRANSCRIPT_DIR, f"{video_id}.json"),
                             encoding="utf-8").read().split()
                start, end = chunk_i * CHUNK_SIZE, (chunk_i + 1) * CHUNK_SIZE
                contexts.append(" ".join(words[start:end]))

                # build timestamped link…
                url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                title = video_title_map.get(video_id, video_id)
                links.append((url, title, start_time))

            # 3) Build the “Clarifier” prompt (#4), assembling contexts properly
            context_block = "\n\n".join(contexts)
            prompt = (
                "Answer this question based on the Houston Faith Church sermons below.\n\n"
                "Whenever an excerpt directly addresses the question, quote or paraphrase it accurately.\n"
                "If no excerpt speaks to the question, use only logical inference grounded in the beliefs and principles shown in these sermons.\n"
                "Never invent new doctrine, speculate, or refer to excerpts as “documents” or “sources.”\n\n"
                "--- SERMONS ---\n"
                f"{context_block}\n"
                "--- END SERMONS ---\n\n"
                f"Question: {question}\n"
                "Answer in a clear, direct, teaching style (no casual tone):"
            )

            # 4) Call Gemini & display
            response = model.generate_content(prompt)
            st.markdown(f"**Answer:**\n\n{response.text.strip()}")

            # 5) Show referenced sermons with timestamps
            def fmt(ts_float):
                ts = int(ts_float)
                h = ts // 3600
                m = (ts % 3600) // 60
                s = ts % 60
                return f"{h}:{m:02d}:{s:02d}"

            st.markdown("**Referenced Sermons:**")
            for url, title, ts in sorted(links, key=lambda x: x[1]):
                st.markdown(f"- [{title} @ {fmt(ts)}]({url})")

