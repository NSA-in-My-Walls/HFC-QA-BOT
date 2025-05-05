# streamlit_app.py
import logging

# Silence Streamlit watcher warnings
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

import os
import json
import pickle
import faiss
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.generativeai import types
from sklearn.metrics.pairwise import cosine_similarity

# ─── Set page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Houston Faith Church Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Configuration ─────────────────────────────────────────────────────────
TRANSCRIPT_DIR = "sermon_transcripts"
CHUNK_SIZE     = 500
INDEX_DIR      = "sermon_faiss_free"
EMBED_MODEL    = "all-MiniLM-L6-v2"

# ─── Load video titles ────────────────────────────────────────────────────────
with open("videos.json", "r", encoding="utf-8") as f:
    vids_data = json.load(f).get("entries", [])
video_title_map = {entry["id"]: entry.get("title", entry["id"]) for entry in vids_data}

# ─── Load environment & configure Gemini ───────────────────────────────────────
load_dotenv("info.env")
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# ─── Cache resource loading ──────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    index = faiss.read_index(os.path.join(INDEX_DIR, "sermon_index.faiss"))
    with open(os.path.join(INDEX_DIR, "sermon_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, meta, embedder

index, meta, embedder = load_resources()

# ─── Sidebar settings ─────────────────────────────────────────────────────────
st.sidebar.title("Settings")
st.sidebar.markdown(
    """
    **Context window (K)**  
    Number of 500 word sermon chunks fetched for analyzing.  
    - **Lower K**: faster, more focused.  
    - **Higher K**: more context, may include off-topic details.  
    """
)
K = st.sidebar.slider("Number of chunks (K)", 1, 20, 8, step=1)

# ─── MMR re-ranking function ─────────────────────────────────────────────────
def mmr(doc_embs: np.ndarray, query_emb: np.ndarray, top_n: int, k: int, lambda_param: float = 0.7):
    # similarity to query
    sim_q = cosine_similarity(doc_embs, query_emb.reshape(1, -1)).reshape(-1)
    # pairwise similarity among docs
    sim_docs = cosine_similarity(doc_embs)
    selected = [int(np.argmax(sim_q))]
    candidates = set(range(top_n)) - set(selected)
    for _ in range(1, k):
        mmr_scores = {}
        for idx in candidates:
            rel = sim_q[idx]
            red = max(sim_docs[idx][j] for j in selected)
            mmr_scores[idx] = lambda_param * rel - (1 - lambda_param) * red
        next_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(next_idx)
        candidates.remove(next_idx)
    return selected

# ─── App UI ───────────────────────────────────────────────────────────────────
st.title("Houston Faith Church Sermon Q&A")
question = st.text_input("Ask a question about our sermons:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            # 1) Embed question
            q_vec = embedder.encode([question])[0]

            # 2) Retrieve top_n candidates from FAISS
            top_n = min(20, len(meta))
            D, I = index.search(np.array([q_vec]), top_n)
            hit_ids = I[0].tolist()

            # 3) Gather initial contexts & links
            contexts_n, links_n = [], []
            for vid_idx in hit_ids:
                video_id, chunk_i, start_time = meta[vid_idx]

                # Load the transcript JSON
                with open(os.path.join(TRANSCRIPT_DIR, f"{video_id}.json"), encoding="utf-8") as jf:
                    segs = json.load(jf)

                # Flatten into words
                words = []
                for seg in segs:
                    words.extend(seg["text"].split())

                # Slice the chunk
                start, end = chunk_i * CHUNK_SIZE, (chunk_i + 1) * CHUNK_SIZE
                contexts_n.append(" ".join(words[start:end]))

                # Prepare the timestamped link
                url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}s"
                title = video_title_map.get(video_id, video_id)
                links_n.append((url, title, start_time))

            # 4) Compute embeddings for those top_n contexts
            doc_embs = embedder.encode(contexts_n, convert_to_numpy=True)

            # 5) Rerank with MMR to pick final K chunks
            sel = mmr(doc_embs, q_vec, top_n=top_n, k=K, lambda_param=0.7)
            contexts = [contexts_n[i] for i in sel]
            links    = [links_n[i]    for i in sel]

            # 6) Build the prompt
            context_block = "\n\n".join(contexts)
            prompt = (
                "You are a Spirit-filled assistant trained on these Houston Faith Church sermon excerpts.\n\n"
                "Always answer with spiritual wisdom. Use scripture-based reasoning to support your training when reasonable.\n"
                "--- SERMONS ---\n"
                f"{context_block}\n"
                "--- END SERMONS ---\n\n"
                f"Question: {question}\n"
                "Answer with using only the excerpts above. Do not mention the sermons, documents, excerpts, or any source material. Speak as a representative of the church:"
            )

            # 7) Call Gemini & display
            response = model.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    temperature=0.3,       # lower randomness for doctrinal precision
                    top_p=0.85,            # nucleus sampling cutoff
                    max_output_tokens=2048 # allow longer answers if needed
                )
            )
            answer = response.text.strip()
            st.markdown(f"**Answer:**\n\n{answer}")

            # 8) Show referenced sermons
            def fmt(ts):
                t = int(ts)
                h, rem = divmod(t, 3600)
                m, s = divmod(rem, 60)
                return f"{h}:{m:02d}:{s:02d}"

            st.markdown("**Referenced Sermons:**")
            for url, title, ts in sorted(links, key=lambda x: x[1]):
                st.markdown(f"- [{title} @ {fmt(ts)}]({url})")
