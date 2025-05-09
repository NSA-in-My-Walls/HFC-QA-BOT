# streamlit_app.py

import logging
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

# Import BM25 for lexical retrieval
from rank_bm25 import BM25Okapi

# ─── Silence noisy Streamlit logs ───────────────────────────────────────────────
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# ─── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Houston Faith Church Q&A",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Constants & paths ─────────────────────────────────────────────────────────
TRANSCRIPT_DIR = "sermon_transcripts"
CHUNK_SIZE     = 500       # words per chunk
INDEX_DIR      = "sermon_faiss_free"
EMBED_MODEL    = "all-MiniLM-L6-v2"

# ─── Load sermon titles from videos.json ────────────────────────────────────────
with open("videos.json", "r", encoding="utf-8") as f:
    vids_data = json.load(f).get("entries", [])
# Map video_id → human-readable title
video_title_map = {
    entry["id"]: entry.get("title", entry["id"])
    for entry in vids_data
}

# ─── Load environment & configure Gemini API ────────────────────────────────────
load_dotenv("info.env")
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

# ─── Cache & load resources (FAISS index, metadata, embedder, BM25) ────────────
@st.cache_resource
def load_resources():
    # 1) Load FAISS vector index & metadata
    index = faiss.read_index(os.path.join(INDEX_DIR, "sermon_index.faiss"))
    with open(os.path.join(INDEX_DIR, "sermon_meta.pkl"), "rb") as f:
        meta = pickle.load(f)   # List of tuples: (video_id, chunk_idx, start_time)

    # 2) Initialize embedding model
    embedder = SentenceTransformer(EMBED_MODEL)

    # 3) Reconstruct chunk texts for BM25
    all_texts = []
    for video_id, chunk_i, _ in meta:
        # Read full transcript JSON for this video
        path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")
        segs = json.load(open(path, encoding="utf-8"))

        # Flatten into a single word list
        words = [w for seg in segs for w in seg["text"].split()]
        start, end = chunk_i * CHUNK_SIZE, (chunk_i + 1) * CHUNK_SIZE
        chunk_text = " ".join(words[start:end])

        all_texts.append(chunk_text)

    # 4) Tokenize and build BM25 index
    tokenized_texts = [txt.lower().split() for txt in all_texts]
    bm25 = BM25Okapi(tokenized_texts)

    return index, meta, embedder, all_texts, bm25

# Unpack cached resources
index, meta, embedder, all_texts, bm25 = load_resources()

# ─── Sidebar: context window slider ─────────────────────────────────────────────
st.sidebar.title("Settings")
st.sidebar.markdown(
    """
    **Context window (K)**  
    Number of 500-word chunks fetched for each question.  
    - Lower K → faster, more focused.  
    - Higher K → more context, may include off-topic details.
    - Default value of (9)
    """
)
K = st.sidebar.slider("Number of chunks (K)", 1, 20, 9, step=1)

# ─── MMR re-ranking ─────────────────────────────────────────────────────────────
def mmr(doc_embs: np.ndarray, query_emb: np.ndarray, top_n: int, k: int,
        lambda_param: float = 0.7) -> list[int]:
    """
    Select k diverse + relevant docs from the top_n candidates.
    Uses Maximal Marginal Relevance.
    """
    # 1) Relevance: similarity of each doc to the query
    sim_q = cosine_similarity(doc_embs, query_emb.reshape(1, -1)).reshape(-1)
    # 2) Redundancy: pairwise similarities among docs
    sim_docs = cosine_similarity(doc_embs)

    # 3) Initialize with the highest-relevance doc
    selected = [int(np.argmax(sim_q))]
    candidates = set(range(top_n)) - set(selected)

    # 4) Iteratively pick docs maximizing (λ·relevance - (1-λ)·redundancy)
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

# ─── App UI: title & input ─────────────────────────────────────────────────────
st.title("Houston Faith Church Sermon Q&A (In Development)")
question = st.text_input("Search a topic or ask a question about our sermons:")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            # ─── 1) Embed the question ──────────────────────────────────
            q_vec    = embedder.encode([question])[0]
            q_tokens = question.lower().split()  # for BM25

            # ─── 2) BM25 lexical retrieval ──────────────────────────────
            bm25_top_n = 50
            # Compute BM25 scores & pick top N chunk indices
            scores     = bm25.get_scores(q_tokens)
            lex_ids    = np.argsort(scores)[::-1][:bm25_top_n]

            # ─── 3) Semantic embed + initial ranking ────────────────────
            # Embed just the BM25-selected texts
            lex_texts = [all_texts[i] for i in lex_ids]
            lex_embs  = embedder.encode(lex_texts, convert_to_numpy=True)
            # Cosine similarity vs. query
            sem_sims  = cosine_similarity(lex_embs, q_vec.reshape(1, -1)).reshape(-1)

            # Pick top M candidates by semantic score (e.g. M = min(20, bm25_top_n))
            M         = min(len(lex_ids), 20)
            sem_order = np.argsort(sem_sims)[::-1][:M]
            sem_ids   = [lex_ids[i] for i in sem_order]
            sem_embs  = lex_embs[sem_order]

            # ─── 4) MMR rerank for diversity ─────────────────────────────
            sel_relative = mmr(sem_embs, q_vec, top_n=M, k=K, lambda_param=0.7)
            sel_global   = [sem_ids[i] for i in sel_relative]

            # ─── 5) Build final contexts & links ────────────────────────
            contexts = []
            links    = []
            for idx in sel_global:
                # Use the precomputed chunk text
                chunk_text = all_texts[idx]
                contexts.append(chunk_text)

                # Reconstruct YouTube link & timestamp
                vid_id, chunk_i, ts = meta[idx]
                url   = f"https://www.youtube.com/watch?v={vid_id}&t={int(ts)}s"
                title = video_title_map.get(vid_id, vid_id)
                links.append((url, title, ts))

            # ─── 6) Build the prompt for Gemini ────────────────────────
            context_block = "\n\n".join(contexts)
            prompt = (
                "You are a Spirit-filled assistant trained on these Houston Faith Church sermon excerpts.\n\n"
                "Always answer with spiritual wisdom. Use scripture-based reasoning to support your training when reasonable.\n"
                "--- SERMONS ---\n"
                f"{context_block}\n"
                "--- END SERMONS ---\n\n"
                f"Question: {question}\n"
                "Answer with using only the excerpts above. Do not mention the sermons, documents, excerpts, or any source material. Speak as a representative of the church."
            )

            # ─── 7) Call Gemini & render the answer ────────────────────
            response = model.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    temperature=0.5,
                    top_p=0.85,
                    max_output_tokens=2048
                )
            )
            answer = response.text.strip()
            st.markdown(f"**Answer:**\n\n{answer}")

            # Build score_map with Python floats up front:
            pre_mmr_scores = sem_sims[sem_order]                 # scores for the M candidates
            sel_scores     = [pre_mmr_scores[i] for i in sel_relative]
            score_map = {
                idx: float(score) for idx, score in zip(sel_global, sel_scores)
            }

            # ─── 8) Primary vs Additional & relevance bars ──────────────
            def fmt(ts: float) -> str:
                t = int(ts)
                h, rem = divmod(t, 3600)
                m, s   = divmod(rem, 60)
                return f"{h}:{m:02d}:{s:02d}"

            primary_idx    = max(sel_global, key=lambda i: score_map[i])
            secondary_idxs = [i for i in sel_global if i != primary_idx]

            # Primary Source
            st.subheader("Primary Source")
            vid, _, ts = meta[primary_idx]
            title      = video_title_map.get(vid, vid)
            url        = f"https://www.youtube.com/watch?v={vid}&t={int(ts)}s"
            sim        = score_map[primary_idx]
            st.markdown(f"- **[{title} @ {fmt(ts)}]({url})**  ({sim:.2f})")
            st.progress(min(sim, 1.0))

            # Additional Sources
            if secondary_idxs:
                st.subheader("Additional Sources - Ranked by similiarity score")
                # Sort additional sources by descending similarity
                secondary_sorted = sorted(
                    secondary_idxs,
                    key=lambda i: score_map[i],
                    reverse=True
                )
                for idx in secondary_sorted:
                    vid, _, ts = meta[idx]
                    title      = video_title_map.get(vid, vid)
                    url        = f"https://www.youtube.com/watch?v={vid}&t={int(ts)}s"
                    sim        = score_map[idx]
                    st.markdown(f"- [{title} @ {fmt(ts)}]({url})  ({sim:.2f})")
                    st.progress(min(sim, 1.0))
