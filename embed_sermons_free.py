# embed_sermons_with_timestamps.py

import os, json, pickle, faiss
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500  # words

documents = []    # texts for embedding
meta = []         # tuples of (video_id, chunk_i, timestamp)

for fname in os.listdir("sermon_transcripts"):
    if not fname.endswith(".json"): continue
    video_id = fname.replace(".json","")
    data = json.load(open(f"sermon_transcripts/{fname}", encoding="utf-8"))
    
    # build a list of (word, start_time) for every word
    words_with_time = []
    for seg in data:
        for w in seg["text"].split():
            words_with_time.append((w, seg["start"]))
    
    # chunk the word list
    for i in range(0, len(words_with_time), CHUNK_SIZE):
        chunk = words_with_time[i:i+CHUNK_SIZE]
        text = " ".join(w for w, _ in chunk)
        start_time = chunk[0][1]        # start time of first word in chunk
        documents.append(text)
        meta.append((video_id, i//CHUNK_SIZE, start_time))

# embed with Sentence-Transformers
model = SentenceTransformer("all-MiniLM-L6-v2")
embs = model.encode(documents, convert_to_numpy=True)
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

# save index + meta
faiss.write_index(index, "sermon_index_timed.faiss")
with open("sermon_meta_timed.pkl","wb") as f:
    pickle.dump(meta, f)
