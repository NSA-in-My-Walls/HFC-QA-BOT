A Streamlit-based Retrieval-Augmented Generation (RAG) application for querying Houston Faith Church sermons. Users ask natural-language questions and the app retrieves relevant sermon excerpts (via BM25 + FAISS + MMR), then uses Google Gemini to generate scripture-based answers.

Features

Hybrid Lexical + Semantic RetrievalUses BM25 to surface exact-term matches and FAISS embeddings to capture semantic relevance, followed by MMR for diversity.

Primary & Additional SourcesHighlights the most relevant sermon chunk with a "Primary Source" label, and lists additional excerpts ranked by similarity.

Relevance BarVisual progress bars showing cosine-similarity scores (0.0 – 1.0) for each referenced chunk.

Scheduled Index UpdatesGitHub Actions workflow to download new transcripts and rebuild indexes weekly.

Repository Structure
├── .github/workflows/weekly-index.yml   # GitHub Actions for transcript & index refresh
├── sermon_transcripts/                  # Transcript JSONs (downloaded)
├── sermon_faiss_free/                   # FAISS index & metadata
├── embed_sermons_free.py                # Script to build FAISS & BM25 indexes
├── download_transcripts.py              # Script to fetch new sermon transcripts
├── streamlit_app.py                     # Main Streamlit application
├── videos.json                          # Video ID → title mapping
├── info.env                             # Environment variables (not committed)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file

Setup & Installation

Clone Repo
git clone https://github.com/<your-org>/hfc-sermon-qabot.git
cd hfc-sermon-qabot

Create virtual environment
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\\Scripts\\activate   # Windows

Install dependencies
pip install -r requirements.txt

Set up environment variables
Copy info.env.example to info.env and add your Google Gemini API key: API_KEY=your_gemini_api_key

Usage
Run the Streamlit app
streamlit run streamlit_app.py: streamlit run streamlit_app.py
In the sidebar, choose the number of context chunks (K).
Ask a question about the sermons and click Get Answer.
View the generated answer and a list of Primary & Additional sermon links with relevance bars.

Weekly Transcript & Index Refresh
A GitHub Actions workflow (.github/workflows/weekly-index.yml) runs every Monday at 02:00 UTC to:
Download the latest transcripts via download_transcripts.py.
Commit any changes to sermon_transcripts/.
Rebuild FAISS & BM25 indexes via embed_sermons_free.py.
Commit updated indexes to sermon_faiss_free/.
You don’t need to manually trigger this; GitHub Actions handles it automatically.

Development
Testing retrieval: add or modify transcript JSONs in sermon_transcripts/, then run embed_sermons_free.py to regenerate the index.
Adjust retrieval params: tweak BM25 top N, FAISS search top_n, and MMR lambda_param in streamlit_app.py.
UI tweaks: modify sidebar settings or relevance bar styles directly in the Streamlit code.
