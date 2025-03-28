import streamlit as st
import openai
import faiss
import json
import numpy as np

# === OpenAI API Key from Streamlit Secrets ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === Load FAISS index and metadata ===
@st.cache_resource
def load_faiss_and_metadata():
    index = faiss.read_index("umc_index.faiss")
    with open("umc_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_faiss_and_metadata()

# === Helper Functions ===
def embed_query(query, model="text-embedding-3-large"):
    response = openai.Embedding.create(input=query, model=model)
    return np.array(response["data"][0]["embedding"]).astype("float32")

def format_reference(entry):
    fields = [
        ("Part", entry.get("part")),
        ("Heading", entry.get("heading")),
        ("Title", entry.get("title")),
        ("Sub-title", entry.get("sub_title")),
        ("¶", entry.get("paragraph_number")),
        ("Paragraph", entry.get("paragraph_title")),
        ("Sub-para", entry.get("sub_para_title")),
    ]
    return " — ".join([f"{label}: {value}" for label, value in fields if value])

def build_prompt(user_query, passages):
    prompt = f"""You are an assistant answering questions using the Book of Discipline of the United Methodist Church.

Use exact wording when needed and include references by paragraph, title, or section.

### Question:
{user_query}

### Relevant Extracts:"""
    for p in passages:
        ref = format_reference(p)
        text = p["text"].strip().replace("\n", " ")
        prompt += f"\n\n→ {ref}:\n\"{text}\""
    prompt += "\n\n### Your Answer:"
    return prompt

def summarize_each_chunk(passages):
    summaries = []
    for p in passages:
        ref = format_reference(p)
        text = p['text'].strip()
        prompt = f"""Summarize the following section from the Book of Discipline of the United Methodist Church. Keep key legal/theological language.

### Reference: {ref}
\"\"\"{text}\"\"\"

### Summary:"""
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000
        )
        summaries.append((ref, response["choices"][0]["message"]["content"]))
    return summaries

# === RAG Query Pipeline ===
def rag_query(user_query, k=5):
    query_vector = embed_query(user_query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    top_chunks = [metadata[i] for i in indices[0] if i < len(metadata)]

    prompt = build_prompt(user_query, top_chunks)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer using the United Methodist Church Book of Discipline."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=2000
    )
    answer = response["choices"][0]["message"]["content"]
    summaries = summarize_each_chunk(top_chunks)
    return answer, summaries, top_chunks

# === Streamlit UI ===
st.set_page_config(page_title="United Methodist Church AI Bot", layout="wide")
st.image("UMC_LOGO.png", width=80)
st.title("📘 United Methodist Church - Book of Discipline Assistant")
st.markdown("Ask a question and get grounded answers with references from the official Book of Discipline.")

# === Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

# === Reset Chat ===
if st.button("🧹 Start New Chat"):
    st.session_state.clear()
    st.rerun()

# === Show Chat History ===
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"""<div style="background-color: #f0f0f0; padding: 12px; border-radius: 8px;">
    {chat['answer']}
    </div>""", unsafe_allow_html=True)

# === Show Retrieved Sources ===
if st.session_state.last_chunks:
    with st.expander("📚 Referenced Chunks", expanded=False):
        for i, ref in enumerate(st.session_state.last_chunks):
            st.markdown(f"**Reference {i+1}:**")
            for key in ["part", "heading", "title", "sub_title", "paragraph_number", "paragraph_title", "sub_para_title"]:
                if ref.get(key):
                    st.markdown(f"- **{key.replace('_', ' ').title()}:** {ref[key]}")
            st.markdown(f"**Text:** {ref['text']}")
            st.markdown("---")

# === User Input ===
query = st.text_input("🔎 Ask your question:", "")

if st.button("Send") and query.strip():
    with st.spinner("Generating answer..."):
        answer, summaries, refs = rag_query(query, k=5)
        st.session_state.chat_history.append({"question": query, "answer": answer})
        st.session_state.last_chunks = refs
        st.rerun()

# === Summarized References ===
if st.session_state.last_chunks:
    st.markdown("### 🧠 Summarized References")
    for i, (ref, summary) in enumerate(summarize_each_chunk(st.session_state.last_chunks), 1):
        with st.expander(f"{i}. {ref}", expanded=False):
            st.markdown(summary)
