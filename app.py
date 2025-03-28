import streamlit as st
import openai
import faiss
import numpy as np
import json

# --- Load Embedding Model + Index ---
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index("umc_index.faiss")
    with open("umc_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

index, metadata = load_index_and_metadata()

# --- OpenAI Key (Streamlit Secrets or Manual) ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Helper Functions ---
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
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1250
        )
        summaries.append((ref, response["choices"][0]["message"]["content"]))
    return summaries

# --- RAG Pipeline ---
def rag_query(user_query, k=5):
    query_vector = embed_query(user_query).reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    top_chunks = [metadata[i] for i in indices[0] if i < len(metadata)]

    prompt = build_prompt(user_query, top_chunks)
    answer = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You answer using the United Methodist Church Book of Discipline."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=750
    )["choices"][0]["message"]["content"]

    summaries = summarize_each_chunk(top_chunks)
    return answer, summaries

# --- Streamlit UI ---
st.set_page_config(page_title="UMC Book of Discipline RAG", layout="wide")
st.title("📘 United Methodist Church - Book of Discipline RAG Assistant")

query = st.text_input("Ask a question:", placeholder="e.g. What does the UMC say about gender justice?")
top_k = st.slider("Number of matching sections", min_value=1, max_value=10, value=5)

if st.button("🔍 Search"):
    with st.spinner("Thinking..."):
        response, citations = rag_query(query, top_k)

    st.markdown("### ✝️ GPT-4o Answer")
    st.write(response)

    st.markdown("### 📚 Referenced Summaries")
    for idx, (ref, summary) in enumerate(citations, 1):
        with st.expander(f"{idx}. {ref}"):
            st.write(summary)