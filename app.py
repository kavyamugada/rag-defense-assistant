import streamlit as st
import base64
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import ollama

# ---------------- CONFIG ----------------
DB_PATH = "vector_db"

st.set_page_config(
    page_title="Defense SOP Assistant",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- LOAD BACKGROUND IMAGE ----------------
def get_base64_bg(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64_bg("bg.png")

# ---------------- CUSTOM CSS ----------------
st.markdown(
    f"""
    <style>
    /* -------- MAIN APP BACKGROUND -------- */
    .stApp {{
        background:
            linear-gradient(
                rgba(0, 0, 0, 0.55),
                rgba(0, 0, 0, 0.55)
            ),
            url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
        animation: bgPulse 20s ease infinite;
    }}

    @keyframes bgPulse {{
        0% {{ filter: brightness(1); }}
        50% {{ filter: brightness(1.12); }}
        100% {{ filter: brightness(1); }}
    }}

    /* -------- SIDEBAR -------- */
    section[data-testid="stSidebar"] {{
        background-color: #000000;
    }}

    section[data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #00c6ff !important;
    }}

    /* -------- GLASS CARDS -------- */
    .glass {{
        background: rgba(255,255,255,0.95);
        color: #000000;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.35);
        animation: fadeIn 0.6s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* -------- INPUT -------- */
    input {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px !important;
        border: 1px solid #ccc !important;
    }}

    /* -------- BUTTON -------- */
    div.stButton > button {{
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1.2em;
        border: none;
        transition: transform 0.2s ease;
    }}

    div.stButton > button:hover {{
        transform: scale(1.05);
        background: linear-gradient(90deg, #0072ff, #00c6ff);
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div class="glass">
        <h1 style="text-align:center;">🛡️ Defense Policy & SOP Assistant</h1>
        <p style="text-align:center; color:#555;">
        Secure • SOP-grounded • No hallucinations
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 System Overview")
    st.write("**Domain:** Defense & Security")
    st.write("**LLM:** Mistral (Ollama)")
    st.write("**Embeddings:** nomic-embed-text")
    st.write("**Vector DB:** Chroma")

    st.markdown("---")
    st.markdown("## 📂 Knowledge Base")
    st.write("• Incident Command Manuals")
    st.write("• Civil Defense SOPs")
    st.write("• Emergency Communication Protocols")
    st.write("• Infrastructure Security Docs")

# ---------------- LOAD VECTOR DB ----------------
@st.cache_resource
def load_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

db = load_db()

# ---------------- RAG FUNCTION ----------------
def ask_with_rag(question):
    docs = db.similarity_search(question, k=2)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a defense SOP assistant.
Answer ONLY using the context below.
If the answer is not found, say exactly:
"Not specified in SOP".

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="mistral:latest",
        messages=[{"role": "user", "content": prompt}]  # ✅ FIXED LINE
    )

    return response["message"]["content"]

# ---------------- MAIN INTERACTION ----------------
st.markdown("### ❓ Ask a Security / SOP Question")

question = st.text_input(
    "",
    placeholder="Example: What is the immediate action for unauthorized access to a secure facility?"
)

ask = st.button("🔍 Get SOP Answer")

# ---------------- OUTPUT ----------------
if ask:
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("🔐 Analyzing SOP documents..."):
            answer = ask_with_rag(question)

        st.markdown(
            f"""
            <div class="glass">
                <h3>✅ SOP-Based Answer</h3>
                <p>{answer}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption("📌 Response is generated strictly from official SOP documents.")
