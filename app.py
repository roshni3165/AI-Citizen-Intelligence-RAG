import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found.")
    st.stop()

client = genai.Client(api_key=api_key)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.set_page_config(
    page_title="AI Citizen Assistant",
    page_icon="🤖"
)

st.title("🤖 AI Citizen Assistant")

# -------------------------
# CHAT HISTORY
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# -------------------------
# DOCUMENT UPLOAD
# -------------------------

uploaded_files = st.file_uploader(
    "📄 Upload your documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []
    all_metadata = []

    for uploaded_file in uploaded_files:

        reader = PdfReader(uploaded_file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text() or ""

            if not page_text.strip():
                continue

            chunks = splitter.split_text(page_text)

            for chunk in chunks:

                all_chunks.append(chunk)

                all_metadata.append({
                    "source": uploaded_file.name,
                    "page": page_number
                })

    st.success(
        f"{len(uploaded_files)} documents processed! ✅"
    )

    # -------------------------
    # EMBEDDINGS
    # -------------------------

    embeddings = model.encode(all_chunks)

    # -------------------------
    # CHROMADB
    # -------------------------

    db = chromadb.Client()

    try:
        db.delete_collection("citizen_documents")
    except:
        pass

    collection = db.create_collection(
        "citizen_documents"
    )

    collection.add(
        ids=[
            str(i)
            for i in range(len(all_chunks))
        ],
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadata
    )

    st.success(
        "Documents stored successfully! ✅"
    )

    # -------------------------
    # CHAT INPUT
    # -------------------------

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:

        # Save user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
    # =================================================
# PERSONAL DOCUMENT SUMMARY
# =================================================

    st.subheader("📊 Personal Document Intelligence")

if st.button("🧠 Analyze My Documents"):

    # Use ALL uploaded document chunks
    full_document = ""

    for chunk, meta in zip(
        all_chunks,
        all_metadata
    ):
        full_document += f"""
SOURCE: {meta['source']}
PAGE: {meta['page']}

CONTENT:
{chunk}

"""

    summary_prompt = f"""
You are a Personal Document Intelligence AI.

Analyze ALL the uploaded documents below.

Extract ONLY information that is actually present.

Create a clear summary with these sections:

1. 👤 Personal Information
2. 🎓 Education
3. 💻 Skills
4. 📜 Certifications
5. 💼 Projects
6. 🏆 Achievements
7. 📅 Important Dates

If information is not available, write:
"Not found in uploaded documents."

Do NOT invent information.

UPLOADED DOCUMENTS:

{full_document}
"""

    with st.spinner("Analyzing your documents..."):

        summary_response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=summary_prompt
        )

    st.subheader("📊 Your Document Summary")

    st.write(summary_response.text)

        with st.chat_message("user"):
            st.write(question)

        # -------------------------
        # RETRIEVE
        # -------------------------

        question_embedding = model.encode(
            [question]
        )[0]

        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=min(20, len(all_chunks))
        )

        chunks = results["documents"][0]
        metadata = results["metadatas"][0]

        context = ""

        for chunk, meta in zip(
            chunks,
            metadata
        ):

            context += f"""
SOURCE: {meta['source']}
PAGE: {meta['page']}

CONTENT:
{chunk}

"""


        # -------------------------
        # CONVERSATION HISTORY
        # -------------------------

        history = ""

        for message in st.session_state.messages[-6:]:

            history += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )


        # -------------------------
        # GEMINI
        # -------------------------

        prompt = f"""
You are an AI-powered Personal Document Assistant.

Answer using ONLY the document context.

Do not invent information.

If the answer is not present, say:

"I could not find this information in the uploaded documents."

If the user is asking which uploaded documents
are relevant or missing for a scholarship or scheme,
compare the requirements in the scheme document
with the uploaded personal documents.

Clearly separate:

1. Documents that appear to be available
2. Documents that appear to be missing
3. The filename that supports each available document

Do not claim a document is available unless it is
actually present in the uploaded documents.

Use the conversation history to understand
follow-up questions.

CONVERSATION HISTORY:
{history}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{question}
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        answer = response.text

        # -------------------------
        # SHOW ANSWER
        # -------------------------

        with st.chat_message("assistant"):
            st.write(answer)

            st.caption("📌 Sources:")

            shown = set()

            for meta in metadata:

                source_key = (
                    meta["source"],
                    meta["page"]
                )

                if source_key not in shown:

                    shown.add(source_key)

                    st.write(
                        f"📄 {meta['source']} "
                        f"— Page {meta['page']}"
                    )

        # Save AI response
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })