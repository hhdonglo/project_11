# =============================================================================
# src/app/chatbot.py
# =============================================================================
# Puls-Events RAG — Cultural Event Chatbot
# Streamlit interface running on the canonical LangChain + Mistral + FAISS
# chain defined in src/rag/langchain_chain.py. This file contains NO
# duplicate retrieve/generate logic — it imports ask() directly, so the
# live demo genuinely exercises the LangChain integration.
#
# Usage:
#   poetry run streamlit run src/app/chatbot.py
# =============================================================================

import logging
import sys
from pathlib import Path

import streamlit as st

# --- Make the project root importable so `from src.rag...` works
# regardless of the working directory Streamlit was launched from ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.langchain_chain import ask  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# --- Streamlit UI ---
st.set_page_config(
    page_title="Puls-Events — Assistant Culturel Paris",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Puls-Events — Assistant Culturel Paris")
st.caption("Découvrez les événements culturels à Paris grâce à l'IA (LangChain + Mistral + FAISS)")

# Sidebar — source transparency
with st.sidebar:
    st.header("📚 Sources récupérées")
    st.caption(
        "Les événements utilisés pour générer la réponse "
        "apparaîtront ici après chaque question."
    )
    source_placeholder = st.empty()

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input(
    "Que recherchez-vous ? Ex: concerts ce weekend, expositions d'art..."
):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display response via the canonical LangChain chain
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            result = ask(prompt)
        st.markdown(result["answer"])

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": result["answer"]
    })

    # Update sidebar with sources
    with source_placeholder.container():
        for i, source in enumerate(result["sources"], 1):
            with st.expander(f"{i}. {source['title']}"):
                st.write(f"📅 {source['date_range']}")
                st.write(f"📍 {source['venue']}, {source['city']}")
                st.write(f"🔗 [Voir l'événement]({source['url']})")
