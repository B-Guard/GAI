import streamlit as st
import os
import re
import asyncio
from dotenv import load_dotenv
from src.chat import RAGChat
from src.retriever import FaissVectorStore, Neo4jRetriever

load_dotenv()

# Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join("data", "vector_store_faiss.pkl"))
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "llama3.2:latest")

# Page config
st.set_page_config(
    page_title="SALVIA - RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for clickable source citations
st.markdown("""
<style>
.source-badge {
    display: inline-block;
    padding: 2px 8px;
    margin: 0 2px;
    background-color: #2e7d32;
    color: white;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    transition: background-color 0.2s;
}
.source-badge:hover {
    background-color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)


def format_answer_with_citations(answer: str, message_index: int = 0) -> str:
    """Make [sX] or [X] tags into clickable"""

    def replace_citation(match):
        source_num = match.group(1)
        return f'<a href="#msg{message_index}-source-{source_num}" class="source-badge">[{source_num}]</a>'

    # Match source notation patterns
    formatted = re.sub(r'\[s?(\d+)\]', replace_citation, answer)
    return formatted


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_instance" not in st.session_state:
    st.session_state.chat_instance = None

# Sidebar configuration
with st.sidebar:
    st.title("Configuration")

    retriever_type = st.selectbox(
        "Retriever Type",
        ["neo4j", "faiss"],
        index=0
    )

    llm_model = st.text_input(
        "LLM Model",
        value=CHAT_MODEL_NAME
    )

    top_k = st.slider(
        "Top K Results",
        min_value=1,
        max_value=10,
        value=5
    )

    # Show FAISS path info if FAISS is selected
    if retriever_type == "faiss":
        st.info(f"FAISS index path: `{FAISS_INDEX_PATH}`")
        if os.path.exists(FAISS_INDEX_PATH):
            file_size = os.path.getsize(FAISS_INDEX_PATH) / 1024  # KB
            st.success(f"FAISS index found ({file_size:.1f} KB)")
        else:
            st.warning("FAISS index not found")
            st.markdown("Run ingestion first:")
            st.code("""python main.py ingest \\
  --llm-model llama3.2:latest \\
  --retriever faiss \\
  --path "data/texts/*.txt\"""", language="bash")

    if st.button("Initialize Chat", type="primary"):
        with st.spinner("Initializing chat system..."):
            try:
                # Initialize retriever
                if retriever_type == "faiss":
                    # Check if file exists before initializing
                    if not os.path.exists(FAISS_INDEX_PATH):
                        st.error("FAISS index not found. Please run ingestion first.")
                    else:
                        retriever = FaissVectorStore(
                            model_name=EMBEDDING_MODEL_NAME,
                            vector_store_path=FAISS_INDEX_PATH
                        )
                        # Initialize chat
                        st.session_state.chat_instance = RAGChat(
                            vector_store=retriever,
                            model_name=llm_model,
                            top_k=top_k
                        )
                        st.success("Chat initialized with FAISS retriever!")
                else:
                    retriever = Neo4jRetriever(
                        llm_model_name=llm_model,
                        embedding_model_name=EMBEDDING_MODEL_NAME,
                        uri=NEO4J_URI,
                        user=NEO4J_USER,
                        password=NEO4J_PASSWORD
                    )
                    # Initialize chat
                    st.session_state.chat_instance = RAGChat(
                        vector_store=retriever,
                        model_name=llm_model,
                        top_k=top_k
                    )
                    st.success("Chat initialized with Neo4j retriever!")

            except Exception as e:
                st.error(f"Error initializing chat: {str(e)}")
                import traceback

                with st.expander("Show full error"):
                    st.code(traceback.format_exc())

    st.divider()

    st.markdown("### System Information")
    st.text(f"Retriever: {retriever_type.upper()}")
    st.text(f"Model: {llm_model}")
    st.text(f"Top K: {top_k}")

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("🤖 SALVIA - RAG Chatbot")
st.markdown("*Slow And Low-Value Intelligent Agent*")

# Check if chat is initialized
if st.session_state.chat_instance is None:
    st.info("← Please initialize the chat system using the sidebar.")
else:
    # Display chat messages
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Citations as clickable badges for assistant messages
            if message["role"] == "assistant":
                formatted_content = format_answer_with_citations(message["content"], msg_idx)
                st.markdown(formatted_content, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        # Add unique anchor for this message's sources
                        st.markdown(f'<div id="msg{msg_idx}-source-{i}"></div>', unsafe_allow_html=True)
                        st.markdown(f"**Source {i}:**")
                        st.text(source[5:505].strip() + "..." if len(source) > 500 else source[5:].strip())
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response (streaming)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_chunks = []
                    sources = []
                    response_placeholder = st.empty()
                    # Stream the answer
                    for chunk, srcs in st.session_state.chat_instance.stream_answer(prompt):
                        if chunk is not None:
                            response_chunks.append(chunk)
                            # Get message index for unique anchors (use current length + 1 for in-progress message)
                            msg_idx = len(st.session_state.messages)
                            formatted_answer = format_answer_with_citations(''.join(response_chunks), msg_idx)
                            response_placeholder.markdown(formatted_answer, unsafe_allow_html=True)
                        else:
                            if srcs:
                                sources = srcs
                            break
                    # Show sources in expander
                    if sources:
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources, 1):
                                # Add unique anchor for this message's sources
                                st.markdown(f'<div id="msg{msg_idx}-source-{i}"></div>', unsafe_allow_html=True)
                                st.markdown(f"**Source {i}:**")
                                st.text(source[:500] + "..." if len(source) > 500 else source)
                                st.divider()
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ''.join(response_chunks),
                        "sources": sources
                    })
                    if not response_chunks:
                        st.warning("Sorry, I couldn't find relevant information to answer that question.")
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    import traceback
                    with st.expander("Show full error"):
                        st.code(traceback.format_exc())

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    <p>Powered by Neo4j GraphRAG & Ollama | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)