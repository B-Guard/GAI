import streamlit as st
import os
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
            st.warning(f"FAISS index not found")
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
                        st.error(f"FAISS index not found. Please run ingestion first.")
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
                        st.success(f"Chat initialized with FAISS retriever!")
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
                    st.success(f"Chat initialized with Neo4j retriever!")

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
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:500] + "..." if len(source) > 500 else source)
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from chat instance
                    answer, sources = st.session_state.chat_instance._process_question(
                        prompt,
                        stream=False
                    )

                    if answer:
                        st.markdown(answer)

                        # Show sources in expander
                        if sources:
                            with st.expander("View Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(source[:500] + "..." if len(source) > 500 else source)
                                    st.divider()

                        # Add to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
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