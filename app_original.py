import streamlit as st
import os
from models.llm import AzureOpenAIModel
from utils.memory import ChatMemory
from utils.rag import RAGSystem
from utils.web_search import WebSearchTool
from config.config import config
import uuid

# Page configuration
st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'memory' not in st.session_state:
    st.session_state.memory = ChatMemory(st.session_state.session_id)
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = AzureOpenAIModel()
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if 'web_search' not in st.session_state:
    st.session_state.web_search = WebSearchTool()

def main():
    st.title("🏥 Medical Research Assistant")
    st.subheader("Your intelligent companion for medical research, document analysis, and knowledge discovery")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Validate API keys
        if not config.validate():
            st.error("⚠️ Please set your API keys in environment variables:")
            st.code("""
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
            """)
            st.stop()
        
        st.success("✅ API keys configured")
        
        # Response mode
        response_mode = st.selectbox(
            "Response Mode",
            ["Detailed", "Concise"],
            help="Choose between detailed explanations or concise answers"
        )
        
        # Tools configuration
        st.header("Research Tools")
        use_rag = st.checkbox("Use Document Knowledge", value=True, 
                             help="Search through uploaded research documents and data files")
        use_web_search = st.checkbox("Enable Web Search", value=True,
                                    help="Search for latest medical research online")
        
        # Document upload
        if use_rag:
            st.header("📚 Research Document & Data Upload")
            st.info("Upload medical research papers, clinical studies, medical documentation, CSV data, or JSON logs")
            uploaded_files = st.file_uploader(
                "Upload documents (PDF, DOCX, TXT, CSV, JSON)",
                type=['pdf', 'docx', 'txt', 'csv', 'json'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Process Research Documents & Data"):
                    with st.spinner("Processing medical documents and data files..."):
                        # Save uploaded files temporarily
                        temp_paths = []
                        for uploaded_file in uploaded_files:
                            temp_path = f"./temp/{uploaded_file.name}"
                            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)
                        
                        # Process documents
                        doc_count = st.session_state.rag_system.process_and_store_documents(temp_paths)
                        st.success(f"✅ Processed {doc_count} research document and data chunks")
                        
                        # Clean up temp files
                        for path in temp_paths:
                            try:
                                os.remove(path)
                            except:
                                pass
        
        # Chat management
        st.header("Chat Management")
        if st.button("Clear Chat History"):
            st.session_state.memory.clear_memory()
            st.rerun()
    
    # Main chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        messages = st.session_state.memory.get_messages(include_metadata=True)
        for message in messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("metadata", {}).get("sources"):
                    with st.expander("Research Sources"):
                        for source in message["metadata"]["sources"]:
                            # Check if source is a dictionary with title and URL
                            if isinstance(source, dict) and "title" in source and "url" in source:
                                st.markdown(f"[{source['title']}]({source['url']})")
                            else:
                                st.write(source)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about medical research, clinical studies, healthcare data, or patient records..."):
        # Add user message to memory
        st.session_state.memory.add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Researching and analyzing data..."):
                response, sources = generate_response(
                    prompt, 
                    response_mode, 
                    use_rag, 
                    use_web_search
                )
            
            st.write(response)
            
            if sources:
                with st.expander("Research Sources"):
                    for source in sources:
                        # Check if source is a dictionary with title and URL
                        if isinstance(source, dict) and "title" in source and "url" in source:
                            st.markdown(f"[{source['title']}]({source['url']})")
                        else:
                            st.write(source)
        
        # Add assistant response to memory
        metadata = {"sources": sources} if sources else {}
        st.session_state.memory.add_message("assistant", response, metadata)

def generate_response(prompt: str, response_mode: str, use_rag: bool, use_web_search: bool) -> tuple:
    """Generate response using available tools with medical research focus"""
    context = ""
    sources = []
    
    # Get RAG context
    if use_rag:
        rag_context, rag_sources = st.session_state.rag_system.retrieve_relevant_context(prompt)
        if rag_context:
            context += rag_context
            # Add actual file names instead of generic message
            sources.extend([f"📄 Data: {source}" for source in rag_sources])
    
    # Get web search context
    web_results = []
    if use_web_search and should_use_web_search(prompt):
        search_results = st.session_state.web_search.search(prompt)
        web_context = st.session_state.web_search.format_search_results(search_results)
        context += web_context
        
        # Get search results with URLs for display
        web_results = st.session_state.web_search.get_search_results_with_urls(search_results)
        if web_results:
            sources.extend([{"title": f"🌐 {result['title']}", "url": result["url"]} for result in web_results])
    
    # Prepare system prompt with medical research focus
    system_prompt = f"""You are a Medical Research Assistant specializing in helping researchers and healthcare professionals with medical research, clinical studies, healthcare data analysis, and knowledge discovery. 

Response Mode: {response_mode}
- If Concise: Provide brief, to-the-point answers with key findings
- If Detailed: Provide comprehensive, well-explained responses with clinical context

Guidelines:
1. Be accurate, evidence-based, and professional in medical terminology
2. Cite your sources when using provided context from documents or data files
3. If you don't know something, say so honestly - do not speculate on medical advice
4. Focus on research-backed information and clinical evidence
5. Adapt your response length to the selected mode
6. Always prioritize patient safety and ethical considerations
7. Clearly distinguish between established facts and emerging research
8. When analyzing CSV or JSON data, provide insights based on the structured information
9. For patient data, maintain privacy and focus on aggregate patterns rather than individual records

Available Context:
{context if context else "No additional context available."}

Important: You are a research assistant, not a licensed medical professional. Always recommend consulting with healthcare providers for medical decisions.
"""
    
    # Get chat history
    messages = st.session_state.memory.get_messages()
    messages.append({"role": "user", "content": prompt})
    
    # Set max tokens based on response mode with minimum limits
    if response_mode == "Concise":
        max_tokens = max(config.CONCISE_MAX_TOKENS, 300)  # Ensure minimum 300 tokens
    else:
        max_tokens = config.DETAILED_MAX_TOKENS
    
    # Generate response
    response = st.session_state.llm_model.generate_response(
        messages=messages,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )
    
    return response, sources

def should_use_web_search(prompt: str) -> bool:
    """Determine if web search should be used for this medical research prompt"""
    web_search_indicators = [
        "latest", "recent", "current", "news", "today", "2024", "2025",
        "what's happening", "breaking", "update", "trend", "new study",
        "clinical trial", "latest research", "recent findings"
    ]
    return any(indicator in prompt.lower() for indicator in web_search_indicators)

if __name__ == "__main__":
    main()