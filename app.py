import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

st.set_page_config(
    page_title="AI News Summarizer",
    page_icon="📰",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp { background-color: #0f1117; }
    h1 { color: #00d4ff; text-align: center; }
    h3 { color: #00d4ff; }
    p { color: #888; text-align: center; }
    .news-card {
        background-color: #1e1e2e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d4ff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📰 AI News Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter any topic — AI searches and summarizes latest news</p>", unsafe_allow_html=True)
st.divider()

api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    api_key=api_key,
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)

search = DuckDuckGoSearchRun()
tools = [search]
agent = create_react_agent(llm, tools)

col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Enter a topic:", placeholder="e.g. Artificial Intelligence, IPL 2025, Stock Market")
with col2:
    num = st.selectbox("Stories:", [3, 5, 7])

if st.button("🔍 Get Latest News", use_container_width=True):
    if topic:
        with st.spinner(f"Searching latest news about {topic}..."):
            prompt = f"""Search for the latest news about "{topic}" and provide {num} recent news summaries.

For each story provide:
1. A clear headline
2. A 2-3 sentence summary
3. Why it matters

Format each story clearly separated. Be factual and concise."""

            response = agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            news = response["messages"][-1].content

        st.markdown(f"### 📰 Latest News: {topic}")
        st.divider()
        st.markdown(news)
        st.divider()
        st.download_button(
            label="📥 Download Summary",
            data=news,
            file_name=f"{topic}_news_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.warning("Please enter a topic first!")

st.divider()
st.markdown("<p>Built by Rohit • Powered by LangChain + DuckDuckGo</p>", unsafe_allow_html=True)
