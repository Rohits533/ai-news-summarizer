import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
import io

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

def generate_pdf(content, topic):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=20*mm, leftMargin=20*mm,
                           topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                fontSize=18, textColor=colors.HexColor('#00d4ff'),
                                spaceAfter=10)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                  fontSize=13, textColor=colors.HexColor('#00d4ff'),
                                  spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                               fontSize=10, spaceAfter=4, leading=14)
    story = []
    story.append(Paragraph(f"Latest News: {topic}", title_style))
    story.append(Spacer(1, 6*mm))
    for line in content.split('\n'):
        if line.strip() == '':
            story.append(Spacer(1, 3*mm))
        elif line.startswith('#') or line.startswith('**'):
            clean = line.replace('#', '').replace('**', '').strip()
            story.append(Paragraph(clean, heading_style))
        else:
            story.append(Paragraph(line.strip(), body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

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

        pdf_buffer = generate_pdf(news, topic)
        st.download_button(
            label="📥 Download as PDF",
            data=pdf_buffer,
            file_name=f"{topic}_news.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.warning("Please enter a topic first!")

st.divider()
st.markdown("<p>Built by Rohit • Powered by LangChain + DuckDuckGo</p>", unsafe_allow_html=True)
