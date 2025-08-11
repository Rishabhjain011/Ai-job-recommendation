import streamlit as st
import pandas as pd
import os
#from langchain_community.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
from langchain.chains.llm import LLMChain
import docx2txt
from PyPDF2 import PdfReader

# Load and prepare job data once (cache to avoid reloading)
@st.cache_data
def load_job_data():
    url = "https://raw.githubusercontent.com/stevendoll/kaggle-jobs/master/indeed-job-listings.csv"
    df = pd.read_csv(url)
    df = df[['job_title', 'company', 'location', 'body']]
    df = df.rename(columns={'job_title': 'title', 'body': 'description'})
    df.dropna(subset=['description'], inplace=True)
    return df

# Prepare documents and vectorstore once (cache)
import sys
import asyncio

@st.cache_resource
def create_vectorstore(df):
    docs = []
    for _, row in df.iterrows():
        meta = {"title": row['title'], "company": row['company'], "location": row['location']}
        docs.append(Document(page_content=row['description'], metadata=meta))
    
    # Correct event loop setup depending on Python version
    if sys.version_info >= (3, 10):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    else:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# Extract text from uploaded resume file
def extract_text_from_file(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file)
        texts = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(texts)
    elif ext in [".docx", ".doc"]:
        file.seek(0)  # Reset file pointer for docx2txt
        return docx2txt.process(file)
    else:
        st.error("Unsupported file format. Please upload PDF or DOCX.")
        return None

# Skill extraction chain (reuse your existing prompt)
def extract_skills(llm, resume_text):
    skills_prompt = PromptTemplate(
        input_variables=["resume_text"],
        template="""
Extract a comma-separated list of core technical skills from the following resume:
{resume_text}
Only list skills, nothing else.
"""
    )
    skills_chain = LLMChain(llm=llm, prompt=skills_prompt)
    return skills_chain.run(resume_text=resume_text).strip()

# Format retrieved jobs for prompt
def format_jobs_for_prompt(retrieved):
    blocks = []
    for i, d in enumerate(retrieved, start=1):
        m = d.metadata
        blocks.append(f"{i}. Title: {m.get('title')}\n   Company: {m.get('company')}\n   Location: {m.get('location')}\n   Description: {d.page_content}")
    return "\n\n".join(blocks)

# Generate job recommendations using LLM and retrieved jobs
def generate_recommendations(llm, vectorstore, profile_text, query_text, top_k=3):
    retrieved = vectorstore.similarity_search(query_text, k=top_k)
    jobs_block = format_jobs_for_prompt(retrieved)

    recommend_prompt_template = """
You are a helpful career assistant.
User profile:
{profile}

Here are candidate job postings:
{jobs}

For each job (top {top_k}), write:
- 1–2 line reason why it matches the user's profile
- 2 short action items to improve fit (skills, keywords, small projects)
Return numbered sections, one per job.
"""
    prompt = PromptTemplate(
        input_variables=["profile", "jobs", "top_k"],
        template=recommend_prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.run(profile=profile_text, jobs=jobs_block, top_k=str(top_k))
    return out, retrieved

# Main Streamlit app
def main():
    st.title("AI-Powered Job Recommendation System")

    # API key input (or set via environment)
    api_key = "AIzaSyCBvGKebqC-oalIwoyD94PPol-ysTkhqYo"
    if not api_key or api_key !="AIzaSyCBvGKebqC-oalIwoyD94PPol-ysTkhqYo":
        st.warning("Please enter your Google API Key to proceed.")
        st.stop()
    os.environ["GOOGLE_API_KEY"] = api_key

    df = load_job_data()
    vectorstore = create_vectorstore(df)

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    st.header("Step 1: Upload Your Resume (PDF or DOCX)")
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'doc'])

    if uploaded_file:
        resume_text = extract_text_from_file(uploaded_file)
        if resume_text:
            st.subheader("Extracted Resume Text (Preview)")
            st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)

            with st.spinner("Extracting skills from resume..."):
                skills = extract_skills(llm, resume_text)
            st.success("Skills extracted:")
            st.write(skills)

            st.header("Step 2: Provide Your Job Preferences")
            remote_pref = st.checkbox("Remote Work Preferred", value=True)
            internship_pref = st.checkbox("Open to Internships", value=True)

            profile_text = f"Skills: {skills}\nPreferences: "
            prefs = []
            if remote_pref:
                prefs.append("Remote")
            if internship_pref:
                prefs.append("Internship")
            profile_text += ", ".join(prefs)

            st.header("Step 3: Get Job Recommendations")
            if st.button("Generate Recommendations"):
                with st.spinner("Generating recommendations..."):
                    recommendations, jobs = generate_recommendations(llm, vectorstore, profile_text, skills, top_k=3)
                st.subheader("Recommended Jobs")
                st.text(recommendations)

if __name__ == "__main__":
    main()
