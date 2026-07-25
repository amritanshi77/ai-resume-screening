import streamlit as st
import pandas as pd
from src.resume_parser import extract_resume_text
from src.preprocess import preprocess_text
from src.matcher import rank_jobs

st.set_page_config(page_title="AI Resume Screening & Job Recommendation", layout="wide")
st.title("AI Resume Screening & Job Recommendation System")
st.write("Upload your resume and get matched with the most relevant job openings.")

@st.cache_data
def load_jobs():
    jobs_df = pd.read_csv("data/jobs.csv")
    jobs_df["processed_description"] = jobs_df["description"].apply(preprocess_text)
    return jobs_df

jobs_df = load_jobs()

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    with st.spinner("Analyzing your resume..."):
        raw_text = extract_resume_text(uploaded_file, uploaded_file.name)
        processed_resume = preprocess_text(raw_text)
        ranked_jobs = rank_jobs(processed_resume, jobs_df)

    st.subheader("Top Job Matches")
    for _, row in ranked_jobs.head(5).iterrows():
        score_pct = round(row["similarity_score"] * 100, 2)
        st.markdown(f"### {row['job_title']} — {row['company']}")
        st.progress(min(int(score_pct), 100))
        st.write(f"**Match Score:** {score_pct}%")
        st.write(row["description"])
        st.divider()

    with st.expander("View extracted resume text"):
        st.text(raw_text)