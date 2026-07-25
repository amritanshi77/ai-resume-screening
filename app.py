import streamlit as st
import pandas as pd
from src.resume_parser import extract_resume_text
from src.preprocess import preprocess_text, extract_skills
from src.matcher import rank_jobs

st.set_page_config(page_title="AI Resume Screening & Job Recommendation", layout="wide")
st.title("AI Resume Screening & Job Recommendation System")
st.write("Upload your resume and get matched with the most relevant job openings.")

# Master list of skills to check for (expand this anytime)
SKILL_LIST = [
    "python", "java", "javascript", "react", "node.js", "express",
    "sql", "excel", "html", "css", "machine learning", "deep learning",
    "pandas", "numpy", "scikit-learn", "spacy", "nlp", "transformers",
    "data visualization", "statistical modeling", "rest apis",
    "database design", "git", "docker", "aws"
]

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
        resume_skills = extract_skills(raw_text, SKILL_LIST)

    st.subheader("Your Detected Skills")
    if resume_skills:
        st.write(", ".join(resume_skills))
    else:
        st.write("No matching skills detected from our skill list.")

    st.subheader("Top Job Matches")
    for _, row in ranked_jobs.head(5).iterrows():
        score_pct = round(row["similarity_score"] * 100, 2)
        job_skills = extract_skills(row["description"], SKILL_LIST)

        matched_skills = [s for s in job_skills if s in resume_skills]
        missing_skills = [s for s in job_skills if s not in resume_skills]

        st.markdown(f"### {row['job_title']} — {row['company']}")
        st.progress(min(int(score_pct), 100))
        st.write(f"**Match Score:** {score_pct}%")
        st.write(row["description"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Matched Skills**")
            st.write(", ".join(matched_skills) if matched_skills else "None")
        with col2:
            st.markdown("**❌ Missing Skills**")
            st.write(", ".join(missing_skills) if missing_skills else "None")

        st.divider()

    with st.expander("View extracted resume text"):
        st.text(raw_text)