from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_text, job_texts):
    """
    resume_text: single preprocessed resume string
    job_texts: list of preprocessed job description strings
    returns: list of similarity scores (0-1) aligned with job_texts
    """
    documents = [resume_text] + job_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    resume_vector = tfidf_matrix[0:1]
    job_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(resume_vector, job_vectors)[0]
    return scores

def rank_jobs(resume_text, jobs_df, text_column="processed_description"):
    """
    jobs_df: DataFrame with a job description text column
    returns: jobs_df with a similarity_score column, sorted descending
    """
    scores = compute_similarity(resume_text, jobs_df[text_column].tolist())
    jobs_df = jobs_df.copy()
    jobs_df["similarity_score"] = scores
    return jobs_df.sort_values("similarity_score", ascending=False)