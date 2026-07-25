# AI Resume Screening & Job Recommendation System

An AI-powered web app that analyzes resumes and matches them with relevant job descriptions using Natural Language Processing (NLP).

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- spaCy

## How It Works
1. User uploads a resume (PDF or DOCX)
2. Text is extracted from the resume
3. Text is cleaned and processed using spaCy (stopword removal, lemmatization)
4. TF-IDF vectorization + cosine similarity is used to compare the resume against job descriptions
5. Top matching jobs are displayed with a similarity score

## Project Structure
ai-resume-screening/
├── data/
│ └── jobs.csv
├── src/
│ ├── resume_parser.py
│ ├── preprocess.py
│ └── matcher.py
├── app.py
├── requirements.txt
└── README.md

## Setup & Run Locally

```bash
git clone https://github.com/amritanshi77/ai-resume-screening.git
cd ai-resume-screening
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```

## Future Improvements
- Extract and highlight matched/missing skills
- Allow users to input custom job descriptions
- Deploy live on Streamlit Community Cloud