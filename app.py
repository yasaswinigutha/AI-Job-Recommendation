
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Job Recommendation Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "jobs.csv"


# =========================================================
# 2. PROFESSIONAL DASHBOARD STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    .main-header {
        font-size: 2.35rem;
        font-weight: 800;
        color: #172554;
        margin-bottom: 0.15rem;
    }

    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 750;
        color: #172554;
    }

    .metric-note {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 4px;
    }

    .job-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        min-height: 190px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .job-rank {
        font-size: 0.8rem;
        font-weight: 700;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .job-name {
        font-size: 1.15rem;
        font-weight: 750;
        color: #172554;
        margin: 12px 0;
    }

    .job-score {
        font-size: 1.65rem;
        font-weight: 800;
        color: #2563eb;
    }

    .job-caption {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 6px;
    }

    .section-heading {
        color: #172554;
        font-size: 1.4rem;
        font-weight: 750;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }

    .info-banner {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 13px 16px;
        color: #1e3a8a;
        margin: 12px 0 20px 0;
    }

    div.stButton > button[kind="primary"] {
        background: #2563eb;
        border: none;
        border-radius: 10px;
        font-weight: 650;
        min-height: 3rem;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. ROLE SKILL CHECKLISTS
# =========================================================

ROLE_SKILLS = {
    "Python Developer": [
        "Python", "SQL", "Git", "REST API", "Django",
        "Flask", "OOP", "Unit Testing"
    ],
    "Data Analyst": [
        "SQL", "Excel", "Python", "Power BI",
        "Tableau", "Statistics", "Data Visualization"
    ],
    "Web Developer": [
        "HTML", "CSS", "JavaScript", "React",
        "REST API", "Git", "Responsive Design"
    ],
    "Java Developer": [
        "Java", "OOP", "SQL", "Spring Boot",
        "REST API", "Git", "Unit Testing"
    ],
    "Machine Learning Engineer": [
        "Python", "Machine Learning", "Scikit-learn",
        "Pandas", "NumPy", "SQL", "Model Evaluation"
    ],
    "Data Scientist": [
        "Python", "SQL", "Statistics", "Pandas",
        "NumPy", "Machine Learning", "Data Visualization"
    ],
    "Frontend Developer": [
        "HTML", "CSS", "JavaScript", "React",
        "Responsive Design", "Git", "Accessibility"
    ],
    "Backend Developer": [
        "Python", "SQL", "REST API", "Authentication",
        "Database Design", "Git", "Django"
    ],
    "SQL Developer": [
        "SQL", "Joins", "Subqueries", "CTE",
        "Window Functions", "Indexing", "Query Optimization"
    ],
    "Full Stack Developer": [
        "HTML", "CSS", "JavaScript", "React",
        "Node.js", "SQL", "REST API", "Git"
    ]
}


SKILL_ALIASES = {
    "rest api": [
        "rest api", "restful api", "rest services",
        "api development", "apis"
    ],
    "node.js": ["node.js", "nodejs", "node js"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "cte": ["cte", "common table expression"],
    "window functions": ["window functions"],
    "query optimization": [
        "query optimization", "query tuning"
    ],
    "unit testing": [
        "unit testing", "unit tests", "pytest"
    ],
    "responsive design": [
        "responsive design", "responsive websites",
        "responsive web design"
    ],
    "power bi": ["power bi", "powerbi"],
    "oop": ["oop", "object-oriented programming"],
    "javascript": ["javascript", "java script"],
    "sql": ["sql", "structured query language", "mysql",
            "postgresql", "sql server", "pl sql"],
    "git": ["git", "github"],
    "database design": ["database design", "database modeling"],
    "indexing": ["indexing", "database indexes", "indexes"],
    "subqueries": ["subqueries", "subquery", "nested queries"],
    "joins": ["joins", "inner join", "left join",
              "right join", "full outer join"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "react.js", "reactjs"],
    "python": ["python"],
    "java": ["java", "java programming"],
    "flask": ["flask"],
    "django": ["django"],
    "spring boot": ["spring boot"],
    "machine learning": ["machine learning", "ml"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "excel": ["excel", "microsoft excel"],
    "statistics": ["statistics", "statistical analysis"],
    "data visualization": ["data visualization",
                           "data visualisation"],
    "authentication": ["authentication", "user authentication"],
    "accessibility": ["accessibility", "web accessibility"],
    "model evaluation": ["model evaluation",
                         "model performance evaluation"],
    "flask": ["flask"],
    "tableau": ["tableau"],
}


# =========================================================
# 4. SKILL DETECTION FUNCTIONS
# =========================================================

def find_resume_skills(resume_text, skills):
    """Detect checklist skills using keyword matching."""

    text = resume_text.lower()
    detected = []

    for skill in skills:
        variants = SKILL_ALIASES.get(
            skill.lower(), [skill.lower()]
        )

        for variant in variants:
            pattern = (
                r"(?<!\w)"
                + re.escape(variant)
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                detected.append(skill)
                break

    return detected


def get_dataset_skills(jobs_df, role):
    """Collect skill descriptions associated with a role."""

    role_rows = jobs_df[jobs_df["job_role"] == role]
    skills = set()

    for skill_text in role_rows["skills"]:
        for item in re.split(r"[,;\n]", str(skill_text)):
            item = item.strip()

            if item:
                skills.add(item)

    return sorted(skills, key=str.lower)


# =========================================================
# 5. LOAD DATASET AND TRAIN THE MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"skills", "job_role"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=["skills", "job_role"]).copy()

    df["skills"] = df["skills"].astype(str).str.strip()
    df["job_role"] = df["job_role"].astype(str).str.strip()

    df = df[
        (df["skills"] != "")
        & (df["job_role"] != "")
    ]

    if df["job_role"].nunique() < 2:
        raise ValueError(
            "At least two different job roles are required."
        )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(max_iter=2000)
        )
    ])

    model.fit(df["skills"], df["job_role"])

    return model, df


try:
    model, jobs_df = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 💼 CareerAI")
    st.caption("AI Job Recommendation System")

    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠 Dashboard")
    st.markdown("📄 Resume Analysis")
    st.markdown("🎯 Job Recommendations")
    st.markdown("🧩 Skill Gap Analysis")

    st.divider()

    st.markdown("### How to use")
    st.markdown(
        """
        1. Upload your resume.
        2. Review extracted text.
        3. Click **Recommend Top 3 Jobs**.
        4. Explore recommendations.
        5. Review skills to consider learning.
        """
    )

    st.divider()

    st.markdown("### Project information")
    st.write("**Technology:** Python")
    st.write("**Interface:** Streamlit")
    st.write("**ML:** TF-IDF + Logistic Regression")
    st.write("**Input:** PDF / TXT resume")

    st.caption(
        "Academic prototype using a small synthetic dataset."
    )


# =========================================================
# 7. DASHBOARD HEADER
# =========================================================

st.markdown(
    '<div class="main-header">AI Job Recommendation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'Discover career options from your resume and explore '
    'skills associated with each recommended role.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-banner">'
    '📌 Recommendations are exploratory. Model scores are '
    'not verified measures of job suitability or hiring chances.'
    '</div>',
    unsafe_allow_html=True
)

st.success(
    f"AI model loaded successfully · "
    f"{len(jobs_df)} training examples · "
    f"{jobs_df['job_role'].nunique()} job roles"
)


# =========================================================
# 8. RESUME UPLOAD
# =========================================================

st.markdown(
    '<div class="section-heading">📄 Resume Analysis</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or TXT)",
    type=["pdf", "txt"],
    help="Use a text-based PDF or plain text resume."
)

resume_text = ""

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    st.error(
                        "The PDF is password-protected. "
                        "Upload an accessible PDF."
                    )
                    st.stop()

            resume_text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        else:
            resume_text = uploaded_file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

    except Exception as error:
        st.error(f"Unable to read the resume: {error}")
        st.stop()

    if not resume_text.strip():
        st.warning(
            "No readable text was extracted. "
            "Scanned PDFs may require OCR."
        )
        st.stop()

    st.success(
        f"Resume uploaded: {uploaded_file.name}"
    )

    with st.expander("🔍 Preview extracted resume text"):
        st.text_area(
            "Resume content",
            resume_text[:5000],
            height=200,
            key="resume_preview"
        )


# =========================================================
# 9. ANALYZE BUTTON
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

current_file_name = (
    uploaded_file.name if uploaded_file is not None else None
)

if (
    st.session_state.analysis_result is not None
    and st.session_state.analysis_result["file_name"]
    != current_file_name
):
    st.session_state.analysis_result = None


if uploaded_file is not None and resume_text.strip():
    if st.button(
        "🚀 Recommend Top 3 Jobs",
        type="primary",
        use_container_width=True
    ):
        probabilities = model.predict_proba([resume_text])[0]
        roles = model.named_steps["classifier"].classes_

        ranked_indices = probabilities.argsort()[::-1][:3]

        recommendations = []

        for idx in ranked_indices:
            role = str(roles[idx])
            score = float(probabilities[idx])

            required_skills = ROLE_SKILLS.get(role, [])

            detected_skills = find_resume_skills(
                resume_text, required_skills
            )

            missing_skills = [
                skill for skill in required_skills
                if skill not in detected_skills
            ]

            if required_skills:
                coverage = (
                    len(detected_skills) / len(required_skills)
                ) * 100
            else:
                coverage = 0.0

            recommendations.append({
                "role": role,
                "score": score,
                "detected": detected_skills,
                "missing": missing_skills,
                "coverage": coverage,
                "dataset_skills": get_dataset_skills(
                    jobs_df, role
                )
            })

        all_detected = set()

        for item in recommendations:
            all_detected.update(item["detected"])

        st.session_state.analysis_result = {
            "file_name": current_file_name,
            "recommendations": recommendations,
            "detected_count": len(all_detected),
            "resume_chars": len(resume_text)
        }


# =========================================================
# 10. DASHBOARD METRICS AND JOB CARDS
# =========================================================

result = st.session_state.analysis_result

if result is not None:
    recommendations = result["recommendations"]

    st.divider()

    st.markdown(
        '<div class="section-heading">📊 Resume Summary</div>',
        unsafe_allow_html=True
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Resume Status</div>
                <div class="metric-value">Ready</div>
                <div class="metric-note">Text extracted successfully</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Roles Recommended</div>
                <div class="metric-value">{len(recommendations)}</div>
                <div class="metric-note">Top matching model classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Checklist Skills Detected</div>
                <div class="metric-value">{result['detected_count']}</div>
                <div class="metric-note">Across the three recommended roles</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-heading">'
        '🎯 Top 3 Job Recommendations</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Model probability estimates are specific to this classifier "
        "and dataset. They are not verified suitability percentages."
    )

    card_columns = st.columns(len(recommendations))

    for rank, (column, item) in enumerate(
        zip(card_columns, recommendations),
        start=1
    ):
        with column:
            st.markdown(
                f"""
                <div class="job-card">
                    <div class="job-rank">Recommendation #{rank}</div>
                    <div class="job-name">{item['role']}</div>
                    <div class="job-score">
                        {item['score'] * 100:.1f}%
                    </div>
                    <div class="job-caption">
                        Model probability estimate
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(item["score"])

            if rank == 1:
                st.success("Highest-ranked result")

    # =====================================================
    # 11. SKILL GAP ANALYSIS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">🧩 Skill Gap Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Review the checklist skills detected in your resume "
        "and the skills that were not detected."
    )

    for rank, item in enumerate(recommendations, start=1):
        role = item["role"]

        with st.expander(
            f"{rank}. {role} — "
            f"{len(item['detected'])} of "
            f"{len(item['detected']) + len(item['missing'])} "
            f"checklist skills detected"
        ):
            st.markdown("#### Skills associated with this role")

            if item["dataset_skills"]:
                st.write(", ".join(item["dataset_skills"]))
            else:
                st.write("No associated skills found in the dataset.")

            detected_col, missing_col = st.columns(2)

            with detected_col:
                st.markdown("#### ✅ Skills detected")

                if item["detected"]:
                    for skill in item["detected"]:
                        st.write(f"• {skill}")
                else:
                    st.info(
                        "No skills from this checklist were detected."
                    )

            with missing_col:
                st.markdown("#### 📚 Skills to consider learning")

                if item["missing"]:
                    for skill in item["missing"]:
                        st.write(f"• {skill}")
                else:
                    st.success(
                        "All checklist skills were detected."
                    )

            if (
                item["detected"] or item["missing"]
            ):
                st.markdown("**Checklist skill coverage**")

                st.progress(item["coverage"] / 100)

                st.write(
                    f"{item['coverage']:.0f}% "
                    f"({len(item['detected'])} of "
                    f"{len(item['detected']) + len(item['missing'])} "
                    f"checklist skills detected)"
                )

    # =====================================================
    # 12. FINAL NOTES
    # =====================================================

    st.divider()

    st.markdown("### 💡 Recommended next steps")

    combined_missing = []

    for item in recommendations:
        combined_missing.extend(item["missing"])

    combined_missing = list(dict.fromkeys(combined_missing))

    if combined_missing:
        st.write(
            "Consider reviewing these skills based on your "
            "recommended roles:"
        )

        st.write(", ".join(combined_missing))
    else:
        st.write(
            "All skills in the current checklists were detected. "
            "You can explore advanced skills for your target roles."
        )

    st.info(
        "Skill detection is based on resume keywords. A skill not "
        "detected here may still be one you know. Checklist coverage "
        "does not measure proficiency. Recommendations are based on "
        "a small synthetic dataset and should be treated as a prototype."
    )

else:
    st.info(
        "👆 Upload your resume above, then click "
        "**Recommend Top 3 Jobs** to populate the dashboard."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Job Recommendation System · "
    "Python · Streamlit · Pandas · Scikit-learn"
)
