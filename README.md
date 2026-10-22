# AI Job Recommendation System 🤖

## 📌 Project Overview

The **AI Job Recommendation System** is a machine learning web application that analyzes a user's resume and recommends suitable job roles based on the skills and information extracted from it.

The application uses Natural Language Processing (NLP) techniques and machine learning to compare resume content with job-related data. It displays the top three recommended job roles and helps users identify skills they may need to improve.

## 🎯 Objectives

- Analyze resumes and identify relevant skills.
- Recommend suitable job roles using machine learning.
- Display the top three job recommendations with model scores.
- Identify detected skills and suggest additional skills to learn.
- Provide a simple, interactive web interface for users.

## ✨ Features

- **Resume Upload:** Upload a resume in PDF or TXT format.
- **AI-Based Recommendations:** Get the top three recommended job roles.
- **Skill Detection:** Identify relevant skills from resume text.
- **Skill Gap Analysis:** Highlight skills to consider learning for recommended roles.
- **Interactive Dashboard:** View recommendations and skill information through a Streamlit interface.
- **Web Deployment:** Access the application online.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **NLP and Text Processing:** TF-IDF Vectorization
- **ML Algorithm:** Logistic Regression
- **Data Handling:** Pandas
- **Web Framework:** Streamlit
- **Resume Processing:** PDF text extraction
- **Version Control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## 🧠 Machine Learning Approach

The project uses the following workflow:

1. **Dataset Preparation:** Prepare job-role examples containing skills and descriptions.
2. **Text Preprocessing:** Combine relevant text fields into a single text representation.
3. **Feature Extraction:** Apply TF-IDF Vectorization to convert text into numerical features.
4. **Model Training:** Train a Logistic Regression classifier to predict job-role categories.
5. **Resume Analysis:** Extract text from the uploaded resume.
6. **Job Recommendation:** Predict job-role scores and display the top three recommendations.
7. **Skill Gap Analysis:** Compare detected resume skills with skills associated with recommended roles and suggest areas for improvement.

## 📊 Dataset

The project uses a dataset containing **110 examples across 10 job roles**, with 11 examples per role in the expanded dataset.

The job roles include:

- Python Developer
- Data Analyst
- Web Developer
- Java Developer
- Machine Learning Engineer
- Data Scientist
- Frontend Developer
- Backend Developer
- SQL Developer
- Full Stack Developer

## 🚀 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/yasaswinigutha/AI-Job-Recommendation.git
```

### 2. Navigate to the Project Folder

```bash
cd AI-Job-Recommendation
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in your terminal.

## 🌐 Live Demo

Try the deployed application:

[AI Job Recommendation System](https://ai-job-recommendation-yasaswinigutha-app.streamlit.app/)

*If this link does not open, use the current app URL displayed in your Streamlit Community Cloud settings.*

## 📁 Project Structure

```text
AI-Job-Recommendation/
│
├── app.py
├── jobs.csv
├── jobs_backup.csv
├── data_preprocessing.py
├── text_preprocessing.py
├── train_model.py
├── expand_dataset.py
├── requirements.txt
├── README.md
└── py.py
```

## 🔮 Future Enhancements

- Improve recommendation accuracy by expanding the dataset.
- Enhance resume parsing to extract more skills and qualifications.
- Add more job roles and industry-specific skills.
- Improve skill-gap analysis with standardized skill categories.
- Provide personalized learning resources for missing skills.
- Evaluate the model using additional metrics and validation methods.

## ⚠️ Disclaimer

The recommendations are intended for educational and career-exploration purposes only. Model scores are exploratory predictions and should not be interpreted as verified suitability, hiring probability, or a guarantee of employment.

## 👩‍💻 Author

**Gutha Venkata Yasaswini**

B.Tech — Computer Science and Engineering

GitHub: [yasaswinigutha](https://github.com/yasaswinigutha)
