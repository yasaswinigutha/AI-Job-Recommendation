
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "dataset" / "jobs.csv"

# Ten varied skill descriptions for each role.
new_data = {
    "Python Developer": [
        "Python, SQL, Django, REST API, Git",
        "Python, Flask, backend development, APIs, Git",
        "Python, Django REST Framework, PostgreSQL, authentication",
        "Python, OOP, unit testing, debugging, Git",
        "Python, FastAPI, JSON, REST services, SQL",
        "Python, Pandas, automation, scripting, file handling",
        "Python, Django, HTML, CSS, database integration",
        "Python, API development, MySQL, error handling, Git",
        "Python, pytest, virtual environments, packaging",
        "Python, SQLAlchemy, Flask, deployment, backend services",
    ],
    "Data Analyst": [
        "Excel, SQL, Power BI, statistics, dashboards",
        "Python, Pandas, NumPy, data cleaning, visualization",
        "SQL, Tableau, reporting, business metrics, Excel",
        "Power BI, DAX, data modeling, dashboards, Excel",
        "Excel, pivot tables, SQL queries, reporting",
        "Python, Matplotlib, exploratory data analysis, statistics",
        "SQL, KPI reporting, data validation, business analysis",
        "Tableau, Excel, data visualization, trend analysis",
        "Power Query, Power BI, ETL, reporting",
        "Python, SQL, data analysis, charts, statistical summaries",
    ],
    "Web Developer": [
        "HTML, CSS, JavaScript, responsive design, Git",
        "React, JavaScript, HTML, CSS, frontend development",
        "HTML, CSS, Bootstrap, responsive web pages",
        "JavaScript, DOM manipulation, form validation, CSS",
        "React, REST APIs, component design, CSS",
        "HTML, CSS, accessibility, responsive layouts",
        "JavaScript, React, state management, Git",
        "Node.js, Express, REST API, JavaScript",
        "TypeScript, React, reusable components, testing",
        "Frontend development, browser debugging, HTML, CSS, JavaScript",
    ],
    "Java Developer": [
        "Java, OOP, collections, exception handling, SQL",
        "Java, Spring Boot, REST APIs, Maven, Git",
        "Core Java, multithreading, JDBC, MySQL",
        "Java, Spring MVC, Hibernate, PostgreSQL",
        "Java, data structures, algorithms, unit testing",
        "Spring Boot, JPA, REST services, database integration",
        "Java, object-oriented design, debugging, JUnit",
        "Java, JDBC, SQL, transaction management",
        "Java, Maven, Git, application development",
        "Java, Spring Security, authentication, API development",
    ],
    "Data Scientist": [
        "Python, Pandas, NumPy, statistics, machine learning",
        "Python, scikit-learn, regression, model evaluation",
        "Jupyter, exploratory analysis, feature engineering, statistics",
        "Python, classification, clustering, data visualization",
        "Machine learning, Python, cross-validation, metrics",
        "Pandas, NumPy, hypothesis testing, statistical modeling",
        "Python, feature selection, model tuning, scikit-learn",
        "Statistics, regression, predictive modeling, Python",
        "Python, NLP, text preprocessing, classification",
        "Machine learning, data cleaning, experimentation, Python",
    ],
    "Machine Learning Engineer": [
        "Python, scikit-learn, model deployment, APIs, Git",
        "Python, TensorFlow, neural networks, model training",
        "PyTorch, Python, deep learning, GPU training",
        "Machine learning pipelines, Docker, Python, testing",
        "Python, FastAPI, inference APIs, model serving",
        "MLflow, experiment tracking, model versioning, Python",
        "Feature engineering, scikit-learn, deployment, monitoring",
        "Python, TensorFlow, data pipelines, model evaluation",
        "Docker, REST APIs, machine learning, CI/CD",
        "Python, model optimization, batch inference, testing",
    ],
    "DevOps Engineer": [
        "Linux, Docker, Kubernetes, CI/CD, Git",
        "Jenkins, Docker, automated deployment, Linux",
        "AWS, Terraform, infrastructure as code, monitoring",
        "Linux, Bash scripting, networking, troubleshooting",
        "Kubernetes, Helm, container orchestration, YAML",
        "GitHub Actions, CI/CD, Docker, automated testing",
        "Ansible, Linux, configuration management, scripting",
        "Prometheus, Grafana, monitoring, alerting",
        "AWS, deployment pipelines, IAM, cloud infrastructure",
        "Terraform, Kubernetes, secrets management, automation",
    ],
    "Database Administrator": [
        "SQL, MySQL, database backup, recovery, monitoring",
        "PostgreSQL, query optimization, indexing, performance tuning",
        "Oracle Database, SQL, backup, access control",
        "MySQL, replication, database security, troubleshooting",
        "SQL Server, T-SQL, backup and recovery",
        "PostgreSQL, user permissions, monitoring, maintenance",
        "Database indexing, query plans, SQL optimization",
        "Oracle, database administration, recovery, performance",
        "MySQL, schema design, data integrity, backups",
        "SQL Server, database monitoring, access management",
    ],
    "Cybersecurity Analyst": [
        "Network security, SIEM, log analysis, incident response",
        "Linux, networking, vulnerability assessment, security tools",
        "Wireshark, TCP/IP, packet analysis, troubleshooting",
        "Security monitoring, threat detection, incident handling",
        "Nmap, vulnerability scanning, risk assessment",
        "SIEM, Splunk, alert triage, log correlation",
        "Firewalls, access control, network monitoring",
        "Security auditing, vulnerability management, reporting",
        "Python, security automation, log parsing, networking",
        "Incident response, threat intelligence, endpoint security",
    ],
    "Cloud Engineer": [
        "AWS, EC2, S3, IAM, cloud networking",
        "Azure, virtual machines, storage, identity management",
        "AWS, VPC, security groups, load balancing",
        "Docker, Kubernetes, cloud deployment, monitoring",
        "Terraform, AWS, infrastructure automation",
        "Azure, networking, monitoring, cloud security",
        "AWS Lambda, serverless architecture, API Gateway",
        "Cloud storage, backup, disaster recovery, IAM",
        "Linux, cloud infrastructure, networking, troubleshooting",
        "AWS, autoscaling, load balancers, CloudWatch",
    ],
    "Frontend Developer": [
    "HTML CSS JavaScript responsive design React",
    "HTML CSS Bootstrap frontend UI development",
    "JavaScript React components API integration",
    "CSS Flexbox Grid web page design",
    "React HTML CSS Git responsive websites",
    "JavaScript DOM manipulation frontend development",
    "React hooks state management HTML CSS",
    "Frontend testing debugging browser developer tools",
    "Responsive UI accessibility JavaScript React",
    "HTML CSS JavaScript website optimization"
],

"Backend Developer": [
    "Python Django REST API authentication",
    "Node.js Express API development MongoDB",
    "Java Spring Boot backend services SQL",
    "Python Flask server side programming",
    "REST APIs database integration Git",
    "Backend authentication authorization security",
    "API development testing debugging",
    "Django database models server side logic",
    "Node.js SQL database server development",
    "Backend deployment environment variables"
],

"SQL Developer": [
    "SQL joins subqueries aggregate functions",
    "MySQL stored procedures triggers views",
    "SQL query optimization indexing",
    "Database design normalization SQL",
    "Oracle SQL PL SQL procedures",
    "SQL Server queries database management",
    "Complex joins window functions CTE",
    "SQL reporting data extraction",
    "Database performance tuning SQL",
    "SQL transactions constraints and indexes"
],

"Full Stack Developer": [
    "HTML CSS JavaScript React Node.js",
    "MERN stack MongoDB Express React Node",
    "React frontend Node.js backend REST API",
    "Full stack web development SQL Git",
    "JavaScript React Express database integration",
    "Frontend backend authentication API development",
    "HTML CSS React Python Django SQL",
    "Web application development deployment Git",
    "React Node.js MongoDB CRUD operations",
    "Full stack debugging testing responsive design"
],
}

# Load the existing dataset
df = pd.read_csv(CSV_PATH)

required = {"skills", "job_role"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required}; found {list(df.columns)}"
    )

# Normalize role names for matching existing labels.
existing_roles = {
    role.strip().casefold(): role.strip()
    for role in df["job_role"].dropna().astype(str).unique()
}

rows = []
for role, descriptions in new_data.items():
    matched_role = existing_roles.get(role.casefold())

    if matched_role is None:
        print(f"Skipping unmatched role: {role}")
        continue

    for description in descriptions:
        rows.append({
            "skills": description,
            "job_role": matched_role,
        })

added = pd.DataFrame(rows)

# Preserve original data and any additional columns.
combined = pd.concat([df, added], ignore_index=True)
combined["skills"] = combined["skills"].astype(str).str.strip()
combined["job_role"] = combined["job_role"].astype(str).str.strip()

combined = combined.drop_duplicates(
    subset=["skills", "job_role"]
).reset_index(drop=True)

# Save a backup before updating the original CSV.
backup_path = CSV_PATH.with_name("jobs_backup.csv")
if not backup_path.exists():
    df.to_csv(backup_path, index=False)

combined.to_csv(CSV_PATH, index=False)

print("\nDataset updated successfully!")
print("Backup saved to:", backup_path)
print("Total examples:", len(combined))
print("\nExamples per job role:")
print(combined["job_role"].value_counts().to_string())
