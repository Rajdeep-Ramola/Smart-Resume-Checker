import streamlit as st
import re
import pandas as pd
import altair as alt
import numpy as np
from scipy.spatial.distance import cosine
import os
# --- REMOVED: requests, json (replaced by openai) ---
import json # Keep json for parsing
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
# --- ADDED: OpenAI imports ---
from openai import OpenAI
from openai import APIError

load_dotenv()

# --- Configuration & Constants ---

# Configuration for local model (used for Semantic Scoring)
LOCAL_MODEL_ID = "BAAI/bge-small-en-v1.5"

# --- OPENAI API CONFIGURATION (NEW) ---
OPENAI_MODEL = "gpt-4o" # Use a model supporting JSON mode
# Key variable changed to OPENAI_API_KEY
try:
    # Use st.secrets if in Streamlit Cloud, otherwise environment variable
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = ""

# Initialize OpenAI Client (Globally/Statically)
OPENAI_CLIENT = None
if OPENAI_API_KEY:
    try:
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Error initializing OpenAI Client: {e}")

# --- DYNAMIC JSON SCHEMAS (UNCHANGED, only used for defining structure) ---

UNIVERSAL_RESUME_PROPS = {
    "Relevant_Skills": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List technical skills, tools, and languages (e.g., Python, Scrum)."},
    "Certifications_and_Degrees": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List professional certifications and degrees (e.g., CPA, MBA, B.Tech CS)."},
    "Most_Relevant_Experience_Summary": {"type": "STRING", "description": "Summarize core professional experience relevant to this field in one concise paragraph."},
    "Key_Achievements": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Extract 3-5 major quantifiable accomplishments (e.g., Reduced cost by 15%)."},
    "Key_Projects": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List 3-5 most impactful personal or professional projects (e.g., E-commerce API, Fraud Detection Model)."},
}

JD_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Job_Title_Extracted": {"type": "STRING", "description": "The exact title of the job role (e.g., Senior Data Engineer)."},
        "Core_Responsibilities_Summary": {"type": "STRING", "description": "Summarize the 5-7 most important duties of this job in a single paragraph."},
        "Must_Have_Skills": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List all hard skills, languages, and required software mentioned as mandatory."},
        "Min_Years_Experience": {"type": "STRING", "description": "The minimum and preferred years of experience required (e.g., '5+ years' or '3-5 years')."},
        "Minimum_Education_Level": {"type": "STRING", "description": "The minimum education level specified (e.g., Bachelor's Degree, Master's Degree)."},
        "Preferred_Certifications": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List any specific certifications mentioned as preferred or required."},
    },
    "required": ["Job_Title_Extracted", "Core_Responsibilities_Summary", "Must_Have_Skills"]
}

# Role-Specific Schema Definition (Used for determining fields to request)
ROLE_SPECIFIC_SCHEMAS = {
    "Software Engineer": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Projects", "Certifications_and_Degrees"], "base": {}},
    "Data Scientist": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Projects", "Certifications_and_Degrees"], "base": {}},
    "Cloud Architect": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Cybersecurity Analyst": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Financial Analyst": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Accountant/Auditor": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Elementary School Teacher": {"fields": ["Teaching_Training_Summary", "Relevant_Skills", "Certifications_and_Degrees", "Key_Achievements"], "base": {
        "Teaching_Training_Summary": {"type": "STRING", "description": "Summarize all teaching experience, classroom management methods, and professional development training in one paragraph."},
    }},
    "University Professor": {"fields": ["Research_and_Publications", "Relevant_Skills", "Certifications_and_Degrees", "Key_Achievements"], "base": {
        "Research_and_Publications": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List major publications, grants, or research areas (e.g., Published 5 papers on deep learning)."},
    }},
    "Project Manager (IT/Tech)": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Digital Marketing Manager": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Sales Specialist": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "HR Generalist": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Supply Chain Analyst": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Health Informatics Specialist": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "Technical Writing": {"fields": ["Most_Relevant_Experience_Summary", "Relevant_Skills", "Key_Achievements", "Certifications_and_Degrees"], "base": {}},
    "High School Teacher": {"fields": ["Teaching_Training_Summary", "Relevant_Skills", "Certifications_and_Degrees", "Key_Achievements"], "base": {
        "Teaching_Training_Summary": {"type": "STRING", "description": "Summarize all teaching experience, classroom management methods, and professional development training in one paragraph."},
    }},
    "Special Education Teacher": {"fields": ["Teaching_Training_Summary", "Relevant_Skills", "Certifications_and_Degrees", "Key_Achievements"], "base": {
        "Teaching_Training_Summary": {"type": "STRING", "description": "Summarize all teaching experience, classroom management methods, and professional development training in one paragraph."},
    }},
}

# --- HIERARCHICAL JOB-SPECIFIC KEYWORDS (UNCHANGED) ---
HIERARCHICAL_JOB_KEYWORDS = {
    "Technology & Engineering": {
        "Software Engineer": {
             'python', 'java', 'c++', 'c#', 'oop', 'algorithms', 'data_structures',
             'testing', 'tdd', 'agile', 'scrum', 'backend', 'api', 'design_patterns', 'git',
             'bs_cs', 'ms_cs', 'btech', 'mtech'
        },
        "Cybersecurity Analyst": {
             'siem', 'soc', 'firewall', 'incident_response', 'palo_alto',
             'cissp', 'security', 'vulnerability', 'penetration', 'forensics',
             'threat_modeling', 'iso27001', 'security+', 'cism', 'ceh', 'bs_it'
        },
        "Cloud Architect": {
             'aws', 'azure', 'gcp', 'terraform', 'kubernetes', 'docker',
             'ci/cd', 'iac', 'vpc', 'ec2', 'lambda', 's3', 'network',
             'aws_cert', 'azure_cert', 'gcp_cert', 'ccnp'
        },
        "Web Developer": {
             'javascript', 'typescript', 'react', 'angular', 'vue', 'html',
             'css', 'node', 'express', 'django', 'flask', 'api', 'rest',
             'bs_cs', 'bca', 'mca', 'fullstack'
        }
    },
    "Data & AI": {
        "Data Scientist": {
             'python', 'r', 'scikit', 'tensorflow', 'pytorch', 'keras',
             'statistics', 'machine', 'learning', 'mlops', 'nlp', 'vision',
             'models', 'predictive', 'analysis', 'experimentation', 'ab_test',
             'ms_data_science', 'phd', 'masters', 'sas', 'spss'
        },
        "Data Engineer": {
             'python', 'sql', 'spark', 'hadoop', 'kafka', 'airflow', 'databricks',
             'snowflake', 'bigquery', 'redshift', 'etl', 'elt', 'dbt', 'pipeline',
             'ms_engineering', 'bs_it', 'btech'
        }
    },
    "Finance & Accounting": {
        "Financial Analyst": {
             'valuation', 'modeling', 'forecasting', 'budgeting', 'excel',
             'sap', 'due_diligence', 'financial_statements', 'pivot_tables',
             'mergers', 'acquisitions', 'vba',
             'cfa', 'mba', 'ms_finance', 'bs_finance'
        },
        "Accountant/Auditor": {
             'GAAP', 'IFRS', 'CPA', 'auditing', 'reconciliation',
             'accounts_payable', 'accounts_receivable', 'tax_prep',
             'quickbooks', 'journal_entries', 'compliance', 'sox',
             'cpa', 'ca', 'ms_accounting', 'bs_accounting'
        }
    },
    "Education Sector": {
        "Elementary School Teacher": {
             'classroom_management', 'common_core', 'phonics', 'curriculum_development',
             'differentiated_instruction', 'parent_communication', 'early_childhood',
             'masters_education', 'ba_education', 'teaching_license'
        },
        "University Professor": {
              'research_grants', 'peer_review', 'dissertation_advising',
              'publications', 'curriculum_design', 'academic_service',
              'phd', 'doctorate', 'tenure'
        }
    },
    "Project Management": {
        "Project Manager (IT/Tech)": {
             'pmp', 'agile', 'scrum', 'waterfall', 'jira', 'confluence',
             'risk_mitigation', 'stakeholder', 'budgeting', 'scheduling',
             'ms_project', 'scope', 'certifications',
             'pmp', 'csm', 'masters', 'mba'
        }
    },
    "Marketing & Sales": {
        "Digital Marketing Manager": {
             'SEO', 'SEM', 'PPC', 'Google_Analytics', 'HubSpot',
             'content_strategy', 'A/B_testing', 'lead_gen', 'social_media',
             'campaigns', 'crm', 'conversion_rate',
             'mba', 'google_cert', 'ms_marketing'
        }
    },
    "Human Resources (HR)": {
        "HR Generalist": {
             'SHRM', 'PHR', 'recruitment', 'onboarding', 'compensation',
             'benefits', 'HRIS', 'compliance', 'employee_relations',
             'talent_acquisition', 'succession_planning',
             'shrm_cp', 'phr', 'mba', 'masters'
        }
    },
    "Supply Chain & Logistics": {
        "Supply Chain Analyst": {
             'SCM', 'ERP', 'procurement', 'forecasting', 'logistics',
             'inventory', 'warehouse', 'SAP', 'optimisation', 'lean_six_sigma',
             'demand_planning',
             'cscp', 'apics', 'mba', 'masters'
        }
    },
    "Healthcare/Clinical": {
        "Health Informatics Specialist": {
             'HIPAA', 'EMR', 'EHR', 'ICD-10', 'clinical_trials',
             'HL7', 'LIS', 'PACS', 'compliance', 'health_data', 'informatics',
             'rhia', 'ccds', 'masters', 'bachelors'
        }
    },
    "Technical Writing": {
        "Technical Writer": {
             'MadCap', 'Confluence', 'markdown', 'DITA', 'API_documentation',
             'user_guides', 'editing', 'xml', 'authoring_tools', 'git',
             'masters', 'bachelors'
        }
    }
}

DOMAIN_OPTIONS = list(HIERARCHICAL_JOB_KEYWORDS.keys())
# --- End HIERARCHICAL_JOB_KEYWORDS ---

# --- Load Model Once (LOCAL EMBEDDING IMPLEMENTATION - UNCHANGED) ---

@st.cache_resource(show_spinner=f"Downloading and loading {LOCAL_MODEL_ID} Sentence Transformer...")
def load_local_embedding_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
        CACHE_DIR = os.path.join(os.getcwd(), ".model_cache")
        os.makedirs(CACHE_DIR, exist_ok=True)
        return SentenceTransformer(model_name, cache_folder=CACHE_DIR)
    except Exception as e:
        print(f"FATAL: Failed to initialize local embedding model. Error: {e}")
        return None

# Global variable to hold the model instance
LOCAL_EMBEDDING_MODEL = load_local_embedding_model(LOCAL_MODEL_ID)


# --- OpenAI Parsing Functions (NEW) ---

@st.cache_data(show_spinner=False)
def _call_openai_api(system_instruction, text_input, schema):
    if not OPENAI_API_KEY or OPENAI_CLIENT is None:
        return None

    # Construct the message payload
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": text_input}
    ]

    try:
        with st.spinner(f"AI Parsing ({OPENAI_MODEL})..."):
            response = OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                response_format={"type": "json_object", "schema": schema},
                temperature=0.0 # Force deterministic output
            )

            # Extract the JSON string from the response
            json_string = response.choices[0].message.content.strip()

            # Safely parse the JSON string
            parsed_data = json.loads(json_string)
            return parsed_data

    except APIError as e:
        st.error(f"OpenAI API Error: {e.status_code}. Details: {e.message}")
        return None
    except json.JSONDecodeError:
        st.error("OpenAI API Error: Received invalid JSON. Model may have failed to produce valid output.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during parsing: {e}")
        return None


@st.cache_data(show_spinner=False)
def parse_resume_with_ai(resume_text, selected_role):
    if not OPENAI_API_KEY or OPENAI_CLIENT is None: return None

    role_info = ROLE_SPECIFIC_SCHEMAS.get(selected_role)

    combined_properties = UNIVERSAL_RESUME_PROPS.copy()
    if role_info and "base" in role_info:
        combined_properties.update(role_info["base"])

    field_order = role_info["fields"] if role_info else list(UNIVERSAL_RESUME_PROPS.keys())

    # Build the final schema structure required by OpenAI
    final_schema = {
        "type": "object", # Must be lowercase 'object' for OpenAI
        "properties": combined_properties,
        "required": list(combined_properties.keys()) # For best JSON compliance
    }
    # Note: 'propertyOrdering' is specific to Gemini, not directly supported in OpenAI schema.

    system_instruction = (
        f"You are an expert resume parser for the '{selected_role}' field. "
        f"Extract the requested data from the resume strictly into the provided JSON format. "
        f"If a specific field is not found or is empty, return 'N/A' for string fields and empty array ([]) for list fields. "
        f"The user has provided the entire resume text. Do not add any extra commentary outside the JSON."
    )
    text_input = f"Analyze this resume for the role of '{selected_role}':\n\n{resume_text}"

    return _call_openai_api(system_instruction, text_input, final_schema)


@st.cache_data(show_spinner=False)
def parse_jd_with_ai(jd_text, selected_role):
    if not OPENAI_API_KEY or OPENAI_CLIENT is None: return None

    # Use the JD_SCHEMA directly (adjusting type case)
    final_schema = JD_SCHEMA.copy()
    final_schema["type"] = "object" # Must be lowercase 'object' for OpenAI

    system_instruction = (
        f"You are an expert job description parser for the '{selected_role}' field. "
        f"Analyze the JD and extract all required fields strictly into the provided JSON format. "
        f"If a field is not explicitly mentioned, return 'N/A' for that specific field. "
        f"The user has provided the entire Job Description text. Do not add any extra commentary outside the JSON."
    )
    text_input = f"Analyze this Job Description for the role of '{selected_role}':\n\n{jd_text}"

    return _call_openai_api(system_instruction, text_input, final_schema)


# --- Stop Words (Unchanged) ---
# ... (STOP_WORDS definition is unchanged)
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'to', 'was', 'with',
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'this', 'that', 'were',
    'have', 'had', 'do', 'does', 'did', 'but', 'so', 'if', 'then', 'than', 'up', 'down',
    'out', 'will', 'would', 'can', 'could', 'should', 'using', 'tools', 'responsibilities',
    'etc', 'e.g', 'i.e', 'also', 'just', 'only', 'may', 'must', 'could', 'which', 'who', 'when'
}

# --- File Extraction (Unchanged) ---
# ... (extract_text_from_pdf function is unchanged)
try:
    from pypdf import PdfReader
except ImportError:
    class PdfReader:
        def __init__(self, *args, **kwargs):
            pass
        @property
        def pages(self):
            return []

def extract_text_from_pdf(uploaded_file):
    """Extracts text content from an uploaded PDF file stream."""
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"PDF Parsing Error: {e}")
        return ""

# --- Core Analysis Logic (Unchanged) ---
# ... (clean_and_tokenize, calculate_keyword_match_score, calculate_semantic_score, run_rules_based_checks are UNCHANGED)

def clean_and_tokenize(text):
    """Converts text to lowercase and tokenizes it, filtering out stop words and non-alphanumeric tokens."""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]

def calculate_keyword_match_score(resume_tokens, jd_tokens):
    """Calculates the keyword match score (Lexical Match)."""
    jd_unique_tokens = set(jd_tokens)
    resume_unique_tokens = set(resume_tokens)

    jd_keywords = {t for t in jd_unique_tokens if len(t) > 3}
    resume_keywords = {t for t in resume_unique_tokens if len(t) > 3}

    if not jd_keywords:
        return 0, set(), jd_keywords

    common_keywords = jd_keywords.intersection(resume_keywords)

    match_score = (len(common_keywords) / len(jd_keywords)) * 100
    missing_keywords = jd_keywords - common_keywords

    return match_score, common_keywords, missing_keywords

def calculate_semantic_score(resume_text, jd_text):
    """Calculates the Semantic Match Score using the LOCALLY LOADED Sentence Transformer model."""
    if not resume_text or not jd_text or LOCAL_EMBEDDING_MODEL is None:
        return 0.0

    texts = [resume_text, jd_text]

    try:
        embeddings = LOCAL_EMBEDDING_MODEL.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        resume_embedding = embeddings[0]
        jd_embedding = embeddings[1]

        similarity = 1 - cosine(resume_embedding, jd_embedding)

        # Scale from [-1, 1] to [0, 100]
        semantic_score = (similarity + 1) / 2 * 100
        return semantic_score

    except Exception as e:
        st.error(f"Error calculating local embeddings or cosine similarity: {e}")
        return 0.0

def run_rules_based_checks(resume_text, job_description_text):
    """Performs compliance and quality checks using regex and logic."""
    issues = []
    passed = []
    bullet_points = [line.strip() for line in re.split(r'[\r\n•*-]+', resume_text) if len(line.strip()) > 10]

    # ------------------- 1. Structural/Compliance Checks -------------------
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_regex = r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4})'

    if not re.search(email_regex, resume_text) or not re.search(phone_regex, resume_text):
        issues.append("🔴 **Fatal Error: Missing Contact Info.** No clear email or phone number found. This guarantees ATS rejection.")

    standard_sections = ["summary", "experience", "education", "skills"]
    resume_lower = resume_text.lower()
    missing_sections = [sec for sec in standard_sections if not re.search(r'\b' + sec + r'\b', resume_lower)]

    if missing_sections:
        issues.append(f"🔴 **Missing Core Sections:** Missing **{', '.join([s.capitalize() for s in missing_sections])}**. Use clear, conventional headings.")
    else:
        passed.append("🟢 Found all expected core sections (Summary, Experience, Education, Skills).")

    date_patterns = [r'\d{4}', r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b', r'\d{1,2}/\d{4}']

    if not any(re.search(p, resume_text) for p in date_patterns):
        issues.append("🟡 **Missing or Inconsistent Dates:** Could not find common date patterns. Ensure clear start/end dates for all experience to allow tenure validation.")
    else:
        passed.append("🟢 Detected employment date patterns for proper tenure tracking.")

    # ------------------- 2. Writing Quality Checks -------------------
    strong_verbs = {'achieved', 'analyzed', 'developed', 'designed', 'executed', 'led', 'managed', 'optimized', 'reduced', 'revamped', 'solved', 'spearheaded', 'improved', 'created', 'generated', 'initiated', 'mentored'}
    weak_starters = ['responsible for', 'duties included', 'worked on', 'tasked with', 'i was']
    weak_starter_count = 0
    strong_starter_count = 0

    for line in bullet_points:
        line_lower = line.lower()
        if any(line_lower.startswith(phrase) for phrase in weak_starters):
            weak_starter_count += 1
        first_word = line_lower.split()[0] if line_lower else ''
        if first_word in strong_verbs:
              strong_starter_count += 1

    if weak_starter_count > 0:
        issues.append(f"🟡 **Passive Language:** Found **{weak_starter_count}** bullets starting with weak phrases. **Suggestion:** Begin every bullet with a strong action verb.")
    else:
        passed.append(f"🟢 Good Action Verb Usage: Found {strong_starter_count} statements starting with strong verbs.")

    quantifier_regex = r'(\d+[\.\,]?\d*\s?|[\$€£¥\%])'
    quantified_bullets = sum(1 for line in bullet_points if re.search(quantifier_regex, line))
    quantification_rate = (quantified_bullets / len(bullet_points)) if bullet_points else 0

    if quantification_rate < 0.4:
        issues.append(f"🔴 **Low Quantification Rate:** Only {quantified_bullets}/{len(bullet_points)} bullets contain measurable results. **Suggestion:** Quantify your impact with metrics, percentages, or dollar amounts.")
    else:
        passed.append(f"🟢 High Impact: {quantified_bullets}/{len(bullet_points)} statements contain measurable results.")

    return issues, passed, quantification_rate, strong_starter_count


# --- Streamlit Application Structure (Minimal changes for API Key display) ---

def app():
    # Initialize session state for parsed data
    if 'parsed_data' not in st.session_state:
        st.session_state.parsed_data = None
    if 'parsed_jd' not in st.session_state:
        st.session_state.parsed_jd = None
    if 'show_analysis' not in st.session_state:
           st.session_state.show_analysis = False

    st.set_page_config(
        layout="wide",
        page_title="ATS AI Resume Analyzer",
        page_icon="🤖"
    )

    st.title("🤖 Advanced ATS Resume & JD Matcher (Manual Parsing)")
    st.markdown(f"_AI parsing runs on demand using **{OPENAI_MODEL}** from OpenAI. Local model: **{LOCAL_MODEL_ID}**_")

    if not OPENAI_API_KEY:
           st.error("🚨 OPENAI_API_KEY NOT FOUND. AI features (parsing/extraction) are disabled. Please set your key.")

    if LOCAL_EMBEDDING_MODEL is None:
           st.warning("⚠️ **Local Model Failed to Load.** Semantic analysis is disabled. Check logs for missing dependencies (torch, sentence-transformers).")

    col_file, col_jd = st.columns([1, 1])

    # --- INPUTS (Resume Content) ---
    resume_text = ""

    with col_file:
        st.subheader("1. Upload/Paste Resume Content")
        uploaded_file = st.file_uploader("Upload PDF or TXT File", type=["pdf", "txt"], help="Uploads are processed locally for text extraction.")

        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.lower().endswith('.txt'):
                resume_text = uploaded_file.read().decode("utf-8")

        if not resume_text:
            resume_text = st.text_area("Resume Content", "", height=300, key="resume_display_paste")
        else:
            resume_text = st.text_area("Extracted Resume Content (Editable)", resume_text, height=300, key="resume_display_edit")


    # --- INPUTS (JD & Role Selection) ---
    with col_jd:
        st.subheader("2. Select Job Role & Paste JD")

        selected_domain = st.selectbox(
            "Select the **Industry Domain**:",
            options=DOMAIN_OPTIONS,
            index=0,
            key="domain_select"
        )

        available_roles = list(HIERARCHICAL_JOB_KEYWORDS.get(selected_domain, {}).keys())

        selected_role = st.selectbox(
            "Select the **Specific Job Role**:",
            options=available_roles,
            index=0 if available_roles else 0,
            key="role_select"
        )

        jd_text = st.text_area("Job Description Content", height=320, key="jd_input",
                                     placeholder=f"Paste the full job description for a {selected_role} role here...")

    st.divider()

    # --- ANALYSIS & PARSING CONTROLS ---

    parse_resume_col, parse_jd_col, run_col = st.columns([1, 1, 1.5])

    with parse_resume_col:
        if st.button(f"🧠 Extract Resume (AI)", type="secondary", use_container_width=True, key="parse_resume_button"):
            if not resume_text:
                st.error("Please provide resume content to extract fields.")
            elif not OPENAI_API_KEY:
                st.error("OpenAI API Key is required for parsing.")
            else:
                st.session_state.parsed_data = parse_resume_with_ai(resume_text, selected_role)
                st.session_state.show_analysis = False # Reset report view

    with parse_jd_col:
        if st.button(f"🔎 Extract JD Info (AI)", type="secondary", use_container_width=True, key="parse_jd_button"):
            if not jd_text:
                st.error("Please paste Job Description content to extract info.")
            elif not OPENAI_API_KEY:
                st.error("OpenAI API Key is required for parsing.")
            else:
                st.session_state.parsed_jd = parse_jd_with_ai(jd_text, selected_role)
                st.session_state.show_analysis = False # Reset report view

    # --- MAIN SCORING BUTTON ---
    with run_col:
        if st.button("Run Full ATS Analysis", type="primary", use_container_width=True):
            if not resume_text or not jd_text:
                st.error("🛑 Please ensure you have pasted the resume and job description content before running the analysis.")
                return
            st.session_state.show_analysis = True

    # --- FINAL REPORT DISPLAY (Controlled by Button - UNCHANGED) ---

    if st.session_state.show_analysis and resume_text and jd_text:

        with st.spinner('Running multi-layer ATS analysis...'):

            # --- Core Calculations ---
            resume_tokens = clean_and_tokenize(resume_text)
            jd_tokens = clean_and_tokenize(jd_text)

            lexical_score, common_keywords, missing_keywords = calculate_keyword_match_score(resume_tokens, jd_tokens)
            semantic_score = calculate_semantic_score(resume_text, jd_text)
            issues, passed_checks, quantification_rate, strong_starter_count = run_rules_based_checks(resume_text, jd_text)

            # --- START DYNAMIC KEYWORD BOOSTER LOGIC ---
            tech_keywords_for_role = HIERARCHICAL_JOB_KEYWORDS[selected_domain][selected_role].copy()

            if st.session_state.parsed_data:
                extracted_skills_list = st.session_state.parsed_data.get('Relevant_Skills', [])
                extracted_certs_list = st.session_state.parsed_data.get('Certifications_and_Degrees', [])

                all_extracted_keywords = set()
                for item in extracted_skills_list:
                    all_extracted_keywords.update(clean_and_tokenize(item))
                for item in extracted_certs_list:
                      all_extracted_keywords.update(clean_and_tokenize(item))

                tech_keywords_for_role.update(all_extracted_keywords)
            # --- END DYNAMIC KEYWORD BOOSTER LOGIC ---

            # --- Final Filtering ---
            missing_tech_keywords = missing_keywords.intersection(tech_keywords_for_role)
            matching_tech_keywords = common_keywords.intersection(tech_keywords_for_role)

            # --- Overall Score (Weighted Average) ---
            w_semantic = 0.40
            w_lexical = 0.30
            w_compliance = 0.30
            compliance_score = max(0, 100 - (len(issues) * 15))

            overall_score = (
                (semantic_score * w_semantic / 100) +
                (lexical_score * w_lexical / 100) +
                (compliance_score * w_compliance / 100)
            ) * 100

            st.header("✨ Comprehensive ATS Match Report")

            # --- SCORE CARDS & DONUT CHART (Professional Dashboard) ---
            st.subheader("🎯 Overall Fit & Score Breakdown")

            score_contribution_data = pd.DataFrame({
                'Metric': ['Semantic Match', 'Lexical Match', 'Compliance Rating'],
                'Contribution': [
                    semantic_score * w_semantic,
                    lexical_score * w_lexical,
                    compliance_score * w_compliance
                ]
            })

            base = alt.Chart(score_contribution_data).encode(
                theta=alt.Theta("Contribution", stack=True)
            )

            pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
                color=alt.Color("Metric", scale=alt.Scale(range=['#0078D4', '#6BBF5E', '#FFC300'])),
                order=alt.Order("Contribution", sort="descending"),
                tooltip=["Metric", alt.Tooltip("Contribution", format=".1f")]
            ).properties(height=300)


            chart_col, metric_col = st.columns([1, 1])

            with chart_col:
                st.altair_chart(pie, use_container_width=True)

            with metric_col:
                st.metric("Overall Match Score", f"**{overall_score:.1f}%**",
                          help="Weighted average of Semantic, Lexical, and Compliance scores (40/30/30).")

                sem_col, lex_col, comp_col = st.columns(3)
                with sem_col:
                    st.metric("Semantic Match", f"{semantic_score:.1f}%")
                with lex_col:
                    st.metric("Lexical Match", f"{lexical_score:.1f}%")
                with comp_col:
                    st.metric("Compliance Rating", f"{compliance_score:.0f}/100")

            st.markdown("---")

            # --- QUALITY INSIGHTS & VISUALS (WITHOUT BAR CHART) ---
            st.subheader("📊 Quality Indicators & Skill Gap Analysis")

            insight_col1, insight_col2 = st.columns(2)

            with insight_col1:
                st.markdown("#### Writing Quality Metrics")
                st.metric("Quantification Rate", f"{quantification_rate * 100:.0f}%",
                                 help="Percentage of experience bullets containing metrics ($, %, numbers). Target is 40% or higher.")
                st.info(f"Strong Action Vers Found: **{strong_starter_count}** statements.")


            with insight_col2:
                st.markdown(f"#### Unfiltered Keyword Match")
                st.metric("Unfiltered Match Count", f"{len(common_keywords)}",
                                 help="Total number of unique keywords matched from the entire Job Description.")
                st.info(f"Total Unique Keywords in JD: **{len(missing_keywords) + len(common_keywords)}**")

            st.markdown("---")

            # --- DETAILED FEEDBACK ---
            st.subheader("3. Detailed Improvement Suggestions (Action Items)")

            issue_cols = st.columns([1, 1])

            with issue_cols[0]:
                st.markdown("#### 🛑 Critical Issues & Formatting Warnings")
                if issues:
                    for issue in issues:
                        st.markdown(f"- {issue}")
                else:
                    st.success("🎉 No critical formatting or compliance issues detected.")

            with issue_cols[1]:
                st.markdown(f"#### Missing Required **{selected_role}** Skills")

                if missing_tech_keywords:
                    st.error(f"🔴 Missing **{len(missing_tech_keywords)}** key technical/certification terms for **{selected_role}**.")
                    with st.expander("Show Top Missing Technical Keywords"):
                        st.code(', '.join(sorted(list(missing_tech_keywords))[:30]), language="text")
                else:
                    st.success(f"✅ All major technical keywords for **{selected_role}** were found in your resume.")


            st.markdown("#### Confirmed Resume Strengths (Passed Checks)")
            if passed_checks:
                for check in passed_checks:
                    st.markdown(f"- {check}")

            st.markdown("---")

            # --- SECTION 4: TOP MATCHING TECH KEYWORDS ---
            st.subheader(f"4. Top Matching Keywords (Your **{selected_role}** Strengths)")
            if matching_tech_keywords:
                st.code(', '.join(sorted(list(matching_tech_keywords))[:30]), language="text")
            else:
                st.info(f"No specific technical or framework keywords for **{selected_role}** were matched.")

    # --- END SCORING/ANALYSIS BLOCK ---

    # --- DISPLAY PARSED DATA SECTIONS (Visible when extracted - UNCHANGED) ---

    if st.session_state.parsed_jd:
        st.markdown("---")
        jd_data = st.session_state.parsed_jd

        st.subheader(f"🔍 Extracted JD Requirements for: {jd_data.get('Job_Title_Extracted', selected_role)}")

        st.markdown("#### Core Responsibilities Summary")
        resp_summary = jd_data.get('Core_Responsibilities_Summary', 'N/A')
        st.info(resp_summary)

        skill_req_col, exp_req_col = st.columns([1.5, 1])

        with skill_req_col:
            st.markdown("#### Must-Have Skills & Certifications")
            skills = jd_data.get('Must_Have_Skills', [])
            certs = jd_data.get('Preferred_Certifications', [])
            all_requirements = skills + certs

            if all_requirements:
                st.code('\n'.join(all_requirements), language='text')
            else:
                st.warning("N/A: No explicit required skills extracted.")

        with exp_req_col:
            st.markdown("#### Minimum Requirements")
            exp_req = jd_data.get('Min_Years_Experience', 'N/A')
            min_edu = jd_data.get('Minimum_Education_Level', 'N/A')

            st.metric(label="Min Years Experience", value=exp_req)
            st.metric(label="Min Education Level", value=min_edu)

        st.markdown("---")

    if st.session_state.parsed_data:
        st.markdown("---")
        st.subheader(f"✅ Extracted Resume Fields for: {selected_role}")

        data = st.session_state.parsed_data

        role_info = ROLE_SPECIFIC_SCHEMAS.get(selected_role)
        field_order = role_info["fields"] if role_info else list(UNIVERSAL_RESUME_PROPS.keys())

        header_map = {
            "Relevant_Skills": "Relevant Skills",
            "Certifications_and_Degrees": "Certifications/Degrees",
            "Key_Projects": "Key Projects",
            "Key_Achievements": "Key Achievements",
            "Most_Relevant_Experience_Summary": "Most Relevant Experience Summary",
            "Teaching_Training_Summary": "Teaching/Training Summary",
            "Research_and_Publications": "Research and Publications"
        }

        summary_field = next((f for f in field_order if 'Summary' in f or 'Experience' in f or 'Training' in f), None)
        if summary_field:
            st.markdown(f"#### {header_map.get(summary_field, summary_field)}")
            st.info(data.get(summary_field, 'N/A'))

        col1, col2 = st.columns(2)
        field_index = 0

        for field in field_order:
            if field == summary_field:
                continue

            display_header = header_map.get(field, field.replace('_', ' '))
            content = data.get(field)

            if content is None or (isinstance(content, (list, str)) and not content):
                content = "N/A"

            with (col1 if field_index % 2 == 0 else col2):
                if isinstance(content, list) and content != "N/A":
                    st.code('\n'.join(content), language='text', label=display_header)
                elif content == "N/A":
                    st.warning(f"{display_header}: N/A")
                else:
                    st.text_area(display_header, content, height=100)
            field_index += 1


if __name__ == "__main__":
    app()