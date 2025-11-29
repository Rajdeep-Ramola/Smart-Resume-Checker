import streamlit as st
import re
import pandas as pd
import altair as alt
import numpy as np
from scipy.spatial.distance import cosine
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv() 

# --- Configuration & Constants ---
LOCAL_MODEL_ID = "BAAI/bge-small-en-v1.5"

# FIX: Updated to the currently supported Groq model
# Alternatives: "llama-3.1-8b-instant" (Faster) or "llama-3.3-70b-versatile" (Better Quality)
GROQ_MODEL = "llama-3.3-70b-versatile" 

try:
    # Get Groq API Key
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = ""

# --- PYDANTIC SCHEMAS (STRUCTURED OUTPUT) ---

class ResumeExtraction(BaseModel):
    """Schema for extracting resume information."""
    # Universal Fields
    professional_summary: str = Field(description="A concise summary of the candidate's background.", default="N/A")
    career_objective: str = Field(description="The candidate's career goals.", default="N/A")
    experience_summary: str = Field(description="A summarized paragraph of work history.", default="N/A")
    education: List[str] = Field(description="List of degrees and schools.", default=[])
    skills: List[str] = Field(description="List of technical and soft skills.", default=[])
    certifications: List[str] = Field(description="List of certifications and licenses.", default=[])
    
    # Role-Specific / Optional Fields
    key_projects: List[str] = Field(description="List of key projects (Tech/Data roles).", default=[])
    key_achievements: List[str] = Field(description="List of quantifiable achievements (Business/Sales/Finance).", default=[])
    research_publications: List[str] = Field(description="List of publications (Academic roles).", default=[])
    teaching_philosophy: str = Field(description="Teaching philosophy (Education roles).", default="N/A")

class JDExtraction(BaseModel):
    """Schema for extracting Job Description requirements."""
    job_title: str = Field(description="The exact job title.", default="N/A")
    responsibilities_summary: str = Field(description="Summary of core duties.", default="N/A")
    required_skills: List[str] = Field(description="List of mandatory skills.", default=[])
    min_experience: str = Field(description="Minimum years of experience.", default="N/A")
    min_education: str = Field(description="Minimum education level.", default="N/A")
    preferred_certifications: List[str] = Field(description="List of preferred certifications.", default=[])

# --- HIERARCHICAL JOB-SPECIFIC KEYWORDS (Unchanged) ---
HIERARCHICAL_JOB_KEYWORDS = {
    "Technology & Engineering": {
        "Software Engineer": {'python', 'java', 'c++', 'oop', 'system_design', 'api', 'git', 'sql', 'docker', 'kubernetes', 'aws', 'testing'},
        "Cybersecurity Analyst": {'siem', 'soc', 'firewall', 'incident_response', 'palo_alto', 'cissp', 'security', 'vulnerability', 'penetration'},
        "Cloud Architect": {'aws', 'azure', 'gcp', 'terraform', 'kubernetes', 'docker', 'ci/cd', 'iac', 'vpc', 'ec2'},
        "Web Developer": {'javascript', 'typescript', 'react', 'angular', 'vue', 'html', 'css', 'node', 'express', 'django'}
    },
    "Data & AI": {
        "Data Scientist": {'python', 'r', 'sql', 'machine_learning', 'statistics', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'},
        "Data Engineer": {'python', 'sql', 'spark', 'hadoop', 'kafka', 'airflow', 'databricks', 'snowflake', 'bigquery', 'etl'}
    },
    "Finance & Accounting": {
        "Financial Analyst": {'excel', 'financial_modeling', 'forecasting', 'valuation', 'sql', 'tableau', 'power_bi', 'accounting'},
        "Accountant/Auditor": {'gaap', 'reconciliation', 'general_ledger', 'tax', 'audit', 'quickbooks', 'excel', 'compliance', 'sox'}
    },
    "Education Sector": {
        "Elementary School Teacher": {'classroom_management', 'common_core', 'phonics', 'curriculum_development', 'differentiation', 'parent_communication'},
        "University Professor": {'research', 'publication', 'grant_writing', 'lecturing', 'curriculum_development', 'mentoring', 'phd'}
    },
    "Business & Management": {
        "Project Manager (IT/Tech)": {'pmp', 'agile', 'scrum', 'waterfall', 'jira', 'risk_management', 'stakeholder', 'budgeting'},
        "Business Analyst": {'sql', 'requirements_gathering', 'process_mapping', 'visio', 'agile', 'user_stories', 'testing', 'uml'}
    },
    "Marketing & Sales": {
        "Digital Marketing Manager": {'seo', 'sem', 'google_analytics', 'hubspot', 'content_strategy', 'campaigns', 'crm', 'conversion_rate'},
        "Sales Specialist": {'crm', 'salesforce', 'pipeline', 'prospecting', 'b2b', 'negotiation', 'territory_management'}
    }
}
DOMAIN_OPTIONS = list(HIERARCHICAL_JOB_KEYWORDS.keys())

# --- Load Model Once ---
@st.cache_resource(show_spinner=f"Downloading {LOCAL_MODEL_ID}...")
def load_local_embedding_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
        CACHE_DIR = os.path.join(os.getcwd(), ".model_cache") 
        os.makedirs(CACHE_DIR, exist_ok=True)
        return SentenceTransformer(model_name, cache_folder=CACHE_DIR)
    except Exception as e:
        print(f"FATAL: Failed to initialize local embedding model. Error: {e}")
        return None

LOCAL_EMBEDDING_MODEL = load_local_embedding_model(LOCAL_MODEL_ID)

# --- LangChain Parsing Functions (Using Groq) ---

def get_llm():
    if not GROQ_API_KEY:
        return None
    # Initialize ChatGroq with the NEW model name
    return ChatGroq(
        model_name=GROQ_MODEL, 
        api_key=GROQ_API_KEY, 
        temperature=0
    )

@st.cache_data(show_spinner=False)
def parse_resume_with_langchain(resume_text, selected_role):
    llm = get_llm()
    if not llm: return None

    # Bind the Pydantic model to Groq for structured output
    structured_llm = llm.with_structured_output(ResumeExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert technical recruiter specializing in {selected_role} roles. Extract information strictly into the requested format."),
        ("human", "Resume Content:\n{text}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        with st.spinner("AI Extracting Resume Data..."):
            return chain.invoke({"text": resume_text})
    except Exception as e:
        st.error(f"LangChain/Groq Error: {e}")
        return None

@st.cache_data(show_spinner=False)
def parse_jd_with_langchain(jd_text, selected_role):
    llm = get_llm()
    if not llm: return None

    structured_llm = llm.with_structured_output(JDExtraction)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert technical recruiter. Extract key requirements for a {selected_role} position."),
        ("human", "Job Description:\n{text}")
    ])

    chain = prompt | structured_llm
    
    try:
        with st.spinner("AI Extracting JD Data..."):
            return chain.invoke({"text": jd_text})
    except Exception as e:
        st.error(f"LangChain/Groq Error: {e}")
        return None

# --- Gap Analysis ---
def check_critical_gaps(resume_data: ResumeExtraction, jd_data: JDExtraction):
    gaps = []
    
    # 1. Check Experience
    if jd_data.min_experience != "N/A":
        if resume_data.experience_summary == "N/A" and not resume_data.key_achievements:
            gaps.append(f"⚠️ **Missing Experience:** JD requires '{jd_data.min_experience}', but no experience summary found.")

    # 2. Check Education
    if jd_data.min_education != "N/A":
        if not resume_data.education:
             gaps.append(f"⚠️ **Missing Education:** JD requires '{jd_data.min_education}', but education section is empty.")
    
    # 3. Check Skills
    if jd_data.required_skills:
        if not resume_data.skills:
             gaps.append("⚠️ **Missing Skills:** JD lists required skills, but resume skills section is empty.")
             
    # 4. Check Certifications (if strictly required)
    if jd_data.preferred_certifications:
        if not resume_data.certifications:
             gaps.append("⚠️ **Missing Certifications:** JD lists preferred certifications, but none found in resume.")

    return gaps

# --- Core Logic ---
try:
    from pypdf import PdfReader
except ImportError:
    class PdfReader:
        def __init__(self, *args, **kwargs): pass
        @property
        def pages(self): return []

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
        return text
    except Exception: return ""

STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to'}
def clean_and_tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]

def calculate_keyword_match_score(resume_tokens, jd_tokens):
    jd_unique_tokens = set(jd_tokens)
    resume_unique_tokens = set(resume_tokens)
    jd_keywords = {t for t in jd_unique_tokens if len(t) > 3} 
    resume_keywords = {t for t in resume_unique_tokens if len(t) > 3}
    if not jd_keywords: return 0, set(), jd_keywords
    common_keywords = jd_keywords.intersection(resume_keywords)
    match_score = (len(common_keywords) / len(jd_keywords)) * 100
    missing_keywords = jd_keywords - common_keywords
    return match_score, common_keywords, missing_keywords

def calculate_semantic_score(resume_text, jd_text):
    if not resume_text or not jd_text or LOCAL_EMBEDDING_MODEL is None: return 0.0
    try:
        embeddings = LOCAL_EMBEDDING_MODEL.encode([resume_text, jd_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        return (similarity + 1) / 2 * 100 
    except: return 0.0

def run_rules_based_checks(resume_text, jd_text):
    issues = []
    passed = []
    bullet_points = [line.strip() for line in re.split(r'[\r\n•*-]+', resume_text) if len(line.strip()) > 10]
    
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if not re.search(email_regex, resume_text):
        issues.append("🔴 **Fatal Error: Missing Contact Info.** No clear email found.")
    
    strong_verbs = {'achieved', 'analyzed', 'developed', 'managed', 'created'} 
    strong_starter_count = 0
    for line in bullet_points:
        if line.lower().split()[0] in strong_verbs: strong_starter_count += 1
            
    quantifier_regex = r'(\d+[\.\,]?\d*\s?|[\$€£¥\%])' 
    quantified_bullets = sum(1 for line in bullet_points if re.search(quantifier_regex, line))
    quantification_rate = (quantified_bullets / len(bullet_points)) if bullet_points else 0
    
    if quantification_rate < 0.4:
        issues.append(f"🔴 **Low Quantification Rate:** Only {quantified_bullets} bullets contain numbers/metrics.")
    else:
        passed.append("🟢 High Impact: Good use of quantifiable metrics.")
        
    return issues, passed, quantification_rate, strong_starter_count

# --- Main App ---

def app():
    if 'resume_data' not in st.session_state: st.session_state.resume_data = None
    if 'jd_data' not in st.session_state: st.session_state.jd_data = None
    if 'show_analysis' not in st.session_state: st.session_state.show_analysis = False
        
    st.set_page_config(layout="wide", page_title="ATS AI Resume Analyzer", page_icon="🤖")
    
    st.title("🤖 Advanced ATS Resume & JD Matcher")
    st.markdown(f"_Powered by **LangChain (Groq/Llama3)** & **{LOCAL_MODEL_ID}**_")
    
    if not GROQ_API_KEY: st.error("🚨 GROQ_API_KEY NOT FOUND.")
    if LOCAL_EMBEDDING_MODEL is None: st.warning("⚠️ Local Model Failed to Load.")

    col_file, col_jd = st.columns([1, 1])
    
    resume_text = ""
    with col_file:
        st.subheader("1. Resume")
        uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
        if uploaded_file:
            if uploaded_file.name.lower().endswith('.pdf'): resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.lower().endswith('.txt'): resume_text = uploaded_file.read().decode("utf-8")
        if not resume_text: resume_text = st.text_area("Paste Resume", height=300)
        else: st.text_area("Extracted Resume", resume_text, height=300)

    with col_jd:
        st.subheader("2. Job Description")
        selected_domain = st.selectbox("Industry Domain", DOMAIN_OPTIONS, index=0)
        available_roles = list(HIERARCHICAL_JOB_KEYWORDS.get(selected_domain, {}).keys())
        selected_role = st.selectbox("Target Role", available_roles, index=0)
        jd_text = st.text_area("Paste JD", height=300)
    
    st.divider()
    
    # --- Buttons ---
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🧠 Extract Resume Info"):
            if resume_text: 
                st.session_state.resume_data = parse_resume_with_langchain(resume_text, selected_role)
                st.session_state.show_analysis = False
    with c2:
        if st.button("🔎 Extract JD Info"):
            if jd_text:
                st.session_state.jd_data = parse_jd_with_langchain(jd_text, selected_role)
                st.session_state.show_analysis = False
    with c3:
        if st.button("🚀 Run Full ATS Analysis", type="primary"):
            if resume_text and jd_text: st.session_state.show_analysis = True

    # --- Gap Analysis (Auto) ---
    if st.session_state.jd_data and st.session_state.resume_data:
        st.markdown("---")
        st.subheader("⚠️ Critical Gap Analysis")
        gaps = check_critical_gaps(st.session_state.resume_data, st.session_state.jd_data)
        if gaps:
            for gap in gaps: st.error(gap)
        else:
            st.success("✅ No critical section gaps found!")

    # --- Display Results ---
    if st.session_state.jd_data:
        st.markdown("---")
        st.subheader("📋 Extracted JD Requirements")
        jd = st.session_state.jd_data
        c1, c2 = st.columns([2, 1])
        with c1:
            st.info(f"**Title:** {jd.job_title}\n\n**Summary:** {jd.responsibilities_summary}")
            st.markdown("**Required Skills:**")
            st.code(", ".join(jd.required_skills), language="text")
        with c2:
            st.metric("Min Experience", jd.min_experience)
            st.metric("Education", jd.min_education)
            if jd.preferred_certifications:
                st.markdown("**Preferred Certs:**")
                for c in jd.preferred_certifications: st.write(f"- {c}")

    if st.session_state.resume_data:
        st.markdown("---")
        st.subheader("👤 Extracted Resume Data")
        res = st.session_state.resume_data
        
        # Universal
        st.text_area("Professional Summary", res.professional_summary, height=100)
        st.text_area("Experience Summary", res.experience_summary, height=100)
        
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Skills:**")
            st.code(", ".join(res.skills), language="text")
            st.markdown("**Education:**")
            for edu in res.education: st.markdown(f"- {edu}")
        with rc2:
            st.markdown("**Certifications:**")
            for c in res.certifications: st.markdown(f"- {c}")
            st.markdown("**Objective:**")
            st.write(res.career_objective)
            
        # Role Specific Display
        st.markdown("#### Role-Specific Details")
        rpc1, rpc2 = st.columns(2)
        with rpc1:
            if res.key_projects:
                st.markdown("**Key Projects:**")
                for p in res.key_projects: st.markdown(f"- {p}")
            if res.research_publications:
                st.markdown("**Research / Publications:**")
                for p in res.research_publications: st.markdown(f"- {p}")
        with rpc2:
            if res.key_achievements:
                st.markdown("**Key Achievements:**")
                for a in res.key_achievements: st.markdown(f"- {a}")
            if res.teaching_philosophy != "N/A":
                st.markdown("**Teaching Philosophy:**")
                st.info(res.teaching_philosophy)

    # --- Full Analysis ---
    if st.session_state.show_analysis and resume_text and jd_text:
        st.markdown("---")
        st.header("✨ ATS Match Report")
        
        with st.spinner('Analyzing...'):
            resume_tokens = clean_and_tokenize(resume_text)
            jd_tokens = clean_and_tokenize(jd_text)
            
            # KEYWORD UPDATE: Merge Base Keywords + Extracted JD Skills
            tech_keywords = set(HIERARCHICAL_JOB_KEYWORDS.get(selected_domain, {}).get(selected_role, []))
            
            if st.session_state.jd_data and st.session_state.jd_data.required_skills:
                 # Dynamic boosting from JD analysis
                 for skill in st.session_state.jd_data.required_skills:
                     tech_keywords.update(clean_and_tokenize(skill))
            
            lexical_score, common_kw, missing_kw = calculate_keyword_match_score(resume_tokens, jd_tokens)
            semantic_score = calculate_semantic_score(resume_text, jd_text)
            issues, passed, quant_rate, strong_v = run_rules_based_checks(resume_text, jd_text)
            
            missing_tech = missing_kw.intersection(tech_keywords)
            matching_tech = common_kw.intersection(tech_keywords)

            w_sem, w_lex, w_comp = 0.4, 0.3, 0.3
            comp_score = max(0, 100 - (len(issues) * 15))
            overall = (semantic_score * w_sem) + (lexical_score * w_lex) + (comp_score * w_comp)
            
            # Display
            mc1, mc2 = st.columns([1, 1])
            source = pd.DataFrame({
                "Category": ["Semantic", "Lexical", "Compliance"],
                "Score": [semantic_score, lexical_score, comp_score]
            })
            base = alt.Chart(source).encode(theta=alt.Theta("Score", stack=True))
            pie = base.mark_arc(outerRadius=100).encode(color=alt.Color("Category"), tooltip=["Category", "Score"])
            text = base.mark_text(radius=120).encode(text=alt.Text("Score", format=".1f"), color=alt.value("black"))
            
            with mc1: st.altair_chart(pie + text, use_container_width=True)
            with mc2:
                st.metric("Overall Match", f"{overall:.1f}%")
                st.metric("Semantic", f"{semantic_score:.1f}%")
                st.metric("Lexical", f"{lexical_score:.1f}%")
                st.metric("Compliance", f"{comp_score}/100")
                
            c1, c2 = st.columns(2)
            with c1:
                st.error(f"Missing Keywords ({len(missing_tech)})")
                st.write(", ".join(list(missing_tech)) if missing_tech else "None")
            with c2:
                st.success(f"Matching Keywords ({len(matching_tech)})")
                st.write(", ".join(list(matching_tech)) if matching_tech else "None")

            if issues:
                st.subheader("Compliance Issues")
                for i in issues: st.write(i)

if __name__ == "__main__":
    app()