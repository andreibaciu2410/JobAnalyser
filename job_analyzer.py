import streamlit as st
import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator
import instructor
from groq import Groq
from dotenv import load_dotenv

# ==============================================================================
# 1. SETUP & SECURITATE
# ==============================================================================
st.set_page_config(page_title="GenAI Headhunter", page_icon="🕵️", layout="wide")

# Încărcăm variabilele din fișierul .env
load_dotenv()

# Încercăm să luăm cheia din OS (local) sau din Streamlit Secrets (cloud)
api_key = os.getenv("GROQ_API_KEY")

# Fallback pentru Streamlit Cloud deployment
if not api_key and "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

# Validare critică: Dacă nu avem cheie, oprim aplicația aici.
if not api_key:
    st.error("⛔ EROARE CRITICĂ: Lipsește `GROQ_API_KEY`.")
    st.info("Te rog creează un fișier `.env` în folderul proiectului și adaugă: GROQ_API_KEY=cheia_ta_aici")
    st.stop()

# Configurare Client Groq Global (pentru a nu-l reinițializa constant)
client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.TOOLS)

# Sidebar Informativ (Fără input de date sensibile)
with st.sidebar:
    st.header("🕵️ GenAI Headhunter")
    st.success("✅ API Key încărcat securizat")
    st.markdown("---")
    st.write("Acest tool demonstrează:")
    st.write("• Web Scraping (BS4)")
    st.write("• Secure Env Variables")
    st.write("• Structured Data (Pydantic)")


# ==============================================================================
# 2. DATA MODELS (PYDANTIC SCHEMAS)
# ==============================================================================

class SalaryRange(BaseModel):
    min: int = Field(..., description="Salariul minim")
    max: int = Field(..., description="Salariul maxim")
    currency: str = Field(..., description="Moneda de plata")

class Location(BaseModel):
    city: Optional[str] = Field(..., description="Locatia jobului")
    country: Optional[str] = Field(..., description="Tara jobului")
    is_remote: bool = Field(False, description="Jobul este remote sau nu")

class RedFlag(BaseModel):
    severity: Literal["low", "medium", "high"] = Field(..., description="Severitatea red flag-ului")
    category: Literal["toxicity", "vague", "unrealistic"] = Field(..., description="Categoria red flag-ului")
    message: str = Field(..., description="Mesajul red flag-ului")

class RawExtraction(BaseModel):
    role_title: Optional[str] = None
    company_name: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)

    salary_range: Optional[SalaryRange] = None
    location: Optional[Location] = None
    is_remote: Optional[bool] = None  # ce reiese explicit din text

    requirements: List[str] = Field(..., description="Cerinte explicite, bullet-like")
    responsibilities: List[str] = Field(..., description="Responsabilitati explicite")
    benefits: List[str] = Field(..., description="Beneficii explicite")
    red_flags: List[RedFlag] = Field(..., description="Doar daca sunt sugerate clar de text")

    @model_validator(mode="after")
    def normalize_location(self):
        if self.location and not self.location.city and not self.location.country:
            self.location = None
        return self
    
class JobAnalysis(BaseModel):
    role_title: str = Field(..., description="Titlul jobului standardizat")
    company_name: str = Field(..., description="Numele companiei")
    seniority: Literal["Intern", "Junior", "Mid", "Senior", "Lead", "Architect", "Unknown"] = Field("Unknown", description="Nivelul de experiență dedus")
    match_score: int = Field(..., ge=0, le=100, description="Scor 0-100: Calitatea descrierii jobului")
    tech_stack: List[str] = Field(..., description="Listă cu tehnologii specifice (ex: Python, AWS, React)")
    red_flags: List[RedFlag] = Field(default_factory=list, description="Lista de semnale de alarmă (toxicitate, stres, vaguitate)")
    summary: str = Field(..., description="Un rezumat scurt al rolului (max 2 fraze) în limba română")
    is_remote: bool = Field(False, description="True dacă jobul este remote sau hibrid")
    salary_range: SalaryRange = Field(..., description="Range salarial")
    location: Location = Field(..., description="Detaliile locatiei jobului")

    @model_validator(mode="after")
    def remote_location_consistency(self):
        # sa avem consistenta intre cele 2 campuri
        if self.is_remote != self.location.is_remote:
            self.red_flags.append(
                RedFlag(
                    severity="medium",
                    category="vague",
                    message=f"Inconsistenta intre is_remote={self.is_remote} si location.is_remote={self.location.is_remote}"
                )
            )
        
        if self.is_remote:
            office_patterns = re.compile(r"\b(office|on[-\s]?site|onsite|hybrid|hibrid|birou|sediu|headquarters|hq)\b", re.IGNORECASE)
            combined = f"{self.location.city} {self.location.country}"

            if office_patterns.search(combined):
                self.red_flags.append(
                    RedFlag(
                        severity="medium",
                        category="vague",
                        message="Remote=True, dar câmpurile de locație conțin indicii de prezență la birou"
                    )
                )

        return self

class StrategicAdvice(BaseModel):
    fit_summary: str = Field(..., description="2-4 fraze: cum suna rolul si pentru cine e potrivit")
    interview_questions: List[str] = Field(..., description="Intrebari concrete pentru clarificari")
    negotiation_angles: List[str] = Field(..., description="Argumente / tactici pentru negociere")
    risk_notes: List[str] = Field(..., description="Riscuri si cum le verifici")
    next_steps: List[str] = Field(..., description="Pasi urmatori")

class ValidationIssue(BaseModel):
    field: str = Field(..., description="Calea campului din RawExtraction/JobAnalysis care are problema")
    severity: Literal["low", "medium", "high"] = Field(..., description="Cat de grava e inconsistenta")
    message: str = Field(..., description="Explicatie scurta despre de ce e problema si ce impact are")
    evidence: Optional[str] = Field(None, description="Fragment scurt din text care sustine sau contrazice")

class ValidationReport(BaseModel):
    cleaned: RawExtraction
    issues: List[ValidationIssue] = Field(default_factory=list)
    confidence: int = Field(100, ge=0, le=100, description="Cat de bine se pot verifica faptele din text")


# ==============================================================================
# 3. UTILS - SCRAPER (Colectare Date)
# ==============================================================================

def scrape_clean_job_text(url: str, max_chars: int = 3000) -> str:
    """
    Descarcă pagina și returnează un text curat, optimizat pentru contextul LLM.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Error: Status code {response.status_code}"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Eliminăm elementele inutile care consumă tokeni
        for junk in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            junk.decompose()
            
        # Extragem textul și eliminăm spațiile multiple
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        
        return text[:max_chars] 
        
    except Exception as e:
        return f"Scraping Error: {str(e)}"

# ==============================================================================
# 4. AI SERVICE LAYER (Logica LLM)
# ==============================================================================

def analyze_job_with_ai(text: str) -> JobAnalysis:
    """
    Trimite textul curățat către Groq și returnează obiectul structurat.
    """
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=JobAnalysis,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Ești un Recruiter Expert în IT. Analizează textul jobului cu obiectivitate. "
                    "Identifică tehnologiile și potențialele probleme (red flags). "
                    "Răspunde strict în formatul cerut."
                )
            },
            {
                "role": "user", 
                "content": f"Analizează acest job description:\n\n{text}"
            }
        ],
        temperature=0.1,
    )

# ==============================================================================
# 5. Agent 1: determinist, doar fapte
# ==============================================================================

def extract_job_facts(text: str) -> RawExtraction:
    return client.chat.completions.create(
        model="qwen/qwen3-32b",
        response_model=RawExtraction,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ești The Extractor. Extragi DOAR fapte brute din textul jobului. "
                    "Nu inventa. Dacă lipsește informația, lasă null / listă goală. "
                    "Nu oferi sfaturi. Nu rezuma."
                ),
            },
            {"role": "user", "content": f"Extrage faptele din acest job:\n\n{text}"},
        ],
        temperature=0.0,
    )

# ==============================================================================
# 6. Agent 2: creativ, insight + strategie
# ==============================================================================

def generate_counceling(facts: RawExtraction) -> StrategicAdvice:
    return client.chat.completions.create(
        model="qwen/qwen3-32b",
        response_model=StrategicAdvice,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ești The Counselor. Primești fapte structurate despre un job și oferi "
                    "insight-uri strategice: potrivire, întrebări de interviu, negociere salariu, riscuri."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generează advice strategic pe baza acestor fapte (JSON):\n\n"
                    f"{facts.model_dump_json(indent=2, exclude_none=True)}"
                ),
            },
        ],
        temperature=0.7,
    )

# ==============================================================================
# 7. Agent 3: The Validator
# ==============================================================================

def validate_extraction(original_text: str, facts: RawExtraction) -> ValidationReport:
    return client.chat.completions.create(
        model="llama-3.1-8b-instant",
        response_model=ValidationReport,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ești The Validator. Verifici consistența dintre textul jobului și JSON-ul RawExtraction. "
                    "Nu inventa dovezi. Dacă un câmp nu e susținut explicit, pune-l la null/empty în cleaned "
                    "și adaugă un issue. Dacă e contrazis, severity=high."
                    "\nReguli:\n"
                    "- tech_stack: doar tehnologii prezente explicit\n"
                    "- salary_range: doar dacă există cifre/monedă în text\n"
                    "- location/is_remote: trebuie să fie susținute de text\n"
                    "- requirements/benefits: trebuie să fie parafraze scurte din text\n"
                    "Returnează cleaned + issues + confidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    "TEXT JOB:\n"
                    f"{original_text}\n\n"
                    "RAW EXTRACTION (JSON):\n"
                    f"{facts.model_dump_json(indent=2, exclude_none=True)}"
                ),
            },
        ],
        temperature=0.0,
    )

# ==============================================================================
# 7. UI - APLICAȚIA STREAMLIT
# ==============================================================================

st.title("🕵️ GenAI Headhunter Assistant")
st.markdown("Transformă orice Job Description într-o analiză structurată folosind AI.")

# Tab-uri
tab1, tab2 = st.tabs(["🚀 Analiză Job", "📊 Market Scan (Batch)"])

# --- TAB 1: ANALIZA UNUI SINGUR LINK ---
with tab1:
    st.subheader("Analizează un Job URL")
    url_input = st.text_input("Introdu URL-ul:", placeholder="https://...")
    
    if st.button("Analizează Job", key="btn_single"):
        if not url_input:
            st.warning("Te rugăm introdu un URL.")
        else:
            with st.spinner("🕷️ Scraping & 🤖 AI Analysis..."):
                raw_text = scrape_clean_job_text(url_input)
            
            if "Error" in raw_text:
                st.error(raw_text)
            else:
                try:
                    data = analyze_job_with_ai(raw_text)
                    report = extract_job_facts(raw_text)
                    facts = validate_extraction(raw_text, report) 
                    advice = generate_counceling(facts.cleaned)
                    
                    #analyze_job_with_ai
                    # -- DISPLAY --
                    st.divider()
                    col_h1, col_h2 = st.columns([3, 1])
                    with col_h1:
                        st.markdown(f"### {data.role_title}")
                        st.caption(f"Companie: **{data.company_name}** | Nivel: **{data.seniority}**")
                        st.caption(f"Locatie: **{data.location.country}**, **{data.location.city}**")
                        st.caption(f"Remote: **{'Da' if data.location.is_remote else 'Nu'}**")
                        st.caption(f"Salariu: **{data.salary_range.min}** - **{data.salary_range.max}** **{data.salary_range.currency}**")
                    with col_h2:
                        color = "normal" if data.match_score > 70 else "inverse"
                        st.metric("Quality Score", f"{data.match_score}/100", delta_color=color)

                    # Detalii
                    c1, c2, c3 = st.columns(3)
                    c1.info(f"**Remote:** {'Da' if data.is_remote else 'Nu'}")
                    c2.success(f"**Tehnologii:** {len(data.tech_stack)}")
                    c3.error(f"**Red Flags:** {len(data.red_flags)}")

                    st.markdown(f"**📝 Rezumat:** {data.summary}")
                    st.markdown("#### 🛠️ Tech Stack")
                    st.write(", ".join([f"`{tech}`" for tech in data.tech_stack]))

                    if data.red_flags:
                        st.markdown("#### 🚩 Avertismente")
                        for flag in data.red_flags:
                            st.warning(f"⚠️ {flag}")

                    st.divider()
                    # extract_job_facts
                    #generate_strategic_advice
                    st.markdown("### 🧾 Facts (Extractor)")
                    st.json(facts.model_dump(exclude_none=True))

                    st.markdown("### 🧠 Advice (Counselor)")
                    st.write(advice.fit_summary)

                    with st.expander("🎤 Interview Questions"):
                        for q in advice.interview_questions:
                            st.write(f"- {q}")

                    with st.expander("💰 Negotiation Angles"):
                        for a in advice.negotiation_angles:
                            st.write(f"- {a}")

                    with st.expander("⚠️ Risks & Checks"):
                        for r in advice.risk_notes:
                            st.write(f"- {r}")

                    with st.expander("✅ Next Steps"):
                        for n in advice.next_steps:
                            st.write(f"- {n}")

                except Exception as e:
                    st.error(f"Eroare AI: {str(e)}")

# --- TAB 2: BATCH PROCESSING ---
with tab2:
    st.subheader("📊 Compară mai multe joburi")
    urls_text = st.text_area("Paste URL-uri (unul pe linie):", height=150)
    
    if st.button("Scanează Piața", key="btn_batch"):
        urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
        
        if not urls:
            st.warning("Nu ai introdus link-uri.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, link in enumerate(urls):
                status_text.text(f"Analizez {i+1}/{len(urls)}...")
                text = scrape_clean_job_text(link)
                
                if "Error" not in text:
                    try:
                        res = analyze_job_with_ai(text)
                        results.append({
                            "Role": res.role_title,
                            "Company": res.company_name,
                            "Seniority": res.seniority,
                            "Tech": res.tech_stack,
                            "Score": res.match_score
                        })
                    except:
                        pass # Continuăm chiar dacă unul crapă
                
                progress_bar.progress((i + 1) / len(urls))
            
            status_text.text("Gata!")
            
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                
                # Grafic simplu
                st.bar_chart(df['Seniority'].value_counts())