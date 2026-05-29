"""
frontend/app_hf.py
------------------
Hugging Face Spaces — single process Streamlit app.
Tab 1: Research Gap Analyser
Tab 2: Paper & Thesis Downloader (9 Journal + 13 PhD Thesis sources)
       Mobile-friendly: uses st.download_button for browser-native download
"""

import sys, re, time, json, os, requests, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any
from urllib.parse import quote
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.pdf_extractor import extract_text_from_pdf
from models.metadata_extractor import extract_metadata_from_text

st.set_page_config(page_title="AI Research Novelty & Gap Detector",
                   page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#f9a825,#fdd835,#1565c0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub-header{font-size:1rem;color:#f9a825;margin-bottom:1.2rem;font-weight:600;}
.novelty-badge-high{background:#1565c0;color:#fdd835;padding:6px 18px;border-radius:20px;font-weight:700;border:2px solid #fdd835;}
.novelty-badge-medium{background:#f9a825;color:#1a237e;padding:6px 18px;border-radius:20px;font-weight:700;}
.novelty-badge-low{background:#e74c3c;color:#fff;padding:6px 18px;border-radius:20px;font-weight:700;}
.gap-card{background:#fffde7;border-left:4px solid #f9a825;padding:10px 14px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.paper-card{background:#e3f2fd;border-left:4px solid #1565c0;padding:10px 14px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.title-suggestion{background:#fffde7;border-left:4px solid #fdd835;padding:10px 14px;border-radius:8px;margin:8px 0;font-style:italic;color:#1a1a1a!important;}
.metric-card{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:8px;margin:6px 0;color:#1a1a1a!important;}
.plag-safe{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.plag-low{background:#fffde7;border-left:4px solid #f9a825;padding:12px 16px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.plag-medium{background:#fff8e1;border-left:4px solid #fdd835;padding:12px 16px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.plag-high{background:#ffebee;border-left:4px solid #e74c3c;padding:12px 16px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.flagged-sent{background:#fffde7;border-left:4px solid #f9a825;padding:8px 12px;border-radius:6px;margin:6px 0;color:#1a1a1a!important;font-size:0.88rem;}
.phrase-match{background:#e3f2fd;border-left:4px solid #1565c0;padding:8px 12px;border-radius:6px;margin:6px 0;color:#1a1a1a!important;font-size:0.88rem;}
.hallucination-safe{background:#e3f2fd;border-left:4px solid #1565c0;padding:10px 14px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.hallucination-warn{background:#fffde7;border-left:4px solid #f9a825;padding:10px 14px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.hallucination-danger{background:#ffebee;border-left:4px solid #e74c3c;padding:10px 14px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.flagged-claim{background:#fffde7;border-left:4px solid #f9a825;padding:8px 12px;border-radius:6px;margin:6px 0;color:#1a1a1a!important;font-size:0.9rem;}
.src-badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:700;margin-right:4px;color:#fff;}
.free-badge{background:#1565c0;color:#fdd835;padding:2px 8px;border-radius:8px;font-size:0.75rem;font-weight:700;}
.link-badge{background:#f9a825;color:#1a237e;padding:2px 8px;border-radius:8px;font-size:0.75rem;font-weight:700;}
.thesis-card{background:#e3f2fd;border-left:4px solid #1565c0;padding:14px 16px;border-radius:8px;margin:8px 0;color:#1a1a1a!important;}
.thesis-tag{background:#1565c0;color:#fdd835;padding:2px 8px;border-radius:8px;font-size:0.7rem;font-weight:700;margin-left:4px;}
.india-tag{background:#f9a825;color:#1a237e;padding:2px 8px;border-radius:8px;font-size:0.7rem;font-weight:700;margin-left:4px;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SECTION A — RESEARCH GAP ANALYSER
# ═══════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Loading AI models (~2 min first time)…")
def load_engines():
    from retrieval.retriever import ResearchRetriever
    from config.settings import FAISS_INDEX_PATH
    if not Path(FAISS_INDEX_PATH).exists():
        import subprocess
        subprocess.run(["python","scripts/seed_data.py"],check=True)
        subprocess.run(["python","scripts/build_index.py"],check=True)
    return ResearchRetriever()

def run_analysis(retriever,title,abstract,keywords,domain,top_k,check_plagiarism,full_text=None):
    from models.similarity_engine import compute_similarity_stats,rank_results,find_duplicate_risk,summarise_domain_overlap,summarise_year_distribution
    from models.novelty_engine import calculate_novelty_score,get_novelty_suggestions
    from models.gap_engine import detect_research_gaps,suggest_improved_titles
    from models.explanation_engine import generate_explanation,enhance_gaps_with_llm
    from models.hallucination_detector import detect_hallucinations
    from models.plagiarism_detector import detect_plagiarism
    papers=retriever.retrieve(title=title,abstract=abstract,keywords=keywords,top_k=top_k)
    sim_stats=compute_similarity_stats(papers); ranked=rank_results(papers,top_n=top_k)
    duplicates=find_duplicate_risk(papers); domain_dist=summarise_domain_overlap(papers); year_dist=summarise_year_distribution(papers)
    novelty=calculate_novelty_score(papers,sim_stats,domain or ""); novelty["suggestions"]=get_novelty_suggestions(novelty["label"],sim_stats,papers)
    gaps=detect_research_gaps(retrieved_papers=papers,user_title=title,user_keywords=keywords)
    gaps["gap_statements"]=enhance_gaps_with_llm(gap_statements=gaps.get("gap_statements",[]),retrieved_papers=papers,user_title=title)
    title_suggestions=suggest_improved_titles(original_title=title,gap_dimensions=gaps.get("gap_dimensions",{}))
    explanation=generate_explanation({"input":{"title":title,"abstract":abstract,"keywords":keywords,"domain":domain},"novelty":novelty,"gaps":gaps,"similarity_stats":sim_stats,"similar_papers":ranked,"title_suggestions":title_suggestions})
    hallucination=detect_hallucinations(llm_output=explanation,retrieved_papers=papers,novelty_data=novelty)
    plagiarism_text = full_text if full_text else abstract
    plagiarism=detect_plagiarism(user_title=title,user_abstract=plagiarism_text,user_keywords=keywords,retrieved_papers=papers,similarity_stats=sim_stats) if check_plagiarism else {}
    return {"status":"success","input":{"title":title,"abstract":abstract,"keywords":keywords,"domain":domain,"full_text":full_text},
            "novelty":{"label":novelty["label"],"percentage":novelty["percentage"],"color":novelty["color"],"description":novelty["description"],"sub_scores":novelty.get("sub_scores",{}),"suggestions":novelty["suggestions"]},
            "similarity":{"stats":sim_stats,"domain_dist":domain_dist,"year_dist":year_dist,"duplicate_risk":len(duplicates)>0,"duplicates":duplicates},
            "similar_papers":ranked,"gaps":gaps,"title_suggestions":title_suggestions,"explanation":explanation,
            "hallucination":{"grounding_score":hallucination.get("grounding_score",100),"label":hallucination.get("label",""),"color":hallucination.get("color","#2ecc71"),"hallucination_count":hallucination.get("hallucination_count",0),"total_claims":hallucination.get("total_claims",0),"flagged_claims":hallucination.get("flagged_claims",[]),"warnings":hallucination.get("warnings",[]),"summary":hallucination.get("summary","")},
            "plagiarism":plagiarism}

def gauge(v, t, s, c):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        title={"text": t, "font": {"size": 13}},
        number={"suffix": s, "font": {"size": 28, "color": c}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": c},
            "steps": [
                {"range": [0,  40],  "color": "#ffebee"},
                {"range": [40, 70],  "color": "#fff8e1"},
                {"range": [70, 100], "color": "#e8f5e9"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=15, r=15, t=55, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def plagiarism_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "<b>Plagiarism Score</b><br><small>Lower is better</small>", "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  15],  "color": "#e8f5e9"},
                {"range": [15, 40],  "color": "#e3f2fd"},
                {"range": [40, 70],  "color": "#fff8e1"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=15, r=15, t=55, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def sim_bar(papers):
    titles=[p["title"][:50]+"…" if len(p["title"])>50 else p["title"] for p in papers]
    scores=[p["similarity_pct"] for p in papers]; colors=["#e53935" if s>=75 else "#fb8c00" if s>=55 else "#43a047" for s in scores]
    fig=go.Figure(go.Bar(x=scores,y=titles,orientation="h",marker_color=colors,text=[f"{s:.1f}%" for s in scores],textposition="outside",textfont=dict(color="#1a1a1a",size=12)))
    fig.update_layout(title="Similarity Scores",xaxis_range=[0,110],height=max(250,len(papers)*40),margin=dict(l=10,r=40,t=40,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#f8f9fa",yaxis=dict(automargin=True)); return fig

def domain_pie(dd):
    fig=px.pie(names=list(dd.keys()),values=list(dd.values()),title="Domain Distribution",color_discrete_sequence=px.colors.qualitative.Set3,hole=0.35)
    fig.update_layout(height=260,margin=dict(l=5,r=5,t=40,b=5),paper_bgcolor="rgba(0,0,0,0)"); return fig

def year_bar(yd):
    years=[str(y) for y in sorted(int(k) for k in yd.keys()) if y>0]; counts=[yd.get(y,yd.get(int(y),0)) for y in years]
    fig=go.Figure(go.Bar(x=years,y=counts,marker_color="#1565c0",text=counts,textposition="outside",textfont=dict(color="#1a1a1a",size=12)))
    fig.update_layout(title="Publication Years",height=230,margin=dict(l=10,r=10,t=40,b=20),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#f8f9fa"); return fig

def render_results(result):
    novelty=result.get("novelty",{}); sim=result.get("similarity",{}); papers=result.get("similar_papers",[])
    gaps=result.get("gaps",{}); titles=result.get("title_suggestions",[]); explanation=result.get("explanation","")
    hallucination=result.get("hallucination",{}); plagiarism=result.get("plagiarism",{})
    label=novelty.get("label","MEDIUM"); pct=novelty.get("percentage",50); n_color=novelty.get("color","#f39c12")
    h_score=hallucination.get("grounding_score",100); h_color=hallucination.get("color","#2ecc71")
    p_score=plagiarism.get("plagiarism_score",0.0); p_color=plagiarism.get("color","#2ecc71")
    o_score=plagiarism.get("originality_score",100.0); risk=plagiarism.get("risk_level","SAFE")
    st.markdown("---"); st.subheader("📊 Analysis Dashboard")
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.plotly_chart(gauge(pct,f"<b>Novelty</b><br>{label}","%",n_color),use_container_width=True,key="ng")
        st.markdown(f'<div style="text-align:center"><span class="novelty-badge-{label.lower()}">{label}</span></div>',unsafe_allow_html=True)
    with c2:
        st.plotly_chart(plagiarism_gauge(p_score,p_color),use_container_width=True,key="pg")
        st.markdown(f'<div style="text-align:center;color:{p_color}"><b>{plagiarism.get("icon","")} {risk} RISK</b></div>',unsafe_allow_html=True)
    with c3: st.plotly_chart(gauge(o_score,"<b>Originality</b>","%","#2ecc71"),use_container_width=True,key="og")
    with c4: st.plotly_chart(gauge(h_score,"<b>AI Grounding</b>","/100",h_color),use_container_width=True,key="hg")
    stats=sim.get("stats",{}); m1,m2,m3,m4=st.columns(4)
    m1.metric("Max Similarity",f"{stats.get('max_sim',0)*100:.1f}%"); m2.metric("Mean Similarity",f"{stats.get('mean_sim',0)*100:.1f}%")
    m3.metric("Papers Retrieved",len(papers)); m4.metric("Plagiarism Score",f"{p_score:.1f}%")
    st.markdown(f'<div class="metric-card">{novelty.get("description","")}</div>',unsafe_allow_html=True)
    if sim.get("duplicate_risk"): st.error(f"🚨 Duplicate Risk! {len(sim.get('duplicates',[]))} paper(s) ≥90% similar.")
    st.markdown("---"); st.subheader("🔍 Plagiarism Detection Report")
    css={"SAFE":"plag-safe","LOW":"plag-low","MEDIUM":"plag-medium","HIGH":"plag-high"}.get(risk,"plag-safe")
    st.markdown(f'<div class="{css}"><b>{plagiarism.get("icon","")} {risk} — {p_score:.1f}% Plagiarism | {o_score:.1f}% Original</b><br>{plagiarism.get("message","")}</div>',unsafe_allow_html=True)
    ps=plagiarism.get("stats",{}); pc1,pc2,pc3,pc4=st.columns(4)
    pc1.metric("Plagiarism",f"{p_score:.1f}%"); pc2.metric("Originality",f"{o_score:.1f}%")
    pc3.metric("Sentences Checked",ps.get("total_sentences_checked",0)); pc4.metric("Phrases Flagged",ps.get("phrases_flagged",0))
    if plagiarism.get("matched_papers"):
        with st.expander("📄 Matched Papers"):
            for mp in plagiarism["matched_papers"]:
                rc="#e74c3c" if mp["risk"]=="HIGH" else "#f39c12" if mp["risk"]=="MEDIUM" else "#3498db"
                st.markdown(f'<div class="paper-card"><b style="color:{rc}">[{mp["risk"]}]</b> {mp["title"]}<br><small>Year:{mp["year"]} · Domain:{mp["domain"]} · Similarity:<b>{mp["similarity_pct"]:.1f}%</b></small></div>',unsafe_allow_html=True)
    sent_matches=plagiarism.get("sentence_matches",[])
    if sent_matches:
        with st.expander(f"⚠️ Flagged Sentences ({len(sent_matches)})"):
            for sm in sent_matches:
                st.markdown(f'<div class="flagged-sent"><b>Your text:</b> "{sm.get("user_sentence","")[:150]}"<br><b>Matches:</b> "{sm.get("paper_sentence","")[:150]}"<br><small>📄 {sm.get("paper_title","")[:60]} ({sm.get("paper_year","")}) — <b>{sm.get("similarity_pct",0):.1f}% similar</b></small></div>',unsafe_allow_html=True)
    else: st.success("✅ No sentence-level matches found.")
    phrase_matches=plagiarism.get("phrase_matches",[])
    if phrase_matches:
        with st.expander(f"🔤 Exact Phrase Matches ({len(phrase_matches)})"):
            for pm in phrase_matches:
                st.markdown(f'<div class="phrase-match">🔤 <b>"{pm.get("phrase","")}"</b> ({pm.get("word_count",0)} words) — {pm.get("paper_title","")[:60]} ({pm.get("paper_year","")})</div>',unsafe_allow_html=True)
    else: st.success("✅ No exact phrase matches found.")
    if plagiarism.get("recommendations"):
        with st.expander("💡 How to Fix Plagiarism Issues"):
            for r in plagiarism["recommendations"]: st.markdown(f"- {r}")
    st.markdown("---"); st.subheader("📚 Most Similar Research Papers")
    if papers:
        st.plotly_chart(sim_bar(papers),use_container_width=True,key="sb")
        for p in papers[:5]:
            with st.expander(f"📄 [{p.get('rank','?')}] {p.get('title','')} — {p.get('similarity_pct',0):.1f}%"):
                c1,c2,c3=st.columns(3)
                c1.markdown(f"**Year:** {p.get('year','N/A')}"); c2.markdown(f"**Domain:** {p.get('domain','N/A')}"); c3.markdown(f"**Region:** {p.get('region','N/A')}")
                st.markdown(f"**Abstract:** {p.get('abstract','')[:400]}…")
                if st.button(f"📥 Find & Download This Paper",key=f"find_{p.get('rank','')}"):
                    st.session_state["dl_search_title"]=p.get("title","")
                    st.info("👉 Switch to the 📥 Paper Downloader tab!")
    else: st.info("No similar papers found!")
    if papers:
        st.markdown("---"); cd,cy=st.columns(2)
        with cd: st.plotly_chart(domain_pie(sim.get("domain_dist",{})),use_container_width=True,key="dp")
        with cy: st.plotly_chart(year_bar(sim.get("year_dist",{})),use_container_width=True,key="yb")
    st.markdown("---"); st.subheader("🔍 Research Gap Report")
    for stmt in gaps.get("gap_statements",[]): st.markdown(f'<div class="gap-card">{stmt}</div>',unsafe_allow_html=True)
    if gaps.get("gap_dimensions"):
        with st.expander("📋 Detailed Gap Dimensions"):
            d1,d2=st.columns(2)
            with d1:
                for key,lbl in [("regional_gaps","📍 Regions"),("population_gaps","👥 Populations"),("temporal_gaps","📅 Time Periods")]:
                    items=gaps["gap_dimensions"].get(key,[])
                    if items: st.markdown(f"**{lbl}:**"); [st.markdown(f"  - {g.title()}") for g in items[:5]]
            with d2:
                for key,lbl in [("methodological_gaps","🔬 Methods"),("thematic_gaps","💡 Themes"),("theoretical_gaps","📚 Theories")]:
                    items=gaps["gap_dimensions"].get(key,[])
                    if items: st.markdown(f"**{lbl}:**"); [st.markdown(f"  - {g.title()}") for g in items[:5]]
    st.markdown("---"); st.subheader("✍️ Suggested Improved Research Titles")
    for i,t in enumerate(titles,1): st.markdown(f'<div class="title-suggestion">📝 <b>Option {i}:</b> {t}</div>',unsafe_allow_html=True)
    if novelty.get("suggestions"):
        st.markdown("---"); st.subheader("💡 Recommendations")
        for s in novelty["suggestions"]: st.markdown(f"- {s}")
    st.markdown("---"); st.subheader("📋 AI-Generated Research Report")
    with st.expander("📖 View Full Report",expanded=True): st.markdown(explanation.replace("\n","  \n"))
    st.markdown("---"); st.subheader("🛡️ AI Hallucination Check")
    h_count=hallucination.get("hallucination_count",0); h_total=hallucination.get("total_claims",0)
    hc1,hc2,hc3,hc4=st.columns(4)
    hc1.metric("Grounding Score",f"{h_score}/100"); hc2.metric("Total Claims",h_total); hc3.metric("Grounded",h_total-h_count); hc4.metric("Flagged",h_count)
    h_css="hallucination-safe" if h_score>=80 else "hallucination-warn" if h_score>=60 else "hallucination-danger"
    st.markdown(f'<div class="{h_css}"><b>{hallucination.get("label","")}</b> — Grounding Score: {h_score}/100</div>',unsafe_allow_html=True)
    if hallucination.get("flagged_claims"):
        with st.expander(f"🔴 {len(hallucination['flagged_claims'])} Flagged Claim(s)"):
            for fc in hallucination["flagged_claims"]:
                sev=fc.get("severity","LOW"); icon="🔴" if sev=="HIGH" else "🟡" if sev=="MEDIUM" else "🟢"
                st.markdown(f'<div class="flagged-claim">{icon} <b>[{sev}]</b> {fc.get("issue","")}<br><small><i>"{fc.get("claim","")[:120]}…"</i></small></div>',unsafe_allow_html=True)
    else: st.success("✅ No hallucinations detected.")
    st.markdown("---")
    st.download_button(label="📥 Download Full Analysis (JSON)",data=json.dumps(result,indent=2,ensure_ascii=False),file_name="research_analysis.json",mime="application/json",use_container_width=True)


# ═══════════════════════════════════════════════════════
# SECTION B — DOWNLOADER HELPERS
# ═══════════════════════════════════════════════════════

UNPAYWALL_EMAIL = "raguljayakumar34@gmail.com"
DL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (ResearchBot/3.0; mailto:raguljayakumar34@gmail.com)",
    "Accept":     "application/json,text/html,*/*"
}

DOMAIN_LIST = [
    "All Domains",
    "── 🇮🇳 Indian Languages ──",
    "Tamil Literature","Tamil Language","Tamil Studies","Hindi Literature",
    "Sanskrit","Kannada","Telugu","Malayalam","Urdu","Bengali",
    "── 📖 English & Linguistics ──",
    "English Literature","English Language Teaching","Linguistics",
    "Applied Linguistics","Postcolonial Studies","Comparative Literature",
    "Translation Studies","Ecocriticism","Discourse Analysis",
    "── 🎭 Arts & Humanities ──",
    "History","Philosophy","Sociology","Political Science","Anthropology",
    "Geography","Cultural Studies","Women Studies","Diaspora Literature",
    "Library Science","Fine Arts","Music","Journalism",
    "── 🔬 Pure Science ──",
    "Physics","Chemistry","Mathematics","Statistics","Botany","Zoology",
    "Microbiology","Biochemistry","Environmental Science","Geology","Astronomy",
    "Marine Science","Genetics","Bioinformatics",
    "── 👥 Social Science ──",
    "Economics","Education","Psychology","Commerce","Public Administration",
    "Social Work","Management","Business Administration","Finance","Marketing",
    "── ⚙️ Engineering ──",
    "Computer Science and Engineering","Electronics and Communication Engineering",
    "Electrical Engineering","Mechanical Engineering","Civil Engineering",
    "Chemical Engineering","Artificial Intelligence","Machine Learning",
    "Deep Learning","Internet of Things","Cybersecurity","Data Science",
    "Robotics and Automation","VLSI Design","Power Systems Engineering",
    "Renewable Energy Engineering","Structural Engineering",
    "Environmental Engineering","Nanotechnology","Materials Science",
    "Biomedical Engineering","Aerospace Engineering","Information Technology",
    "── 🏥 Medical & Health ──",
    "Medicine","Pharmacy","Nursing","Public Health","Ayurveda",
    "Dentistry","Biotechnology","Neuroscience","Oncology",
    "── 🌾 Agriculture ──",
    "Agriculture","Horticulture","Agronomy","Soil Science",
    "Agricultural Engineering","Fisheries","Food Technology","Veterinary Science",
]


def make_result(source, title, authors, year, abstract, pdf_url, page_url,
                doi=None, doc_type="Article", country="Global", pages=None):
    return {
        "source":source,"title":(title or "").strip(),"authors":authors or "N/A",
        "year":str(year or "N/A"),"abstract":((abstract or "No abstract available.")[:350]+"…"),
        "pdf_url":pdf_url,"page_url":page_url or "","free":pdf_url is not None,
        "doi":doi,"type":doc_type,"country":country,"pages":pages,
    }


def get_unpaywall_pdf(doi):
    if not doi: return None
    try:
        r=requests.get(f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}",headers=DL_HEADERS,timeout=10)
        if r.status_code==200:
            best=r.json().get("best_oa_location")
            if best: return best.get("url_for_pdf") or best.get("url")
    except: pass
    return None


def fetch_pdf_bytes(pdf_url):
    """
    Fetch PDF into memory — works on both desktop and mobile.
    Returns (bytes, "ok") on success or (None, error_message) on failure.
    """
    try:
        resp = requests.get(pdf_url, headers=DL_HEADERS, timeout=60,
                            stream=True, allow_redirects=True)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        ct = resp.headers.get("content-type","").lower()
        if "html" in ct and "pdf" not in ct:
            return None, "Paywalled — server returned login page"
        data = resp.content
        if len(data) < 10000:
            return None, "File too small — not a valid PDF"
        if data[:5] != b"%PDF-":
            return None, "Not a valid PDF — may be paywalled"
        return data, "ok"
    except Exception as e:
        return None, str(e)


def save_to_folder(pdf_bytes, title, source, folder):
    """Also save to server folder for desktop users."""
    try:
        safe     = re.sub(r'[\\/*?:"<>|🇮🇳🌍🇪🇺🇬🇧🇫🇷🇸🇪🇦🇺]',"",title)[:70].strip()
        src_safe = re.sub(r'[\\/*?:"<>|🇮🇳🌍🇪🇺🇬🇧🇫🇷🇸🇪🇦🇺]',"",source).strip()
        filepath = folder / f"[{src_safe}] {safe}.pdf"
        if not filepath.exists():
            with open(filepath,"wb") as f: f.write(pdf_bytes)
        return str(filepath)
    except: return ""


# ═══════════════════════════════════════════════════════
# JOURNAL SOURCES
# ═══════════════════════════════════════════════════════

def search_arxiv(q,n=5):
    results=[]
    try:
        resp=requests.get(f"http://export.arxiv.org/api/query?search_query=all:{quote(q)}&max_results={n}&sortBy=relevance",headers=DL_HEADERS,timeout=15)
        root=ET.fromstring(resp.content); ns={"atom":"http://www.w3.org/2005/Atom"}
        for e in root.findall("atom:entry",ns):
            pid=e.find("atom:id",ns).text.strip(); aid=pid.split("/abs/")[-1]
            auths=[a.find("atom:name",ns).text for a in e.findall("atom:author",ns)]
            results.append(make_result("arXiv",e.find("atom:title",ns).text.strip().replace("\n"," "),", ".join(auths[:3])+(" et al." if len(auths)>3 else ""),e.find("atom:published",ns).text[:4],e.find("atom:summary",ns).text.strip(),f"https://arxiv.org/pdf/{aid}.pdf",pid,doc_type="Preprint"))
    except: pass
    return results

def search_semantic_scholar(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.semanticscholar.org/graph/v1/paper/search",params={"query":q,"limit":n,"fields":"title,authors,year,abstract,openAccessPdf,externalIds,url"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("data",[]):
            oa=p.get("openAccessPdf"); pdf=oa.get("url") if oa else None
            doi=p.get("externalIds",{}).get("DOI"); aut=[a["name"] for a in p.get("authors",[])]
            results.append(make_result("Semantic Scholar",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("year"),p.get("abstract"),pdf,p.get("url",""),doi))
    except: pass
    return results

def search_ieee(q,n=5):
    results=[]
    try:
        resp=requests.get("https://ieeexplore.ieee.org/rest/search",params={"queryText":q,"newsearch":"true","pageNumber":1,"rowsPerPage":n},headers={**DL_HEADERS,"Referer":"https://ieeexplore.ieee.org"},timeout=15)
        for p in resp.json().get("records",[]):
            doi=p.get("doi",""); pdf=get_unpaywall_pdf(doi) if doi else None
            aut=[a.get("preferredName","") for a in p.get("authors",[])]
            results.append(make_result("IEEE Xplore",p.get("articleTitle",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publicationYear"),p.get("abstract"),pdf,f"https://ieeexplore.ieee.org/document/{p.get('articleNumber','')}",doi))
    except: pass
    return results

def search_springer(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.springernature.com/meta/v2/json",params={"q":q,"p":n,"api_key":"b374210d2db59a18aa3bfc1eac3e1876"},headers=DL_HEADERS,timeout=15)
        for r in resp.json().get("records",[]):
            doi=r.get("doi",""); pdf=get_unpaywall_pdf(doi) if doi else None
            aut=[c.get("creator","") for c in r.get("creators",[])]
            results.append(make_result("Springer",r.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),r.get("publicationDate","")[:4],r.get("abstract"),pdf,f"https://doi.org/{doi}" if doi else "",doi,doc_type=r.get("contentType","Article")))
    except: pass
    return results

def search_openalex(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.openalex.org/works",params={"search":q,"per-page":n,"mailto":UNPAYWALL_EMAIL},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            oa=p.get("open_access",{}); pdf=oa.get("oa_url") if oa.get("is_oa") else None
            doi=(p.get("doi","").replace("https://doi.org/","")) if p.get("doi") else None
            aut=[a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
            ab_inv=p.get("abstract_inverted_index"); abstract=""
            if ab_inv:
                words={v:k for k,vals in ab_inv.items() for v in vals}
                abstract=" ".join(words[i] for i in sorted(words))
            results.append(make_result("OpenAlex",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publication_year"),abstract,pdf,p.get("id",""),doi))
    except: pass
    return results

def search_pubmed(q,n=5):
    results=[]
    try:
        s=requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",params={"db":"pmc","term":q,"retmax":n,"retmode":"json"},headers=DL_HEADERS,timeout=15)
        ids=s.json().get("esearchresult",{}).get("idlist",[])
        if not ids: return results
        f=requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",params={"db":"pmc","id":",".join(ids),"retmode":"json"},headers=DL_HEADERS,timeout=15)
        data=f.json().get("result",{})
        for pid in ids:
            p=data.get(pid,{})
            if not p: continue
            aut=[a.get("name","") for a in p.get("authors",[])]
            results.append(make_result("PubMed Central",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("pubdate","")[:4],"PubMed Central open access article.",f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pid}/pdf/",f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pid}/"))
    except: pass
    return results

def search_doaj(q,n=5):
    results=[]
    try:
        resp=requests.get("https://doaj.org/api/search/articles",params={"q":q,"pageSize":n},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            bib=p.get("bibjson",{}); links=bib.get("link",[])
            pdf=next((l["url"] for l in links if l.get("type")=="fulltext"),None)
            doi=next((i["id"] for i in bib.get("identifier",[]) if i.get("type")=="doi"),None)
            aut=[a.get("name","") for a in bib.get("author",[])]
            results.append(make_result("DOAJ",bib.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),bib.get("year"),bib.get("abstract"),pdf,f"https://doaj.org/article/{p.get('id','')}",doi))
    except: pass
    return results

def search_core(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.core.ac.uk/v3/search/works",params={"q":q,"limit":n},headers={**DL_HEADERS,"Authorization":"Bearer "},timeout=15)
        for p in resp.json().get("results",[]):
            pdf=p.get("downloadUrl") or p.get("fullTextIdentifier")
            aut=[a.get("name","") for a in p.get("authors",[])]
            results.append(make_result("CORE",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("yearPublished"),p.get("abstract"),pdf,p.get("links",[{}])[0].get("url","") if p.get("links") else "",p.get("doi")))
    except: pass
    return results

def search_crossref(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.crossref.org/works",params={"query.title":q,"rows":n,"mailto":UNPAYWALL_EMAIL},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("message",{}).get("items",[]):
            doi=p.get("DOI",""); pdf=get_unpaywall_pdf(doi) if doi else None
            aut=[f"{a.get('given','')} {a.get('family','')}".strip() for a in p.get("author",[])]
            pub=p.get("published",{}).get("date-parts",[[""]])[0]
            ab=re.sub(r'<[^>]+',"",p.get("abstract",""))
            results.append(make_result("CrossRef",p.get("title",[""])[0] if p.get("title") else "",", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),pub[0] if pub else "N/A",ab,pdf,f"https://doi.org/{doi}" if doi else "",doi))
    except: pass
    return results


# ═══════════════════════════════════════════════════════
# PhD THESIS SOURCES
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# 🇮🇳 INDIAN JOURNAL SOURCES
# ═══════════════════════════════════════════════════════

def search_openalex_india(q,n=5):
    """OpenAlex filtered to Indian institutions"""
    results=[]
    try:
        resp=requests.get("https://api.openalex.org/works",params={"search":q,"per-page":n,"mailto":UNPAYWALL_EMAIL,"filter":"institutions.country_code:IN"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            oa=p.get("open_access",{}); pdf=oa.get("oa_url") if oa.get("is_oa") else None
            doi=(p.get("doi","").replace("https://doi.org/","")) if p.get("doi") else None
            aut=[a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
            ab_inv=p.get("abstract_inverted_index"); abstract=""
            if ab_inv:
                words={v:k for k,vals in ab_inv.items() for v in vals}
                abstract=" ".join(words[i] for i in sorted(words))
            results.append(make_result("OpenAlex India 🇮🇳",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publication_year"),abstract,pdf,p.get("id",""),doi,doc_type="Indian Research",country="India"))
    except: pass
    return results

def search_semantic_india(q,n=5):
    """Semantic Scholar filtered to Indian papers"""
    results=[]
    try:
        resp=requests.get("https://api.semanticscholar.org/graph/v1/paper/search",params={"query":f"{q} India","limit":n,"fields":"title,authors,year,abstract,openAccessPdf,externalIds,url"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("data",[]):
            oa=p.get("openAccessPdf"); pdf=oa.get("url") if oa else None
            doi=p.get("externalIds",{}).get("DOI"); aut=[a["name"] for a in p.get("authors",[])]
            results.append(make_result("Semantic India 🇮🇳",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("year"),p.get("abstract"),pdf,p.get("url",""),doi,doc_type="Indian Research",country="India"))
    except: pass
    return results

def search_nopr_niscair(q,n=5):
    """NISCAIR/NOPR — CSIR India journals"""
    results=[]
    try:
        resp=requests.get("http://nopr.niscair.res.in/rest/items/find-by-metadata-field",json={"key":"dc.title","value":q,"language":"en_US"},headers={**DL_HEADERS,"Content-Type":"application/json"},timeout=15)
        for item in resp.json()[:n]:
            item_id=item.get("id",""); title_=item.get("name",""); handle=item.get("handle","")
            page_url=f"http://nopr.niscair.res.in/handle/{handle}"; pdf_url=None
            try:
                bs=requests.get(f"http://nopr.niscair.res.in/rest/items/{item_id}/bitstreams",headers=DL_HEADERS,timeout=10)
                for b in bs.json():
                    if b.get("mimeType","")=="application/pdf":
                        pdf_url=f"http://nopr.niscair.res.in/rest/bitstreams/{b['id']}/retrieve"; break
            except: pass
            if title_:
                results.append(make_result("NISCAIR/NOPR 🇮🇳",title_,"CSIR Researcher","N/A","CSIR Journal Article — NISCAIR Open Access",pdf_url,page_url,doc_type="Journal Article",country="India"))
    except: pass
    return results

def search_current_science(q,n=5):
    """Current Science — IISc Bangalore (reputed Indian journal)"""
    results=[]
    try:
        resp=requests.get(f"https://www.currentscience.ac.in/php/search.php",params={"q":q,"type":"title"},headers=DL_HEADERS,timeout=15)
        rows=re.findall(r'<tr[^>]*>(.*?)</tr>',resp.text,re.DOTALL)
        for row in rows[:n]:
            t=re.search(r'<b>(.*?)</b>',row); href=re.search(r'href="([^"]+\.pdf)"',row)
            auth=re.search(r'<i>(.*?)</i>',row); yr=re.search(r'(\d{4})',row)
            if t and href:
                pdf=href.group(1) if href.group(1).startswith("http") else f"https://www.currentscience.ac.in{href.group(1)}"
                results.append(make_result("Current Science 🇮🇳",re.sub(r'<[^>]+',"",t.group(1)).strip(),re.sub(r'<[^>]+',"",auth.group(1)).strip() if auth else "N/A",yr.group(1) if yr else "N/A","Current Science — IISc Bangalore (Reputed Indian Journal)",pdf,"https://www.currentscience.ac.in",doc_type="Journal Article",country="India"))
    except: pass
    return results

def search_doaj_india(q,n=5):
    """DOAJ filtered to Indian journals"""
    results=[]
    try:
        resp=requests.get("https://doaj.org/api/search/articles",params={"q":f"{q} India","pageSize":n},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            bib=p.get("bibjson",{}); links=bib.get("link",[])
            pdf=next((l["url"] for l in links if l.get("type")=="fulltext"),None)
            doi=next((i["id"] for i in bib.get("identifier",[]) if i.get("type")=="doi"),None)
            aut=[a.get("name","") for a in bib.get("author",[])]
            results.append(make_result("DOAJ India 🇮🇳",bib.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),bib.get("year"),bib.get("abstract"),pdf,f"https://doaj.org/article/{p.get('id','')}",doi,doc_type="Journal Article",country="India"))
    except: pass
    return results

def search_irjet(q,n=5):
    """IRJET — International Research Journal of Engineering and Technology"""
    results=[]
    try:
        resp=requests.get(f"https://www.irjet.net/search-result.php",params={"search":q},headers=DL_HEADERS,timeout=15)
        items=re.findall(r'href="(https://www\.irjet\.net/archives/[^"]+\.pdf)"[^>]*>(.*?)</a>',resp.text,re.DOTALL)
        for pdf_url,t in items[:n]:
            t=re.sub(r'<[^>]+',"",t).strip()
            if t:
                results.append(make_result("IRJET 🇮🇳",t,"Indian Researcher","N/A","International Research Journal of Engineering and Technology — Free PDF",pdf_url,"https://www.irjet.net",doc_type="Journal Article",country="India"))
    except: pass
    return results

def search_ijser(q,n=5):
    """IJSER — International Journal of Scientific and Engineering Research"""
    results=[]
    try:
        resp=requests.get(f"https://www.ijser.org/research-paper-search.aspx",params={"title":q},headers=DL_HEADERS,timeout=15)
        pdfs=re.findall(r'href="(/paper/[^"]+\.pdf)"',resp.text)
        titles_=re.findall(r'<h\d[^>]*>(.*?)</h\d>',resp.text,re.DOTALL)
        for i,pdf_path in enumerate(pdfs[:n]):
            t=re.sub(r'<[^>]+',"",titles_[i]).strip() if i<len(titles_) else f"IJSER Paper {i+1}"
            results.append(make_result("IJSER 🇮🇳",t,"Indian Researcher","N/A","International Journal of Scientific & Engineering Research — Free PDF",f"https://www.ijser.org{pdf_path}","https://www.ijser.org",doc_type="Journal Article",country="India"))
    except: pass
    return results

def search_jetir(q,n=5):
    """JETIR — Journal of Emerging Technologies and Innovative Research"""
    results=[]
    try:
        resp=requests.get(f"https://www.jetir.org/search.php",params={"q":q},headers={**DL_HEADERS,"Referer":"https://www.jetir.org"},timeout=15)
        pdfs=re.findall(r'href="(https://www\.jetir\.org/papers/[^"]+\.pdf)"',resp.text)
        titles_=re.findall(r'<b>(.*?)</b>',resp.text)
        for i,pdf_url in enumerate(pdfs[:n]):
            t=re.sub(r'<[^>]+',"",titles_[i]).strip() if i<len(titles_) else f"JETIR Paper {i+1}"
            results.append(make_result("JETIR 🇮🇳",t,"Indian Researcher","N/A","Journal of Emerging Technologies and Innovative Research — Free PDF",pdf_url,"https://www.jetir.org",doc_type="Journal Article",country="India"))
    except: pass
    return results

def search_ijcrt(q,n=5):
    """IJCRT — International Journal of Creative Research Thoughts"""
    results=[]
    try:
        resp=requests.get(f"https://ijcrt.org/search.php",params={"search":q},headers={**DL_HEADERS,"Referer":"https://ijcrt.org"},timeout=15)
        pdfs=re.findall(r'href="(https://ijcrt\.org/papers/[^"]+\.pdf)"',resp.text)
        titles_=re.findall(r'<font[^>]*color="#0000FF"[^>]*><b>(.*?)</b></font>',resp.text,re.DOTALL)
        for i,pdf_url in enumerate(pdfs[:n]):
            t=re.sub(r'<[^>]+',"",titles_[i]).strip() if i<len(titles_) else f"IJCRT Paper {i+1}"
            results.append(make_result("IJCRT 🇮🇳",t,"Indian Researcher","N/A","International Journal of Creative Research Thoughts — Free PDF",pdf_url,"https://ijcrt.org",doc_type="Journal Article",country="India"))
    except: pass
    return results


def thesis_shodhganga(q,n=5):
    results=[]
    try:
        resp=requests.get("http://shodhganga.inflibnet.ac.in:8080/jspui/oai/request",params={"verb":"Search","query":q,"metadataPrefix":"oai_dc","rows":n},headers=DL_HEADERS,timeout=15)
        root=ET.fromstring(resp.content); ns={"oai":"http://www.openarchives.org/OAI/2.0/","dc":"http://purl.org/dc/elements/1.1/"}
        for record in root.findall(".//oai:record",ns):
            meta=record.find(".//oai:metadata",ns)
            if meta is None: continue
            t=meta.find(".//dc:title",ns); auth=meta.find(".//dc:creator",ns)
            date=meta.find(".//dc:date",ns); desc=meta.find(".//dc:description",ns); ident=meta.find(".//dc:identifier",ns)
            title_=t.text.strip() if t is not None else ""; page_url=ident.text.strip() if ident is not None else ""
            pdf_url=None
            if page_url and "handle" in page_url:
                hid=page_url.split("handle/")[-1]
                pdf_url=f"https://shodhganga.inflibnet.ac.in/bitstream/handle/{hid}/thesis.pdf?sequence=1&isAllowed=y"
            if title_:
                results.append(make_result("Shodhganga 🇮🇳",title_,auth.text if auth is not None else "Indian PhD Scholar",date.text[:4] if date is not None else "N/A",desc.text[:300] if desc is not None else "Indian PhD Thesis",pdf_url,page_url,doc_type="PhD Thesis 🎓",country="India",pages="150-400"))
    except: pass
    if not results:
        try:
            resp=requests.get("https://api.openalex.org/works",params={"search":q,"per-page":n,"mailto":UNPAYWALL_EMAIL,"filter":"institutions.country_code:IN,type:dissertation"},headers=DL_HEADERS,timeout=15)
            for p in resp.json().get("results",[]):
                oa=p.get("open_access",{}); pdf=oa.get("oa_url") if oa.get("is_oa") else None
                aut=[a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
                ab_inv=p.get("abstract_inverted_index"); abstract=""
                if ab_inv:
                    words={v:k for k,vals in ab_inv.items() for v in vals}; abstract=" ".join(words[i] for i in sorted(words))
                results.append(make_result("Shodhganga 🇮🇳",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publication_year"),abstract or "Indian PhD Thesis",pdf,p.get("id",""),None,doc_type="PhD Thesis 🎓",country="India",pages="150-400"))
        except: pass
    return results[:n]

def thesis_ndltd(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.openalex.org/works",params={"search":q,"per-page":n,"mailto":UNPAYWALL_EMAIL,"filter":"type:dissertation"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            oa=p.get("open_access",{}); pdf=oa.get("oa_url") if oa.get("is_oa") else None
            aut=[a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
            ab_inv=p.get("abstract_inverted_index"); abstract=""
            if ab_inv:
                words={v:k for k,vals in ab_inv.items() for v in vals}; abstract=" ".join(words[i] for i in sorted(words))
            results.append(make_result("NDLTD Global 🌍",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publication_year"),abstract,pdf,p.get("id",""),None,doc_type="PhD Thesis 🎓",pages="150-400"))
    except: pass
    return results[:n]

def thesis_zenodo(q,n=5):
    results=[]
    try:
        resp=requests.get("https://zenodo.org/api/records",params={"q":q,"size":n,"sort":"mostrecent","type":"publication","subtype":"thesis"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("hits",{}).get("hits",[]):
            meta=p.get("metadata",{}); files=p.get("files",[])
            pdf=next((f["links"]["self"] for f in files if f.get("type")=="pdf"),None)
            aut=[a.get("name","") for a in meta.get("creators",[])]
            results.append(make_result("Zenodo Theses",meta.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),str(meta.get("publication_date",""))[:4],meta.get("description","")[:300],pdf,p.get("links",{}).get("html",""),meta.get("doi",""),doc_type="PhD Thesis 🎓",pages="100-400"))
    except: pass
    return results[:n]

def thesis_hal(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.archives-ouvertes.fr/search/",params={"q":q,"rows":n,"fl":"title_s,authFullName_s,producedDate_s,abstract_s,fileMain_s,uri_s","fq":"docType_s:THESE"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("response",{}).get("docs",[]):
            pdf=p.get("fileMain_s"); aut=p.get("authFullName_s",[])
            results.append(make_result("HAL Theses 🇫🇷",p.get("title_s",[""])[0] if p.get("title_s") else "",", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),str(p.get("producedDate_s",""))[:4],p.get("abstract_s",[""])[0] if p.get("abstract_s") else "French PhD Thesis",pdf,p.get("uri_s",""),doc_type="PhD Thesis 🎓",country="France",pages="150-350"))
    except: pass
    return results[:n]

def thesis_dart_europe(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.crossref.org/works",params={"query.title":q,"rows":n,"mailto":UNPAYWALL_EMAIL,"filter":"type:dissertation"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("message",{}).get("items",[]):
            doi=p.get("DOI",""); pdf=get_unpaywall_pdf(doi) if doi else None
            aut=[f"{a.get('given','')} {a.get('family','')}".strip() for a in p.get("author",[])]
            pub=p.get("published",{}).get("date-parts",[[""]])[0]
            results.append(make_result("DART-Europe 🇪🇺",p.get("title",[""])[0] if p.get("title") else "",", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),pub[0] if pub else "N/A","PhD Dissertation",pdf,f"https://doi.org/{doi}" if doi else "",doi,doc_type="PhD Thesis 🎓",pages="150-400"))
    except: pass
    return results[:n]

def thesis_ethos(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.semanticscholar.org/graph/v1/paper/search",params={"query":f"{q} doctoral thesis","limit":n,"fields":"title,authors,year,abstract,openAccessPdf,url"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("data",[]):
            oa=p.get("openAccessPdf"); pdf=oa.get("url") if oa else None
            aut=[a["name"] for a in p.get("authors",[])]
            if any(w in (p.get("abstract") or "").lower() for w in ["thesis","dissertation","doctoral","phd"]):
                results.append(make_result("EThOS 🇬🇧",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("year"),"PhD Thesis",pdf,p.get("url",""),None,doc_type="PhD Thesis 🎓",country="UK",pages="150-400"))
    except: pass
    return results[:n]

def thesis_oatd(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.openalex.org/works",params={"search":f"{q} dissertation thesis","per-page":n,"mailto":UNPAYWALL_EMAIL,"filter":"type:dissertation,open_access.is_oa:true"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            oa=p.get("open_access",{}); pdf=oa.get("oa_url")
            aut=[a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
            doi=(p.get("doi","").replace("https://doi.org/","")) if p.get("doi") else None
            ab_inv=p.get("abstract_inverted_index"); abstract=""
            if ab_inv:
                words={v:k for k,vals in ab_inv.items() for v in vals}; abstract=" ".join(words[i] for i in sorted(words))
            results.append(make_result("OATD 🌐",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publication_year"),abstract,pdf,p.get("id",""),doi,doc_type="PhD Thesis 🎓",pages="150-400"))
    except: pass
    return results[:n]

def thesis_springer(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.springernature.com/meta/v2/json",params={"q":f'{q} series:"Springer Theses"',"p":n,"api_key":"b374210d2db59a18aa3bfc1eac3e1876"},headers=DL_HEADERS,timeout=15)
        for r in resp.json().get("records",[]):
            doi=r.get("doi",""); pdf=get_unpaywall_pdf(doi) if doi else None
            aut=[c.get("creator","") for c in r.get("creators",[])]
            results.append(make_result("Springer Theses",r.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),r.get("publicationDate","")[:4],r.get("abstract","") or "Springer Theses — Recognized PhD Research",pdf,f"https://doi.org/{doi}" if doi else "",doi,doc_type="PhD Thesis 🎓",pages="150-300"))
    except: pass
    return results[:n]

def thesis_ieee_dissertations(q,n=5):
    results=[]
    try:
        resp=requests.get("https://ieeexplore.ieee.org/rest/search",params={"queryText":f"{q} doctoral dissertation","newsearch":"true","pageNumber":1,"rowsPerPage":n},headers={**DL_HEADERS,"Referer":"https://ieeexplore.ieee.org"},timeout=15)
        for p in resp.json().get("records",[]):
            doi=p.get("doi",""); pdf=get_unpaywall_pdf(doi) if doi else None
            aut=[a.get("preferredName","") for a in p.get("authors",[])]
            results.append(make_result("IEEE Dissertations",p.get("articleTitle",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publicationYear"),p.get("abstract"),pdf,f"https://ieeexplore.ieee.org/document/{p.get('articleNumber','')}",doi,doc_type="PhD Thesis 🎓",pages="100-300"))
    except: pass
    return results[:n]

def thesis_proquest(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.openalex.org/works",params={"search":q,"per-page":n,"mailto":UNPAYWALL_EMAIL,"filter":"type:dissertation,open_access.is_oa:true"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            oa=p.get("open_access",{}); pdf=oa.get("oa_url")
            aut=[a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
            doi=(p.get("doi","").replace("https://doi.org/","")) if p.get("doi") else None
            ab_inv=p.get("abstract_inverted_index"); abstract=""
            if ab_inv:
                words={v:k for k,vals in ab_inv.items() for v in vals}; abstract=" ".join(words[i] for i in sorted(words))
            results.append(make_result("PQDT Open",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publication_year"),abstract,pdf,p.get("id",""),doi,doc_type="PhD Dissertation 🎓",pages="150-400"))
    except: pass
    return results[:n]

def thesis_base(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi",params={"func":"PerformSearch","query":f"dcdoctype:thesis {q}","hits":n,"format":"json","boost":"oa"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("response",{}).get("docs",[]):
            urls=p.get("dclink",[]) if isinstance(p.get("dclink"),list) else [p.get("dclink","")]
            pdf=next((u for u in urls if ".pdf" in (u or "").lower()),urls[0] if urls else None)
            aut=p.get("dccreator","")
            if isinstance(aut,list): aut=", ".join(aut[:3])
            title_=p.get("dctitle","")
            if isinstance(title_,list): title_=title_[0] if title_ else ""
            if title_:
                results.append(make_result("BASE Theses",title_,aut or "N/A",str(p.get("dcyear","N/A")),"PhD Thesis from BASE academic search",pdf,pdf or "",doc_type="PhD Thesis 🎓",pages="100-400"))
    except: pass
    return results[:n]

def thesis_opendoar(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.core.ac.uk/v3/search/works",params={"q":f"{q} type:thesis","limit":n},headers={**DL_HEADERS,"Authorization":"Bearer "},timeout=15)
        for p in resp.json().get("results",[]):
            pdf=p.get("downloadUrl") or p.get("fullTextIdentifier")
            aut=[a.get("name","") for a in p.get("authors",[])]
            results.append(make_result("OpenDOAR Repos",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("yearPublished"),p.get("abstract"),pdf,p.get("links",[{}])[0].get("url","") if p.get("links") else "",p.get("doi"),doc_type="PhD Thesis 🎓",pages="100-400"))
    except: pass
    return results[:n]

def thesis_diva(q,n=5):
    results=[]
    try:
        resp=requests.get("https://api.openalex.org/works",params={"search":q,"per-page":n,"mailto":UNPAYWALL_EMAIL,"filter":"type:dissertation,institutions.country_code:SE|NO|DK|FI"},headers=DL_HEADERS,timeout=15)
        for p in resp.json().get("results",[]):
            oa=p.get("open_access",{}); pdf=oa.get("oa_url") if oa.get("is_oa") else None
            aut=[a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
            results.append(make_result("DiVA Portal 🇸🇪",p.get("title",""),", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),p.get("publication_year"),"Scandinavian PhD Thesis",pdf,p.get("id",""),None,doc_type="PhD Thesis 🎓",country="Scandinavia",pages="150-350"))
    except: pass
    return results[:n]


# ═══════════════════════════════════════════════════════
# SOURCE REGISTRY
# ═══════════════════════════════════════════════════════

JOURNAL_SOURCES = {
    "arXiv":            (search_arxiv,           "CS/Physics/Math preprints",  "#b71c1c"),
    "Semantic Scholar": (search_semantic_scholar, "All fields open access",     "#1565c0"),
    "IEEE Xplore":      (search_ieee,             "Engineering & CS journals",  "#00838f"),
    "Springer":         (search_springer,         "All fields",                 "#e65100"),
    "OpenAlex":         (search_openalex,         "All fields",                 "#0d47a1"),
    "PubMed Central":   (search_pubmed,           "Biomedical & life science",  "#1b5e20"),
    "DOAJ":             (search_doaj,             "Open access journals",       "#f57f17"),
    "CORE":             (search_core,             "Largest OA aggregator",      "#4a148c"),
    "CrossRef":         (search_crossref,         "All fields via DOI",         "#37474f"),
}

INDIA_SOURCES = {
    "OpenAlex India 🇮🇳":  (search_openalex_india,  "All Indian institution papers",     "#e65100"),
    "Semantic India 🇮🇳":  (search_semantic_india,   "Indian research papers",            "#6a1b9a"),
    "NISCAIR/NOPR 🇮🇳":    (search_nopr_niscair,     "CSIR journals — FREE PDF",          "#1b5e20"),
    "Current Science 🇮🇳": (search_current_science,  "IISc Bangalore journal — FREE PDF", "#00695c"),
    "DOAJ India 🇮🇳":      (search_doaj_india,       "Indian open access journals",       "#f57f17"),
    "IRJET 🇮🇳":           (search_irjet,            "Engineering & Technology — FREE",   "#d84315"),
    "IJSER 🇮🇳":           (search_ijser,            "Scientific & Engineering — FREE",   "#4527a0"),
    "JETIR 🇮🇳":           (search_jetir,            "Emerging Tech & Innovation — FREE", "#00838f"),
    "IJCRT 🇮🇳":           (search_ijcrt,            "Creative Research Thoughts — FREE", "#558b2f"),
}

THESIS_SOURCES = {
    "Shodhganga 🇮🇳":    (thesis_shodhganga,   "Indian PhD theses — INFLIBNET",    "#bf360c"),
    "NDLTD Global 🌍":   (thesis_ndltd,         "Global PhD dissertations",         "#1a237e"),
    "Zenodo Theses":     (thesis_zenodo,         "CERN open repository — theses",    "#1565c0"),
    "HAL Theses 🇫🇷":    (thesis_hal,           "French PhD theses — always free",  "#ad1457"),
    "DART-Europe 🇪🇺":   (thesis_dart_europe,   "700+ European universities",       "#004d40"),
    "EThOS 🇬🇧":         (thesis_ethos,         "British Library UK theses",        "#4e342e"),
    "OATD 🌐":           (thesis_oatd,           "Open access theses worldwide",     "#283593"),
    "Springer Theses":   (thesis_springer,       "Springer recognized PhD research", "#e65100"),
    "IEEE Dissertations":(thesis_ieee_dissertations,"IEEE doctoral dissertations",   "#00838f"),
    "PQDT Open":         (thesis_proquest,       "ProQuest open dissertations",      "#4527a0"),
    "BASE Theses":       (thesis_base,           "BASE academic thesis search",      "#33691e"),
    "OpenDOAR Repos":    (thesis_opendoar,       "Worldwide institutional repos",    "#006064"),
    "DiVA Portal 🇸🇪":   (thesis_diva,          "Scandinavian university theses",   "#37474f"),
}


# ═══════════════════════════════════════════════════════
# DOWNLOADER UI
# ═══════════════════════════════════════════════════════

def render_downloader_tab():
    st.markdown("### 📥 Research Paper & PhD Thesis Downloader")
    st.info("📱 **Mobile friendly** — PDF downloads directly to your phone/device!")

    # Search row
    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
    with col1:
        prefill = st.session_state.pop("dl_search_title","")
        title   = st.text_input("🔍 Title / Keywords", value=prefill,
                                placeholder="e.g., Machine Learning Water Quality India")
    with col2:
        domain_filter = st.selectbox("📚 Domain", DOMAIN_LIST)
    with col3:
        max_results = st.number_input("Results/source", min_value=1, max_value=10, value=3, step=1)
    with col4:
        folder_name = st.text_input("💾 Save folder", value="downloaded_papers")

    dl_folder = Path(folder_name)
    dl_folder.mkdir(exist_ok=True)

    def build_query(base, domain):
        if domain and domain != "All Domains" and not domain.startswith("──"):
            return f"{base} {domain}"
        return base

    # Source selection
    col_j, col_i, col_t = st.columns(3)
    with col_j:
        st.markdown("#### 📄 Global Journal Sources")
        j_sel = st.multiselect("", list(JOURNAL_SOURCES.keys()),
                               default=list(JOURNAL_SOURCES.keys()), key="j_sel",
                               label_visibility="collapsed")
    with col_i:
        st.markdown("#### 🇮🇳 Indian Sources")
        i_sel = st.multiselect("", list(INDIA_SOURCES.keys()),
                               default=list(INDIA_SOURCES.keys()), key="i_sel",
                               label_visibility="collapsed")
    with col_t:
        st.markdown("#### 🎓 PhD Thesis Sources (150-400 pages)")
        t_sel = st.multiselect("", list(THESIS_SOURCES.keys()),
                               default=list(THESIS_SOURCES.keys()), key="t_sel",
                               label_visibility="collapsed")

    all_sel    = j_sel + i_sel + t_sel
    search_btn = st.button(f"🔍 Search {len(all_sel)} Sources", use_container_width=True, type="primary")

    if search_btn and title.strip():
        query = build_query(title.strip(), domain_filter)
        all_results = []
        prog   = st.progress(0)
        status = st.empty()
        for idx, src in enumerate(all_sel):
            status.caption(f"🔍 Searching {src}…")
            fn = JOURNAL_SOURCES.get(src) or INDIA_SOURCES.get(src) or THESIS_SOURCES.get(src)
            if fn:
                try: all_results += fn[0](query, max_results)
                except: pass
            prog.progress((idx+1)/len(all_sel))
        status.empty(); prog.empty()

        seen, unique = set(), []
        for r in all_results:
            key = r["title"].lower()[:60]
            if key not in seen and r["title"].strip():
                seen.add(key); unique.append(r)

        st.session_state["dl_results"] = unique
        st.session_state["dl_folder"]  = folder_name

    results   = st.session_state.get("dl_results",[])
    dl_folder = Path(st.session_state.get("dl_folder", folder_name))

    if results:
        free_count   = sum(1 for r in results if r["free"])
        thesis_count = sum(1 for r in results if "Thesis" in r.get("type","") or "Dissertation" in r.get("type",""))
        link_count   = sum(1 for r in results if not r["free"] and r.get("page_url"))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Found",       len(results))
        c2.metric("✅ Free PDF",        free_count)
        c3.metric("🎓 PhD Theses",     thesis_count)
        c4.metric("🔗 Link Available", link_count)

        st.markdown("---")
        fc1,fc2,fc3 = st.columns(3)
        with fc1: show_thesis = st.checkbox("PhD Theses only 🎓", value=False)
        with fc2: show_free   = st.checkbox("Free PDF only ✅",    value=False)
        with fc3:
            src_opts   = list(set(r["source"] for r in results))
            src_filter = st.multiselect("Source filter", src_opts, default=src_opts)

        filtered = [r for r in results
                    if (not show_thesis or "Thesis" in r.get("type","") or "Dissertation" in r.get("type",""))
                    and (not show_free   or r["free"])
                    and r["source"] in src_filter]

        st.markdown(f"**Showing {len(filtered)} results**")

        for i, paper in enumerate(filtered):
            is_thesis = "Thesis" in paper.get("type","") or "Dissertation" in paper.get("type","")
            src_info  = JOURNAL_SOURCES.get(paper["source"]) or INDIA_SOURCES.get(paper["source"]) or THESIS_SOURCES.get(paper["source"])
            src_color = src_info[2] if src_info else "#555"
            icon      = "🎓" if is_thesis else "📄"
            pages     = f" · ~{paper['pages']} pages" if paper.get("pages") else ""

            with st.expander(
                f"{icon} [{paper['source']}] {paper['title'][:75]}{'…' if len(paper['title'])>75 else ''}{pages}"
            ):
                if paper["free"]:
                    status_html = '<span class="free-badge">✅ FREE PDF — Direct Download</span>'
                elif paper.get("page_url"):
                    status_html = '<span class="link-badge">🔗 Link — Visit Page to Download</span>'
                else:
                    status_html = ""

                st.markdown(
                    f'<span class="src-badge" style="background:{src_color}">{paper["source"]}</span> '
                    f'{status_html} {"<span class=thesis-tag>🎓 PhD THESIS</span>" if is_thesis else ""}',
                    unsafe_allow_html=True
                )

                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**Authors:** {paper['authors']}")
                c2.markdown(f"**Year:** {paper['year']}")
                c3.markdown(f"**Pages:** {paper.get('pages','N/A')}")
                if paper.get("doi"): st.markdown(f"**DOI:** `{paper['doi']}`")
                st.markdown(f"**Abstract:** {paper['abstract']}")

                if paper["free"] and paper["pdf_url"]:
                    # ── MOBILE-FRIENDLY DOWNLOAD ──────────────────
                    # First fetch into memory, then show download button
                    # This works on both mobile and desktop
                    fetch_key = f"fetched_{i}"
                    if fetch_key not in st.session_state:
                        if st.button("⬇️ Prepare Download", key=f"prep_{i}", use_container_width=True):
                            with st.spinner("⏳ Fetching PDF… please wait"):
                                pdf_bytes, msg = fetch_pdf_bytes(paper["pdf_url"])
                            if pdf_bytes:
                                st.session_state[fetch_key] = pdf_bytes
                                size_mb = len(pdf_bytes)/(1024*1024)
                                st.success(f"✅ Ready! {size_mb:.1f} MB — click button below to save")
                                # Also save to server folder
                                save_to_folder(pdf_bytes, paper["title"], paper["source"], dl_folder)
                            else:
                                st.warning(f"⚠️ {msg}")
                                if paper.get("page_url"):
                                    st.link_button("🌐 Open Page to Download", paper["page_url"], use_container_width=True)

                    if fetch_key in st.session_state:
                        pdf_bytes = st.session_state[fetch_key]
                        safe_name = re.sub(r'[\\/*?:"<>|]',"",paper["title"])[:60].strip()+".pdf"
                        size_mb   = len(pdf_bytes)/(1024*1024)
                        st.download_button(
                            label=f"📥 Download PDF to Device ({size_mb:.1f} MB)",
                            data=pdf_bytes,
                            file_name=safe_name,
                            mime="application/pdf",
                            key=f"dl_{i}",
                            use_container_width=True,
                            type="primary",
                        )
                        st.caption("📱 On mobile: tap button → PDF saves to Downloads folder")

                elif paper.get("page_url"):
                    st.markdown(
                        f'<div class="thesis-card">'
                        f'🔗 <b>{"Full PhD Thesis available at:" if is_thesis else "Paper page:"}</b><br>'
                        f'<a href="{paper["page_url"]}" target="_blank" style="color:#1565c0">'
                        f'{paper["page_url"][:80]}{"…" if len(paper["page_url"])>80 else ""}</a><br>'
                        f'<small>Tap/Click the link → find the PDF/Download button on that page</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.link_button("🌐 Open Page", paper["page_url"], use_container_width=True)

    # Downloaded files panel
    st.markdown("---")
    st.markdown("### 📂 Downloaded Papers & Theses (Server)")
    files = sorted(dl_folder.glob("*.pdf"), key=lambda f: f.stat().st_size, reverse=True)
    if files:
        total_mb = sum(f.stat().st_size for f in files)/(1024*1024)
        st.success(f"{len(files)} file(s) · {total_mb:.1f} MB total")
        for f in files:
            size_mb = f.stat().st_size/(1024*1024)
            c1,c2 = st.columns([5,1])
            icon = "🎓" if any(w in f.name for w in ["Thesis","Shodhganga","NDLTD","EThOS","DART","OATD","DiVA","HAL"]) else "📄"
            c1.markdown(f"{icon} `{f.name}` — **{size_mb:.1f} MB**")
            if c2.button("🗑️", key=f"del_{f.name}"): f.unlink(); st.rerun()
    else:
        st.info("No files on server yet. Downloads also save here automatically.")

    with st.expander("📊 All Sources Reference"):
        st.markdown("**📄 Global Journal Sources (9)**")
        for name,(fn,desc,col) in JOURNAL_SOURCES.items():
            st.markdown(f"- **{name}** — {desc}")
        st.markdown("**🇮🇳 Indian Sources (9)**")
        for name,(fn,desc,col) in INDIA_SOURCES.items():
            st.markdown(f"- **{name}** — {desc}")
        st.markdown("**🎓 PhD Thesis Sources (13) — Full 150-400 page documents**")
        for name,(fn,desc,col) in THESIS_SOURCES.items():
            st.markdown(f"- **{name}** — {desc}")
        st.info("📌 Some sources (Shodhganga, EThOS, DART-Europe) require university login for PDF. A direct page link is shown — click it to access and download from the university portal.")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("## 🔬 Research Detector Agent")
        st.markdown("**For PhD Scholars — India**")
        st.divider()
        st.markdown("### ⚙️ Analysis Settings")
        top_k = int(st.number_input("Papers to search", min_value=1, value=10, step=1))
        check_plag = st.toggle("Plagiarism Check", value=True)
        st.divider()
        st.markdown("### 📖 Features")
        st.markdown("""
- 📊 Novelty Score
- 🔍 Plagiarism Check
- 🔬 Gap Detection
- 🤖 AI Report (Gemini)
- 🛡️ Hallucination Check
- 📥 Downloader:
  - 9 Global Journal sources
  - 9 Indian sources 🇮🇳
  - 13 PhD Thesis sources 🎓
- 📱 Mobile Download Support
        """)
        st.caption("© 2024 · AI Research Tools · India")

    st.markdown('<p class="main-header">🔬 AI Research Novelty & Gap Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">For PhD Scholars in India · 9 Global + 9 Indian + 13 PhD Thesis Sources · Mobile Friendly 📱</p>', unsafe_allow_html=True)

    try:
        retriever = load_engines()
        st.sidebar.success(f"✅ {retriever.get_corpus_size()} papers loaded")
    except Exception as e:
        st.error(f"❌ Failed to load engines: {e}"); return

    tab1, tab2 = st.tabs(["🔬 Research Gap Analyser", "📥 Paper & Thesis Downloader 📱"])

    with tab1:
        # Initialize session state for extracted metadata
        if "extracted_title" not in st.session_state:
            st.session_state.extracted_title = ""
        if "extracted_abstract" not in st.session_state:
            st.session_state.extracted_abstract = ""
        if "extracted_keywords" not in st.session_state:
            st.session_state.extracted_keywords = ""
        if "extracted_domain" not in st.session_state:
            st.session_state.extracted_domain = ""
        if "extracted_full_text" not in st.session_state:
            st.session_state.extracted_full_text = ""

        # Option 1: File Upload
        st.subheader("📁 Option 1: Upload a Journal Paper (PDF / TXT)")
        uploaded_file = st.file_uploader("Upload your journal paper to automatically extract Title, Abstract, Keywords, and Domain", type=["pdf", "txt"])

        if uploaded_file is not None:
            if st.session_state.get("last_uploaded_file") != uploaded_file.name:
                with st.spinner("⏳ Extracting metadata from paper..."):
                    try:
                        filename = uploaded_file.name
                        file_bytes = uploaded_file.getvalue()
                        # Extract text
                        if filename.endswith(".pdf"):
                            text = extract_text_from_pdf(file_bytes)
                        else:
                            text = file_bytes.decode("utf-8", errors="ignore")
                            
                        # Extract metadata
                        meta = extract_metadata_from_text(text)
                        
                        st.session_state.extracted_title = meta.get("title", "")
                        st.session_state.extracted_abstract = meta.get("abstract", "")
                        st.session_state.extracted_keywords = meta.get("keywords", "")
                        st.session_state.extracted_domain = meta.get("domain", "")
                        st.session_state.extracted_full_text = text
                        st.session_state.last_uploaded_file = uploaded_file.name
                        st.success("✅ Metadata extracted! Review the populated fields below.")
                    except Exception as e:
                        st.error(f"❌ Failed to extract metadata: {e}")
        else:
            if "last_uploaded_file" in st.session_state:
                st.session_state.extracted_title = ""
                st.session_state.extracted_abstract = ""
                st.session_state.extracted_keywords = ""
                st.session_state.extracted_domain = ""
                st.session_state.extracted_full_text = ""
                del st.session_state.last_uploaded_file

        st.markdown("---")

        with st.form("form"):
            st.subheader("📝 Option 2: Review & Edit Paper Details")
            col1,col2 = st.columns([3,1])
            with col1:
                title = st.text_input("Research Title *",
                    value=st.session_state.extracted_title,
                    placeholder="e.g., Deep Learning for Medical Image Segmentation in Indian Hospitals")
            with col2:
                domain_options = [d for d in DOMAIN_LIST if not d.startswith("──")]
                default_idx = 0
                if st.session_state.extracted_domain in domain_options:
                    default_idx = domain_options.index(st.session_state.extracted_domain)
                domain = st.selectbox("Domain", domain_options, index=default_idx)
            abstract = st.text_area("Abstract (recommended)",
                value=st.session_state.extracted_abstract,
                placeholder="Describe your research objectives and methodology…",
                height=120)
            keywords = st.text_input("Keywords",
                value=st.session_state.extracted_keywords,
                placeholder="e.g., deep learning, medical imaging, CNN, India")
            submitted = st.form_submit_button("🚀 Analyse Research", use_container_width=True)

        if submitted:
            if not title.strip(): st.warning("⚠️ Please enter a research title.")
            else:
                with st.spinner("🔍 Running full analysis…"):
                    t0 = time.time()
                    result = run_analysis(retriever=retriever,title=title.strip(),abstract=abstract.strip(),
                                          keywords=keywords.strip(),domain=domain or "",
                                          top_k=top_k,check_plagiarism=check_plag,
                                          full_text=st.session_state.extracted_full_text or None)
                st.success(f"✅ Analysis complete in {time.time()-t0:.1f}s")
                render_results(result)

    with tab2:
        render_downloader_tab()


if __name__ == "__main__":
    main()