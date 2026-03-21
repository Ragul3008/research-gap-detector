"""
frontend/app_hf.py
------------------
Hugging Face Spaces version — runs WITHOUT a separate FastAPI server.
Tabs:
  1. 🔬 Research Gap Analyser
  2. 📥 Paper Downloader Agent (10+ sources)
"""

import sys
import re
import time
import json
import os
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import quote

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="AI Research Novelty & Gap Detector",
    page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size:2.2rem; font-weight:800;
        background:linear-gradient(90deg,#1a237e,#0d47a1,#1565c0);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .sub-header { font-size:1rem; color:#546e7a; margin-bottom:1.2rem; }
    .novelty-badge-high   { background:#2ecc71; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; }
    .novelty-badge-medium { background:#f39c12; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; }
    .novelty-badge-low    { background:#e74c3c; color:#fff; padding:6px 18px; border-radius:20px; font-weight:700; }
    .gap-card        { background:#fff3e0; border-left:4px solid #f57c00; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; font-size:0.95rem; }
    .paper-card      { background:#e8f5e9; border-left:4px solid #388e3c; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .title-suggestion{ background:#e3f2fd; border-left:4px solid #1976d2; padding:10px 14px; border-radius:8px; margin:8px 0; font-style:italic; color:#1a1a1a !important; }
    .metric-card     { background:#f8f9fa; border-left:4px solid #1565c0; padding:12px 16px; border-radius:8px; margin:6px 0; color:#1a1a1a !important; }
    .plag-safe       { background:#e8f5e9; border-left:4px solid #2ecc71; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .plag-low        { background:#e3f2fd; border-left:4px solid #3498db; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .plag-medium     { background:#fff8e1; border-left:4px solid #f39c12; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .plag-high       { background:#ffebee; border-left:4px solid #e74c3c; padding:12px 16px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .flagged-sent    { background:#fce4ec; border-left:4px solid #c62828; padding:8px 12px; border-radius:6px; margin:6px 0; color:#1a1a1a !important; font-size:0.88rem; }
    .phrase-match    { background:#f3e5f5; border-left:4px solid #7b1fa2; padding:8px 12px; border-radius:6px; margin:6px 0; color:#1a1a1a !important; font-size:0.88rem; }
    .hallucination-safe   { background:#e8f5e9; border-left:4px solid #2ecc71; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .hallucination-warn   { background:#fff8e1; border-left:4px solid #f39c12; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .hallucination-danger { background:#ffebee; border-left:4px solid #e74c3c; padding:10px 14px; border-radius:8px; margin:8px 0; color:#1a1a1a !important; }
    .flagged-claim   { background:#fce4ec; border-left:4px solid #c62828; padding:8px 12px; border-radius:6px; margin:6px 0; color:#1a1a1a !important; font-size:0.9rem; }
    .source-badge    { display:inline-block; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:700; margin-right:4px; }
    .arxiv           { background:#b71c1c; color:#fff; }
    .semanticscholar { background:#1565c0; color:#fff; }
    .ieee            { background:#00838f; color:#fff; }
    .springer        { background:#e65100; color:#fff; }
    .pubmed          { background:#1b5e20; color:#fff; }
    .core            { background:#4a148c; color:#fff; }
    .openalex        { background:#0d47a1; color:#fff; }
    .doaj            { background:#f57f17; color:#fff; }
    .europepmc       { background:#006064; color:#fff; }
    .eric            { background:#880e4f; color:#fff; }
    .biorxiv         { background:#3e2723; color:#fff; }
    .free-badge      { background:#2ecc71; color:#fff; padding:2px 8px; border-radius:8px; font-size:0.75rem; font-weight:700; }
    .paid-badge      { background:#e74c3c; color:#fff; padding:2px 8px; border-radius:8px; font-size:0.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# SECTION A — RESEARCH GAP ANALYSER
# ═══════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Loading AI models (first time only, ~2 min)…")
def load_engines():
    from retrieval.retriever import ResearchRetriever
    from config.settings import FAISS_INDEX_PATH, METADATA_PATH
    if not Path(FAISS_INDEX_PATH).exists():
        import subprocess
        subprocess.run(["python", "scripts/seed_data.py"],  check=True)
        subprocess.run(["python", "scripts/build_index.py"], check=True)
    return ResearchRetriever()


def run_analysis(retriever, title, abstract, keywords, domain, top_k, check_plagiarism):
    from models.similarity_engine import (
        compute_similarity_stats, rank_results,
        find_duplicate_risk, summarise_domain_overlap, summarise_year_distribution
    )
    from models.novelty_engine import calculate_novelty_score, get_novelty_suggestions
    from models.gap_engine import detect_research_gaps, suggest_improved_titles
    from models.explanation_engine import generate_explanation, enhance_gaps_with_llm
    from models.hallucination_detector import detect_hallucinations
    from models.plagiarism_detector import detect_plagiarism

    papers      = retriever.retrieve(title=title, abstract=abstract, keywords=keywords, top_k=top_k)
    sim_stats   = compute_similarity_stats(papers)
    ranked      = rank_results(papers, top_n=top_k)
    duplicates  = find_duplicate_risk(papers)
    domain_dist = summarise_domain_overlap(papers)
    year_dist   = summarise_year_distribution(papers)
    novelty     = calculate_novelty_score(papers, sim_stats, domain or "")
    novelty["suggestions"] = get_novelty_suggestions(novelty["label"], sim_stats, papers)
    gaps = detect_research_gaps(retrieved_papers=papers, user_title=title, user_keywords=keywords)
    gaps["gap_statements"] = enhance_gaps_with_llm(gap_statements=gaps.get("gap_statements",[]), retrieved_papers=papers, user_title=title)
    title_suggestions = suggest_improved_titles(original_title=title, gap_dimensions=gaps.get("gap_dimensions",{}))
    full_analysis = {"input": {"title":title,"abstract":abstract,"keywords":keywords,"domain":domain},
                     "novelty":novelty,"gaps":gaps,"similarity_stats":sim_stats,
                     "similar_papers":ranked,"title_suggestions":title_suggestions}
    explanation   = generate_explanation(full_analysis)
    hallucination = detect_hallucinations(llm_output=explanation, retrieved_papers=papers, novelty_data=novelty)
    plagiarism    = {}
    if check_plagiarism:
        plagiarism = detect_plagiarism(user_title=title, user_abstract=abstract,
                                       user_keywords=keywords, retrieved_papers=papers, similarity_stats=sim_stats)
    return {
        "status":"success",
        "input":{"title":title,"abstract":abstract,"keywords":keywords,"domain":domain},
        "novelty":{"label":novelty["label"],"percentage":novelty["percentage"],"color":novelty["color"],
                   "description":novelty["description"],"sub_scores":novelty.get("sub_scores",{}),"suggestions":novelty["suggestions"]},
        "similarity":{"stats":sim_stats,"domain_dist":domain_dist,"year_dist":year_dist,
                      "duplicate_risk":len(duplicates)>0,"duplicates":duplicates},
        "similar_papers":ranked,"gaps":gaps,"title_suggestions":title_suggestions,"explanation":explanation,
        "hallucination":{"grounding_score":hallucination.get("grounding_score",100),"label":hallucination.get("label",""),
                         "color":hallucination.get("color","#2ecc71"),"hallucination_count":hallucination.get("hallucination_count",0),
                         "total_claims":hallucination.get("total_claims",0),"flagged_claims":hallucination.get("flagged_claims",[]),
                         "warnings":hallucination.get("warnings",[]),"summary":hallucination.get("summary","")},
        "plagiarism":plagiarism,
    }


def gauge(value, title, suffix, color):
    fig = go.Figure(go.Indicator(mode="gauge+number",value=value,title={"text":title,"font":{"size":13}},
        number={"suffix":suffix,"font":{"size":28,"color":color}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":color},
               "steps":[{"range":[0,40],"color":"#ffebee"},{"range":[40,70],"color":"#fff8e1"},{"range":[70,100],"color":"#e8f5e9"}]}))
    fig.update_layout(height=220,margin=dict(l=15,r=15,t=55,b=10),paper_bgcolor="rgba(0,0,0,0)")
    return fig

def plagiarism_gauge(score, color):
    fig = go.Figure(go.Indicator(mode="gauge+number",value=score,
        title={"text":"<b>Plagiarism Score</b><br><small>Lower is better</small>","font":{"size":13}},
        number={"suffix":"%","font":{"size":28,"color":color}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":color},
               "steps":[{"range":[0,15],"color":"#e8f5e9"},{"range":[15,40],"color":"#e3f2fd"},
                        {"range":[40,70],"color":"#fff8e1"},{"range":[70,100],"color":"#ffebee"}]}))
    fig.update_layout(height=220,margin=dict(l=15,r=15,t=55,b=10),paper_bgcolor="rgba(0,0,0,0)")
    return fig

def sim_bar(papers):
    titles = [p["title"][:50]+"…" if len(p["title"])>50 else p["title"] for p in papers]
    scores = [p["similarity_pct"] for p in papers]
    colors = ["#e53935" if s>=75 else "#fb8c00" if s>=55 else "#43a047" for s in scores]
    fig = go.Figure(go.Bar(x=scores,y=titles,orientation="h",marker_color=colors,
                           text=[f"{s:.1f}%" for s in scores],textposition="outside"))
    fig.update_layout(title="Similarity Scores",xaxis_range=[0,110],height=max(250,len(papers)*40),
                      margin=dict(l=10,r=40,t=40,b=20),paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#f8f9fa",yaxis=dict(automargin=True))
    return fig

def domain_pie(dd):
    fig = px.pie(names=list(dd.keys()),values=list(dd.values()),title="Domain Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set3,hole=0.35)
    fig.update_layout(height=260,margin=dict(l=5,r=5,t=40,b=5),paper_bgcolor="rgba(0,0,0,0)")
    return fig

def year_bar(yd):
    years  = [str(y) for y in sorted(int(k) for k in yd.keys()) if y>0]
    counts = [yd.get(y,yd.get(int(y),0)) for y in years]
    fig = go.Figure(go.Bar(x=years,y=counts,marker_color="#1565c0",text=counts,textposition="outside"))
    fig.update_layout(title="Publication Years",height=230,margin=dict(l=10,r=10,t=40,b=20),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#f8f9fa")
    return fig


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
    m1.metric("Max Similarity",f"{stats.get('max_sim',0)*100:.1f}%")
    m2.metric("Mean Similarity",f"{stats.get('mean_sim',0)*100:.1f}%")
    m3.metric("Papers Retrieved",len(papers)); m4.metric("Plagiarism Score",f"{p_score:.1f}%")
    st.markdown(f'<div class="metric-card">{novelty.get("description","")}</div>',unsafe_allow_html=True)
    if sim.get("duplicate_risk"): st.error(f"🚨 **Duplicate Risk!** {len(sim.get('duplicates',[]))} paper(s) ≥90% similar.")

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
                    st.info("👉 Switch to the **📥 Paper Downloader** tab to download this paper!")
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
    hc1.metric("Grounding Score",f"{h_score}/100"); hc2.metric("Total Claims",h_total)
    hc3.metric("Grounded",h_total-h_count); hc4.metric("Flagged",h_count)
    h_css="hallucination-safe" if h_score>=80 else "hallucination-warn" if h_score>=60 else "hallucination-danger"
    st.markdown(f'<div class="{h_css}"><b>{hallucination.get("label","")}</b> — Grounding Score: {h_score}/100</div>',unsafe_allow_html=True)
    if hallucination.get("flagged_claims"):
        with st.expander(f"🔴 {len(hallucination['flagged_claims'])} Flagged Claim(s)"):
            for fc in hallucination["flagged_claims"]:
                sev=fc.get("severity","LOW"); icon="🔴" if sev=="HIGH" else "🟡" if sev=="MEDIUM" else "🟢"
                st.markdown(f'<div class="flagged-claim">{icon} <b>[{sev}]</b> {fc.get("issue","")}<br><small><i>"{fc.get("claim","")[:120]}…"</i></small></div>',unsafe_allow_html=True)
    else: st.success("✅ No hallucinations detected.")

    st.markdown("---")
    st.download_button(label="📥 Download Full Analysis (JSON)",data=json.dumps(result,indent=2,ensure_ascii=False),
                       file_name="research_analysis.json",mime="application/json",use_container_width=True)


# ═══════════════════════════════════════════════════════
# SECTION B — PAPER DOWNLOADER (10+ SOURCES)
# ═══════════════════════════════════════════════════════

UNPAYWALL_EMAIL = "raguljayakumar34@gmail.com"
DL_HEADERS      = {"User-Agent": "Mozilla/5.0 (ResearchDownloader/1.0; mailto:raguljayakumar34@gmail.com)"}


def get_unpaywall_pdf(doi: str):
    if not doi: return None
    try:
        resp = requests.get(f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}", headers=DL_HEADERS, timeout=10)
        if resp.status_code == 200:
            best = resp.json().get("best_oa_location")
            if best: return best.get("url_for_pdf") or best.get("url")
    except: pass
    return None


def make_result(source, title, authors, year, abstract, pdf_url, page_url, doi=None):
    return {"source":source,"title":title,"authors":authors,"year":str(year),
            "abstract":(abstract or "")[:300]+"…","pdf_url":pdf_url,
            "page_url":page_url,"free":pdf_url is not None,"doi":doi}


# ── 1. arXiv ─────────────────────────────────
def search_arxiv(title, n=5):
    results = []
    try:
        query = quote(f'ti:"{title}"')
        resp  = requests.get(f"http://export.arxiv.org/api/query?search_query={query}&max_results={n}&sortBy=relevance", headers=DL_HEADERS, timeout=15)
        root  = ET.fromstring(resp.content)
        ns    = {"atom":"http://www.w3.org/2005/Atom"}
        for e in root.findall("atom:entry", ns):
            pid  = e.find("atom:id",ns).text.strip()
            aid  = pid.split("/abs/")[-1]
            auths= [a.find("atom:name",ns).text for a in e.findall("atom:author",ns)]
            results.append(make_result("arXiv",
                e.find("atom:title",ns).text.strip().replace("\n"," "),
                ", ".join(auths[:3])+(" et al." if len(auths)>3 else ""),
                e.find("atom:published",ns).text[:4],
                e.find("atom:summary",ns).text.strip(),
                f"https://arxiv.org/pdf/{aid}.pdf", pid))
    except Exception as ex: st.warning(f"arXiv: {ex}")
    return results


# ── 2. Semantic Scholar ───────────────────────
def search_semantic_scholar(title, n=5):
    results = []
    try:
        resp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                            params={"query":title,"limit":n,"fields":"title,authors,year,abstract,openAccessPdf,externalIds,url"},
                            headers=DL_HEADERS, timeout=15)
        for p in resp.json().get("data",[]):
            oa  = p.get("openAccessPdf"); pdf = oa.get("url") if oa else None
            doi = p.get("externalIds",{}).get("DOI")
            aut = [a["name"] for a in p.get("authors",[])]
            results.append(make_result("Semantic Scholar",p.get("title",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                p.get("year","N/A"),p.get("abstract"),pdf,p.get("url",""),doi))
    except Exception as ex: st.warning(f"Semantic Scholar: {ex}")
    return results


# ── 3. IEEE Xplore ────────────────────────────
def search_ieee(title, n=5):
    results = []
    try:
        resp = requests.get(f"https://ieeexplore.ieee.org/rest/search",
                            params={"queryText":title,"newsearch":"true","pageNumber":1,"rowsPerPage":n},
                            headers={**DL_HEADERS,"Referer":"https://ieeexplore.ieee.org"}, timeout=15)
        for p in resp.json().get("records",[]):
            doi = p.get("doi",""); pdf = get_unpaywall_pdf(doi) if doi else None
            aut = [a.get("preferredName","") for a in p.get("authors",[])]
            results.append(make_result("IEEE",p.get("articleTitle",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                p.get("publicationYear","N/A"),p.get("abstract"),pdf,
                f"https://ieeexplore.ieee.org/document/{p.get('articleNumber','')}",doi))
    except Exception as ex: st.warning(f"IEEE: {ex}")
    return results


# ── 4. Springer ──────────────────────────────
def search_springer(title, n=5):
    results = []
    try:
        resp = requests.get("https://api.springernature.com/meta/v2/json",
                            params={"q":f'title:"{title}"',"p":n,"api_key":"b374210d2db59a18aa3bfc1eac3e1876"},
                            headers=DL_HEADERS, timeout=15)
        for r in resp.json().get("records",[]):
            doi = r.get("doi",""); pdf = get_unpaywall_pdf(doi) if doi else None
            aut = [c.get("creator","") for c in r.get("creators",[])]
            results.append(make_result("Springer",r.get("title",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                r.get("publicationDate","")[:4],r.get("abstract"),pdf,
                f"https://doi.org/{doi}" if doi else "",doi))
    except Exception as ex: st.warning(f"Springer: {ex}")
    return results


# ── 5. PubMed Central ────────────────────────
def search_pubmed(title, n=5):
    results = []
    try:
        # Search for IDs
        search = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                              params={"db":"pmc","term":title,"retmax":n,"retmode":"json"},
                              headers=DL_HEADERS, timeout=15)
        ids = search.json().get("esearchresult",{}).get("idlist",[])
        if not ids: return results
        # Fetch details
        fetch = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                             params={"db":"pmc","id":",".join(ids),"retmode":"json"},
                             headers=DL_HEADERS, timeout=15)
        data = fetch.json().get("result",{})
        for pid in ids:
            p = data.get(pid,{})
            if not p: continue
            pmcid   = p.get("articleids",[{}])
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pid}/pdf/"
            aut     = [a.get("name","") for a in p.get("authors",[])]
            results.append(make_result("PubMed Central",p.get("title",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                p.get("pubdate","")[:4],"See PubMed Central for abstract.",
                pdf_url,f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pid}/"))
    except Exception as ex: st.warning(f"PubMed: {ex}")
    return results


# ── 6. CORE.ac.uk ────────────────────────────
def search_core(title, n=5):
    results = []
    try:
        resp = requests.get("https://api.core.ac.uk/v3/search/works",
                            params={"q":title,"limit":n},
                            headers={**DL_HEADERS,"Authorization":"Bearer "},
                            timeout=15)
        for p in resp.json().get("results",[]):
            pdf_url = p.get("downloadUrl") or p.get("fullTextIdentifier")
            doi     = p.get("doi")
            aut     = [a.get("name","") for a in p.get("authors",[])]
            results.append(make_result("CORE",p.get("title",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                str(p.get("yearPublished","N/A")),p.get("abstract"),
                pdf_url,p.get("sourceFulltextUrls",[p.get("id","")])[0] if p.get("sourceFulltextUrls") else "",doi))
    except Exception as ex: st.warning(f"CORE: {ex}")
    return results


# ── 7. OpenAlex ──────────────────────────────
def search_openalex(title, n=5):
    results = []
    try:
        resp = requests.get("https://api.openalex.org/works",
                            params={"search":title,"per-page":n,"mailto":UNPAYWALL_EMAIL},
                            headers=DL_HEADERS, timeout=15)
        for p in resp.json().get("results",[]):
            oa      = p.get("open_access",{})
            pdf_url = oa.get("oa_url") if oa.get("is_oa") else None
            doi     = p.get("doi","").replace("https://doi.org/","") if p.get("doi") else None
            aut     = [a.get("author",{}).get("display_name","") for a in p.get("authorships",[])]
            ab_inv  = p.get("abstract_inverted_index")
            abstract= ""
            if ab_inv:
                words = {v:k for k,vals in ab_inv.items() for v in vals}
                abstract = " ".join(words[i] for i in sorted(words))
            results.append(make_result("OpenAlex",p.get("title",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                p.get("publication_year","N/A"),abstract,pdf_url,
                p.get("id",""),doi))
    except Exception as ex: st.warning(f"OpenAlex: {ex}")
    return results


# ── 8. DOAJ ──────────────────────────────────
def search_doaj(title, n=5):
    results = []
    try:
        resp = requests.get("https://doaj.org/api/search/articles",
                            params={"q":title,"pageSize":n},
                            headers=DL_HEADERS, timeout=15)
        for p in resp.json().get("results",[]):
            bib     = p.get("bibjson",{})
            links   = bib.get("link",[])
            pdf_url = next((l["url"] for l in links if l.get("type")=="fulltext"),None)
            doi     = next((i["id"] for i in bib.get("identifier",[]) if i.get("type")=="doi"),None)
            aut     = [a.get("name","") for a in bib.get("author",[])]
            results.append(make_result("DOAJ",bib.get("title",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                str(bib.get("year","N/A")),bib.get("abstract"),
                pdf_url,f"https://doaj.org/article/{p.get('id','')}",doi))
    except Exception as ex: st.warning(f"DOAJ: {ex}")
    return results


# ── 9. Europe PMC ────────────────────────────
def search_europepmc(title, n=5):
    results = []
    try:
        resp = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                            params={"query":title,"resultType":"core","pageSize":n,"format":"json"},
                            headers=DL_HEADERS, timeout=15)
        for p in resp.json().get("resultList",{}).get("result",[]):
            is_oa   = p.get("isOpenAccess","N") == "Y"
            pmcid   = p.get("pmcid","")
            pdf_url = f"https://europepmc.org/articles/{pmcid}/pdf" if pmcid and is_oa else None
            doi     = p.get("doi")
            aut     = p.get("authorString","")
            results.append(make_result("Europe PMC",p.get("title",""),aut,
                str(p.get("pubYear","N/A")),p.get("abstractText"),pdf_url,
                f"https://europepmc.org/article/MED/{p.get('pmid','')}",doi))
    except Exception as ex: st.warning(f"Europe PMC: {ex}")
    return results


# ── 10. ERIC (Education) ─────────────────────
def search_eric(title, n=5):
    results = []
    try:
        resp = requests.get("https://api.ies.ed.gov/eric/",
                            params={"search":title,"format":"json","rows":n},
                            headers=DL_HEADERS, timeout=15)
        for p in resp.json().get("response",{}).get("docs",[]):
            eric_id = p.get("id","")
            pdf_url = f"https://files.eric.ed.gov/fulltext/{eric_id}.pdf" if eric_id else None
            aut     = p.get("author",[])
            results.append(make_result("ERIC",p.get("title",""),
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                str(p.get("publicationdateyear","N/A")),p.get("description"),
                pdf_url,f"https://eric.ed.gov/?id={eric_id}"))
    except Exception as ex: st.warning(f"ERIC: {ex}")
    return results


# ── 11. bioRxiv / medRxiv ────────────────────
def search_biorxiv(title, n=5):
    results = []
    try:
        resp = requests.get("https://api.biorxiv.org/details/biorxiv/2000-01-01/2099-12-31/0",
                            headers=DL_HEADERS, timeout=15)
        # Use CrossRef to search biorxiv
        resp = requests.get("https://api.crossref.org/works",
                            params={"query.title":title,"filter":"type:posted-content","rows":n,
                                    "mailto":UNPAYWALL_EMAIL},
                            headers=DL_HEADERS, timeout=15)
        for p in resp.json().get("message",{}).get("items",[]):
            doi     = p.get("DOI","")
            pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf" if doi else None
            aut     = [f"{a.get('given','')} {a.get('family','')}".strip() for a in p.get("author",[])]
            pub     = p.get("posted",{}).get("date-parts",[[""]])[0]
            results.append(make_result("bioRxiv",p.get("title",[""])[0],
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                str(pub[0]) if pub else "N/A","Preprint — see link for abstract.",
                pdf_url,f"https://doi.org/{doi}" if doi else "",doi))
    except Exception as ex: st.warning(f"bioRxiv: {ex}")
    return results


# ── 12. CrossRef ─────────────────────────────
def search_crossref(title, n=5):
    results = []
    try:
        resp = requests.get("https://api.crossref.org/works",
                            params={"query.title":title,"rows":n,"mailto":UNPAYWALL_EMAIL},
                            headers=DL_HEADERS, timeout=15)
        for p in resp.json().get("message",{}).get("items",[]):
            doi     = p.get("DOI","")
            pdf_url = get_unpaywall_pdf(doi) if doi else None
            aut     = [f"{a.get('given','')} {a.get('family','')}".strip() for a in p.get("author",[])]
            pub     = p.get("published",{}).get("date-parts",[[""]])[0]
            ab_list = p.get("abstract","")
            results.append(make_result("CrossRef",p.get("title",[""])[0] if p.get("title") else "",
                ", ".join(aut[:3])+(" et al." if len(aut)>3 else ""),
                str(pub[0]) if pub else "N/A",re.sub(r'<[^>]+>',"",ab_list),
                pdf_url,f"https://doi.org/{doi}" if doi else "",doi))
    except Exception as ex: st.warning(f"CrossRef: {ex}")
    return results


# ── Download function ────────────────────────
def download_paper(pdf_url, title, source, folder):
    try:
        safe     = re.sub(r'[\\/*?:"<>|]',"",title)[:80].strip()
        filename = f"[{source}] {safe}.pdf"
        filepath = folder / filename
        if filepath.exists(): return True, str(filepath)
        resp = requests.get(pdf_url, headers=DL_HEADERS, timeout=30, stream=True)
        if resp.status_code != 200: return False, f"HTTP {resp.status_code}"
        with open(filepath,"wb") as f:
            for chunk in resp.iter_content(8192): f.write(chunk)
        if filepath.stat().st_size < 5000:
            filepath.unlink(); return False, "File too small — not a valid PDF"
        return True, str(filepath)
    except Exception as e: return False, str(e)


# ── All sources config ────────────────────────
ALL_SOURCES = {
    "arXiv":            search_arxiv,
    "Semantic Scholar": search_semantic_scholar,
    "IEEE":             search_ieee,
    "Springer":         search_springer,
    "PubMed Central":   search_pubmed,
    "CORE":             search_core,
    "OpenAlex":         search_openalex,
    "DOAJ":             search_doaj,
    "Europe PMC":       search_europepmc,
    "ERIC":             search_eric,
    "bioRxiv":          search_biorxiv,
    "CrossRef":         search_crossref,
}


def render_downloader_tab():
    st.markdown("### 📥 Research Paper Downloader Agent")
    st.markdown("Search across **12 sources** — arXiv · Semantic Scholar · IEEE · Springer · PubMed · CORE · OpenAlex · DOAJ · Europe PMC · ERIC · bioRxiv · CrossRef")

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        prefill = st.session_state.pop("dl_search_title","")
        title   = st.text_input("🔍 Enter Paper Title", value=prefill, placeholder="e.g., Attention Is All You Need")
    with col2:
        max_results = st.number_input("Results/source", min_value=1, max_value=10, value=3, step=1)
    with col3:
        folder_name = st.text_input("Save folder", value="downloaded_papers")

    sources = st.multiselect("🌐 Select Sources to Search",
                             list(ALL_SOURCES.keys()),
                             default=["arXiv","Semantic Scholar","IEEE","Springer","OpenAlex","CORE","CrossRef"])

    dl_folder = Path(folder_name)
    dl_folder.mkdir(exist_ok=True)

    search_btn = st.button("🔍 Search All Selected Sources", use_container_width=True, type="primary")

    if search_btn and title.strip():
        all_results = []
        prog = st.progress(0)
        for idx, src in enumerate(sources):
            st.caption(f"Searching {src}…")
            fn = ALL_SOURCES.get(src)
            if fn:
                try: all_results += fn(title, max_results)
                except: pass
            prog.progress((idx+1)/len(sources))

        # Deduplicate by title
        seen, unique = set(), []
        for r in all_results:
            key = r["title"].lower()[:50]
            if key not in seen and r["title"]:
                seen.add(key); unique.append(r)

        st.session_state["dl_results"] = unique
        st.session_state["dl_folder"]  = folder_name

    results   = st.session_state.get("dl_results",[])
    dl_folder = Path(st.session_state.get("dl_folder", folder_name))

    if results:
        free_count = sum(1 for r in results if r["free"])
        st.success(f"✅ Found **{len(results)}** papers across all sources — **{free_count}** with free PDF download")

        col1, col2 = st.columns(2)
        with col1: show_free = st.checkbox("Show free PDFs only", value=False)
        with col2:
            src_filter = st.multiselect("Filter by source", list(set(r["source"] for r in results)),
                                        default=list(set(r["source"] for r in results)))

        filtered = [r for r in results if (not show_free or r["free"]) and r["source"] in src_filter]
        st.markdown(f"**Showing {len(filtered)} results**")

        for i, paper in enumerate(filtered):
            free_html = '<span class="free-badge">✅ FREE PDF</span>' if paper["free"] else '<span class="paid-badge">🔒 Paywalled</span>'
            src_class = paper["source"].lower().replace(" ","").replace("central","")

            with st.expander(f"[{paper['source']}] {paper['title'][:80]}{'…' if len(paper['title'])>80 else ''} {'✅' if paper['free'] else '🔒'}"):
                st.markdown(f'<span class="source-badge {src_class}">{paper["source"]}</span> {free_html}', unsafe_allow_html=True)
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**Authors:** {paper['authors'] or 'N/A'}")
                c2.markdown(f"**Year:** {paper['year']}")
                c3.markdown(f"**DOI:** {paper['doi'] or 'N/A'}")
                st.markdown(f"**Abstract:** {paper['abstract']}")

                bc1, bc2 = st.columns(2)
                with bc1:
                    if paper.get("page_url"):
                        st.link_button("🌐 Open Paper Page", paper["page_url"], use_container_width=True)
                with bc2:
                    if paper["free"] and paper["pdf_url"]:
                        if st.button("📥 Download PDF", key=f"dl_{i}", use_container_width=True):
                            with st.spinner("Downloading…"):
                                ok, res = download_paper(paper["pdf_url"], paper["title"], paper["source"], dl_folder)
                                if ok: st.success(f"✅ Saved: `{res}`"); st.balloons()
                                else:  st.error(f"❌ Failed: {res}")
                    elif paper.get("page_url"):
                        st.link_button("🔒 Publisher Site", paper["page_url"], use_container_width=True)

        free_papers = [r for r in filtered if r["free"] and r["pdf_url"]]
        if free_papers:
            st.markdown("---")
            if st.button(f"📥 Download All {len(free_papers)} Free Papers at Once", type="primary", use_container_width=True):
                prog = st.progress(0); ok_count = 0
                for idx, paper in enumerate(free_papers):
                    with st.spinner(f"Downloading {idx+1}/{len(free_papers)}: {paper['title'][:50]}…"):
                        ok, res = download_paper(paper["pdf_url"], paper["title"], paper["source"], dl_folder)
                        if ok: ok_count += 1; st.success(f"✅ {paper['title'][:60]}")
                        else:  st.warning(f"⚠️ {paper['title'][:50]} — {res}")
                        prog.progress((idx+1)/len(free_papers)); time.sleep(0.3)
                st.success(f"🎉 Downloaded {ok_count}/{len(free_papers)} papers to `{dl_folder}/`")

    # Downloaded files panel
    st.markdown("---")
    st.markdown("### 📂 Your Downloaded Papers")
    files = list(dl_folder.glob("*.pdf"))
    if files:
        st.success(f"{len(files)} paper(s) saved in `{dl_folder}/`")
        for f in files:
            size_kb = f.stat().st_size // 1024
            c1,c2 = st.columns([5,1])
            c1.markdown(f"📄 `{f.name}` — {size_kb} KB")
            if c2.button("🗑️", key=f"del_{f.name}", help="Delete"): f.unlink(); st.rerun()
    else:
        st.info("No papers downloaded yet. Search above to get started!")

    with st.expander("📊 Source Reference Table"):
        st.markdown("""
| # | Source | Field | Free PDF |
|---|--------|-------|----------|
| 1 | **arXiv** | CS, Physics, Math, Biology | ✅ Always |
| 2 | **Semantic Scholar** | All fields | ✅ Open access |
| 3 | **IEEE Xplore** | Engineering & CS | ✅ Via Unpaywall |
| 4 | **Springer** | All fields | ✅ Via Unpaywall |
| 5 | **PubMed Central** | Biomedical & Life Science | ✅ Always |
| 6 | **CORE** | All fields (largest OA) | ✅ Always |
| 7 | **OpenAlex** | All fields | ✅ Open access |
| 8 | **DOAJ** | Open access journals | ✅ Always |
| 9 | **Europe PMC** | Life sciences | ✅ Open access |
| 10 | **ERIC** | Education research | ✅ Always |
| 11 | **bioRxiv** | Biology preprints | ✅ Always |
| 12 | **CrossRef** | All fields | ✅ Via Unpaywall |
        """)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.markdown("## 🔬 Research Detector Agent")
        st.markdown("**For PhD Scholars — India**")
        st.divider()
        st.markdown("### ⚙️ Analysis Settings")
        top_k = int(st.number_input("Number of papers to search", min_value=1, value=10, step=1,
                                    help="Enter how many papers to retrieve (no limit)"))
        check_plag = st.toggle("Plagiarism Check", value=True)
        st.divider()
        st.markdown("### 📖 What This Tool Does")
        st.markdown("""
- 📊 **Novelty Score** — Is your topic original?
- 🔍 **Plagiarism Check** — Similarity to existing papers
- 🔬 **Gap Detection** — What's missing in literature?
- 🤖 **AI Report** — Gemini-powered explanation
- 🛡️ **Hallucination Check** — AI accuracy verification
- 📥 **Paper Downloader** — 12 sources, download instantly
        """)
        st.divider()
        st.markdown("### 🎓 Supported Departments")
        st.markdown("Arts & Science · English · Engineering")
        st.caption("© 2024 · AI Research Tools · India")

    st.markdown('<p class="main-header">🔬 AI Research Novelty & Gap Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">For Arts, Science & Engineering PhD Scholars in India · Powered by AI + Plagiarism + Hallucination Check + 12-Source Paper Downloader</p>', unsafe_allow_html=True)

    try:
        retriever = load_engines()
        st.sidebar.success(f"✅ {retriever.get_corpus_size()} papers loaded")
    except Exception as e:
        st.error(f"❌ Failed to load engines: {e}"); return

    tab1, tab2 = st.tabs(["🔬 Research Gap Analyser", "📥 Paper Downloader (12 Sources)"])

    with tab1:
        with st.form("form"):
            st.subheader("📝 Enter Your Research Details")
            col1, col2 = st.columns([3,1])
            with col1:
                title = st.text_input("Research Title *", placeholder="e.g., Deep Learning for Medical Image Segmentation in Indian Hospitals")
            with col2:
                domain = st.selectbox("Domain", ["","Sociology","History","Political Science","Economics",
                    "Psychology","Environmental Science","Literature","Philosophy","Anthropology","Education",
                    "Geography","Women Studies","Commerce","Linguistics","Botany","Zoology","Chemistry",
                    "Physics","Mathematics","Statistics","English Literature","English Language Teaching",
                    "Postcolonial Studies","Comparative Literature","Translation Studies",
                    "Computer Science and Engineering","Electronics and Communication Engineering",
                    "Electrical Engineering","Mechanical Engineering","Civil Engineering","Chemical Engineering",
                    "Artificial Intelligence","Machine Learning","Deep Learning","Internet of Things",
                    "Cybersecurity","Data Science","Robotics and Automation","VLSI Design",
                    "Power Systems Engineering","Renewable Energy Engineering","Structural Engineering",
                    "Environmental Engineering","Nanotechnology","Materials Science","Biomedical Engineering"])
            abstract = st.text_area("Abstract (recommended)", placeholder="Describe your research objectives and methodology…", height=120)
            keywords = st.text_input("Keywords", placeholder="e.g., deep learning, medical imaging, CNN, India")
            submitted = st.form_submit_button("🚀 Analyse Research", use_container_width=True)

        if submitted:
            if not title.strip(): st.warning("⚠️ Please enter a research title.")
            else:
                with st.spinner("🔍 Running full analysis…"):
                    t0 = time.time()
                    result = run_analysis(retriever=retriever, title=title.strip(),
                                          abstract=abstract.strip(), keywords=keywords.strip(),
                                          domain=domain or "", top_k=top_k, check_plagiarism=check_plag)
                st.success(f"✅ Analysis complete in {time.time()-t0:.1f}s")
                render_results(result)

    with tab2:
        render_downloader_tab()


if __name__ == "__main__":
    main()