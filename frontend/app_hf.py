"""
frontend/app_hf.py
------------------
Hugging Face Spaces version — runs WITHOUT a separate FastAPI server.
Everything runs in a single Streamlit process.
Perfect for student access via a public URL.
"""

import sys
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="AI Research Novelty & Gap Detector",
    page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────
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
    .llm-badge       { background:#6200ea; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load all engines ONCE (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Loading AI models (first time only, ~2 min)…")
def load_engines():
    """Load all ML engines once and cache them."""
    from retrieval.retriever import ResearchRetriever
    from config.settings import FAISS_INDEX_PATH, METADATA_PATH

    # Build index if not exists
    if not Path(FAISS_INDEX_PATH).exists():
        st.info("📦 Building index for the first time…")
        import subprocess
        subprocess.run(["python", "scripts/seed_data.py"],  check=True)
        subprocess.run(["python", "scripts/build_index.py"], check=True)

    retriever = ResearchRetriever()
    return retriever


def run_analysis(
    retriever,
    title: str,
    abstract: str,
    keywords: str,
    domain: str,
    top_k: int,
    check_plagiarism: bool,
) -> Dict[str, Any]:
    """Run the full pipeline directly (no HTTP call needed)."""
    from models.similarity_engine import (
        compute_similarity_stats, rank_results,
        find_duplicate_risk, summarise_domain_overlap,
        summarise_year_distribution
    )
    from models.novelty_engine import calculate_novelty_score, get_novelty_suggestions
    from models.gap_engine import detect_research_gaps, suggest_improved_titles
    from models.explanation_engine import generate_explanation, enhance_gaps_with_llm
    from models.hallucination_detector import detect_hallucinations
    from models.plagiarism_detector import detect_plagiarism

    # 1. Retrieve
    papers = retriever.retrieve(
        title=title, abstract=abstract,
        keywords=keywords, top_k=top_k
    )

    # 2. Similarity
    sim_stats   = compute_similarity_stats(papers)
    ranked      = rank_results(papers, top_n=top_k)
    duplicates  = find_duplicate_risk(papers)
    domain_dist = summarise_domain_overlap(papers)
    year_dist   = summarise_year_distribution(papers)

    # 3. Novelty
    novelty = calculate_novelty_score(papers, sim_stats, domain or "")
    novelty["suggestions"] = get_novelty_suggestions(novelty["label"], sim_stats, papers)

    # 4. Gaps
    gaps = detect_research_gaps(retrieved_papers=papers,
                                user_title=title, user_keywords=keywords)
    gaps["gap_statements"] = enhance_gaps_with_llm(
        gap_statements=gaps.get("gap_statements", []),
        retrieved_papers=papers, user_title=title
    )

    # 5. Title suggestions
    title_suggestions = suggest_improved_titles(
        original_title=title,
        gap_dimensions=gaps.get("gap_dimensions", {})
    )

    # 6. Explanation
    full_analysis = {
        "input":             {"title": title, "abstract": abstract,
                              "keywords": keywords, "domain": domain},
        "novelty":           novelty,
        "gaps":              gaps,
        "similarity_stats":  sim_stats,
        "similar_papers":    ranked,
        "title_suggestions": title_suggestions,
    }
    explanation = generate_explanation(full_analysis)

    # 7. Hallucination
    hallucination = detect_hallucinations(
        llm_output=explanation,
        retrieved_papers=papers,
        novelty_data=novelty,
    )

    # 8. Plagiarism
    plagiarism = {}
    if check_plagiarism:
        plagiarism = detect_plagiarism(
            user_title=title, user_abstract=abstract,
            user_keywords=keywords,
            retrieved_papers=papers, similarity_stats=sim_stats,
        )

    return {
        "status": "success",
        "input":  {"title": title, "abstract": abstract,
                   "keywords": keywords, "domain": domain},
        "novelty": {
            "label":       novelty["label"],
            "percentage":  novelty["percentage"],
            "color":       novelty["color"],
            "description": novelty["description"],
            "sub_scores":  novelty.get("sub_scores", {}),
            "suggestions": novelty["suggestions"],
        },
        "similarity": {
            "stats":          sim_stats,
            "domain_dist":    domain_dist,
            "year_dist":      year_dist,
            "duplicate_risk": len(duplicates) > 0,
            "duplicates":     duplicates,
        },
        "similar_papers":    ranked,
        "gaps":              gaps,
        "title_suggestions": title_suggestions,
        "explanation":       explanation,
        "hallucination": {
            "grounding_score":     hallucination.get("grounding_score", 100),
            "label":               hallucination.get("label", ""),
            "color":               hallucination.get("color", "#2ecc71"),
            "hallucination_count": hallucination.get("hallucination_count", 0),
            "total_claims":        hallucination.get("total_claims", 0),
            "flagged_claims":      hallucination.get("flagged_claims", []),
            "warnings":            hallucination.get("warnings", []),
            "summary":             hallucination.get("summary", ""),
        },
        "plagiarism": plagiarism,
    }


# ─── Charts ──────────────────────────────────
def gauge(value, title, suffix, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 13}},
        number={"suffix": suffix, "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": color},
            "steps": [
                {"range": [0,  40],  "color": "#ffebee"},
                {"range": [40, 70],  "color": "#fff8e1"},
                {"range": [70, 100], "color": "#e8f5e9"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=15,r=15,t=55,b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plagiarism_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "<b>Plagiarism Score</b><br><small>Lower is better</small>",
               "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": color},
            "steps": [
                {"range": [0,  15],  "color": "#e8f5e9"},
                {"range": [15, 40],  "color": "#e3f2fd"},
                {"range": [40, 70],  "color": "#fff8e1"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
        }
    ))
    fig.update_layout(height=220, margin=dict(l=15,r=15,t=55,b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def sim_bar(papers):
    titles = [p["title"][:50]+"…" if len(p["title"])>50 else p["title"] for p in papers]
    scores = [p["similarity_pct"] for p in papers]
    colors = ["#e53935" if s>=75 else "#fb8c00" if s>=55 else "#43a047" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=titles, orientation="h",
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores], textposition="outside",
    ))
    fig.update_layout(title="Similarity Scores", xaxis_range=[0,110],
                      height=max(250,len(papers)*40),
                      margin=dict(l=10,r=40,t=40,b=20),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#f8f9fa",
                      yaxis=dict(automargin=True))
    return fig


def domain_pie(dd):
    fig = px.pie(names=list(dd.keys()), values=list(dd.values()),
                 title="Domain Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set3, hole=0.35)
    fig.update_layout(height=260, margin=dict(l=5,r=5,t=40,b=5),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def year_bar(yd):
    years  = [str(y) for y in sorted(int(k) for k in yd.keys()) if y > 0]
    counts = [yd.get(y, yd.get(int(y), 0)) for y in years]
    fig = go.Figure(go.Bar(x=years, y=counts, marker_color="#1565c0",
                           text=counts, textposition="outside"))
    fig.update_layout(title="Publication Years", height=230,
                      margin=dict(l=10,r=10,t=40,b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f8f9fa")
    return fig


def donut(labels, values, colors, title):
    fig = go.Figure(go.Pie(labels=labels, values=values,
                           hole=0.5, marker_colors=colors))
    fig.update_layout(title=title, height=220,
                      margin=dict(l=5,r=5,t=40,b=5),
                      paper_bgcolor="rgba(0,0,0,0)", showlegend=True)
    return fig


# ─── Render results ──────────────────────────
def render_results(result: Dict[str, Any]):
    novelty       = result.get("novelty", {})
    sim           = result.get("similarity", {})
    papers        = result.get("similar_papers", [])
    gaps          = result.get("gaps", {})
    titles        = result.get("title_suggestions", [])
    explanation   = result.get("explanation", "")
    hallucination = result.get("hallucination", {})
    plagiarism    = result.get("plagiarism", {})

    label   = novelty.get("label", "MEDIUM")
    pct     = novelty.get("percentage", 50)
    n_color = novelty.get("color", "#f39c12")
    h_score = hallucination.get("grounding_score", 100)
    h_color = hallucination.get("color", "#2ecc71")
    p_score = plagiarism.get("plagiarism_score", 0.0)
    p_color = plagiarism.get("color", "#2ecc71")
    o_score = plagiarism.get("originality_score", 100.0)

    # ── Top gauges ────────────────────────────
    st.markdown("---")
    st.subheader("📊 Analysis Dashboard")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.plotly_chart(gauge(pct, f"<b>Novelty</b><br>{label}", "%", n_color),
                        use_container_width=True, key="ng")
        st.markdown(f'<div style="text-align:center"><span class="novelty-badge-{label.lower()}">{label}</span></div>',
                    unsafe_allow_html=True)
    with c2:
        st.plotly_chart(plagiarism_gauge(p_score, p_color),
                        use_container_width=True, key="pg")
        risk = plagiarism.get("risk_level","SAFE")
        st.markdown(f'<div style="text-align:center;color:{p_color}"><b>{plagiarism.get("icon","")} {risk} RISK</b></div>',
                    unsafe_allow_html=True)
    with c3:
        st.plotly_chart(gauge(o_score, "<b>Originality</b>", "%", "#2ecc71"),
                        use_container_width=True, key="og")
    with c4:
        st.plotly_chart(gauge(h_score, "<b>AI Grounding</b>", "/100", h_color),
                        use_container_width=True, key="hg")

    stats = sim.get("stats", {})
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Max Similarity",    f"{stats.get('max_sim',0)*100:.1f}%")
    m2.metric("Mean Similarity",   f"{stats.get('mean_sim',0)*100:.1f}%")
    m3.metric("Papers Retrieved",  len(papers))
    m4.metric("Plagiarism Score",  f"{p_score:.1f}%")

    st.markdown(f'<div class="metric-card">{novelty.get("description","")}</div>',
                unsafe_allow_html=True)

    if sim.get("duplicate_risk"):
        st.error(f"🚨 **Duplicate Risk!** {len(sim.get('duplicates',[]))} paper(s) ≥90% similar.")

    # ── Plagiarism ────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Plagiarism Detection Report")
    css_map = {"SAFE":"plag-safe","LOW":"plag-low","MEDIUM":"plag-medium","HIGH":"plag-high"}
    css = css_map.get(risk,"plag-safe")
    st.markdown(
        f'<div class="{css}"><b>{plagiarism.get("icon","")} {risk} — '
        f'{p_score:.1f}% Plagiarism | {o_score:.1f}% Original</b><br>'
        f'{plagiarism.get("message","")}</div>',
        unsafe_allow_html=True
    )

    ps = plagiarism.get("stats",{})
    pc1,pc2,pc3,pc4 = st.columns(4)
    pc1.metric("Plagiarism",    f"{p_score:.1f}%")
    pc2.metric("Originality",   f"{o_score:.1f}%")
    pc3.metric("Sentences Checked", ps.get("total_sentences_checked",0))
    pc4.metric("Phrases Flagged",   ps.get("phrases_flagged",0))

    matched = plagiarism.get("matched_papers",[])
    if matched:
        with st.expander("📄 Matched Papers"):
            for mp in matched:
                rc = "#e74c3c" if mp["risk"]=="HIGH" else "#f39c12" if mp["risk"]=="MEDIUM" else "#3498db"
                st.markdown(
                    f'<div class="paper-card"><b style="color:{rc}">[{mp["risk"]}]</b> '
                    f'{mp["title"]}<br><small>Year: {mp["year"]} · Domain: {mp["domain"]} · '
                    f'Similarity: <b>{mp["similarity_pct"]:.1f}%</b></small></div>',
                    unsafe_allow_html=True
                )

    sent_matches = plagiarism.get("sentence_matches",[])
    if sent_matches:
        with st.expander(f"⚠️ Flagged Sentences ({len(sent_matches)})"):
            for sm in sent_matches:
                st.markdown(
                    f'<div class="flagged-sent"><b>Your text:</b> "{sm.get("user_sentence","")[:150]}"<br>'
                    f'<b>Matches:</b> "{sm.get("paper_sentence","")[:150]}"<br>'
                    f'<small>📄 {sm.get("paper_title","")[:60]} ({sm.get("paper_year","")}) — '
                    f'<b>{sm.get("similarity_pct",0):.1f}% similar</b></small></div>',
                    unsafe_allow_html=True
                )
    else:
        st.success("✅ No sentence-level matches found.")

    phrase_matches = plagiarism.get("phrase_matches",[])
    if phrase_matches:
        with st.expander(f"🔤 Exact Phrase Matches ({len(phrase_matches)})"):
            for pm in phrase_matches:
                st.markdown(
                    f'<div class="phrase-match">🔤 <b>"{pm.get("phrase","")}"</b> '
                    f'({pm.get("word_count",0)} words) — '
                    f'{pm.get("paper_title","")[:60)} ({pm.get("paper_year","")})</div>',
                    unsafe_allow_html=True
                )
    else:
        st.success("✅ No exact phrase matches found.")

    plag_recs = plagiarism.get("recommendations",[])
    if plag_recs:
        with st.expander("💡 How to Fix Plagiarism Issues"):
            for r in plag_recs: st.markdown(f"- {r}")

    # ── Similar papers ────────────────────────
    st.markdown("---")
    st.subheader("📚 Most Similar Research Papers")
    if papers:
        st.plotly_chart(sim_bar(papers), use_container_width=True, key="sb")
        for p in papers[:5]:
            with st.expander(f"📄 [{p.get('rank','?')}] {p.get('title','')} — {p.get('similarity_pct',0):.1f}%"):
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**Year:** {p.get('year','N/A')}")
                c2.markdown(f"**Domain:** {p.get('domain','N/A')}")
                c3.markdown(f"**Region:** {p.get('region','N/A')}")
                st.markdown(f"**Abstract:** {p.get('abstract','')[:400]}…")
    else:
        st.info("No similar papers found!")

    # ── Distribution charts ───────────────────
    if papers:
        st.markdown("---")
        cd,cy = st.columns(2)
        with cd: st.plotly_chart(domain_pie(sim.get("domain_dist",{})),
                                 use_container_width=True, key="dp")
        with cy: st.plotly_chart(year_bar(sim.get("year_dist",{})),
                                 use_container_width=True, key="yb")

    # ── Gap report ────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Research Gap Report")
    for stmt in gaps.get("gap_statements",[]):
        st.markdown(f'<div class="gap-card">{stmt}</div>', unsafe_allow_html=True)

    gap_dims = gaps.get("gap_dimensions",{})
    if gap_dims:
        with st.expander("📋 Detailed Gap Dimensions"):
            d1,d2 = st.columns(2)
            with d1:
                for key,lbl in [("regional_gaps","📍 Regions"),
                                 ("population_gaps","👥 Populations"),
                                 ("temporal_gaps","📅 Time Periods")]:
                    items = gap_dims.get(key,[])
                    if items:
                        st.markdown(f"**{lbl}:**")
                        for g in items[:5]: st.markdown(f"  - {g.title()}")
            with d2:
                for key,lbl in [("methodological_gaps","🔬 Methods"),
                                 ("thematic_gaps","💡 Themes"),
                                 ("theoretical_gaps","📚 Theories")]:
                    items = gap_dims.get(key,[])
                    if items:
                        st.markdown(f"**{lbl}:**")
                        for g in items[:5]: st.markdown(f"  - {g.title()}")

    # ── Title suggestions ─────────────────────
    st.markdown("---")
    st.subheader("✍️ Suggested Improved Research Titles")
    for i,t in enumerate(titles,1):
        st.markdown(f'<div class="title-suggestion">📝 <b>Option {i}:</b> {t}</div>',
                    unsafe_allow_html=True)

    # ── Recommendations ───────────────────────
    sugg = novelty.get("suggestions",[])
    if sugg:
        st.markdown("---")
        st.subheader("💡 Recommendations")
        for s in sugg: st.markdown(f"- {s}")

    # ── AI Report ────────────────────────────
    st.markdown("---")
    st.subheader("📋 AI-Generated Research Report")
    with st.expander("📖 View Full Report", expanded=True):
        st.markdown(explanation.replace("\n","  \n"))

    # ── Hallucination ─────────────────────────
    st.markdown("---")
    st.subheader("🛡️ AI Hallucination Check")
    h_count = hallucination.get("hallucination_count",0)
    h_total = hallucination.get("total_claims",0)
    hc1,hc2,hc3,hc4 = st.columns(4)
    hc1.metric("Grounding Score", f"{h_score}/100")
    hc2.metric("Total Claims",    h_total)
    hc3.metric("Grounded",        h_total - h_count)
    hc4.metric("Flagged",         h_count)

    h_css = ("hallucination-safe" if h_score>=80
             else "hallucination-warn" if h_score>=60
             else "hallucination-danger")
    st.markdown(
        f'<div class="{h_css}"><b>{hallucination.get("label","")}</b> — '
        f'Grounding Score: {h_score}/100</div>',
        unsafe_allow_html=True
    )

    flagged = hallucination.get("flagged_claims",[])
    if flagged:
        with st.expander(f"🔴 {len(flagged)} Flagged Claim(s)"):
            for fc in flagged:
                sev  = fc.get("severity","LOW")
                icon = "🔴" if sev=="HIGH" else "🟡" if sev=="MEDIUM" else "🟢"
                st.markdown(
                    f'<div class="flagged-claim">{icon} <b>[{sev}]</b> {fc.get("issue","")}<br>'
                    f'<small><i>"{fc.get("claim","")[:120]}…"</i></small></div>',
                    unsafe_allow_html=True
                )
    else:
        st.success("✅ No hallucinations detected.")

    # ── Download ──────────────────────────────
    st.markdown("---")
    st.download_button(
        label="📥 Download Full Analysis (JSON)",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="research_analysis.json",
        mime="application/json",
        use_container_width=True,
    )


# ─── Main ────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("## 🔬 Research Detector v3.0")
        st.markdown("**For PhD Scholars — India**")
        st.divider()
        st.markdown("### ⚙️ Settings")
        top_k      = st.slider("Papers to retrieve", 3, 15, 8)
        check_plag = st.toggle("Plagiarism Check", value=True)
        st.divider()
        st.markdown("### 📖 What This Tool Does")
        st.markdown("""
- 📊 **Novelty Score** — Is your topic original?
- 🔍 **Plagiarism Check** — Similarity to existing papers
- 🔬 **Gap Detection** — What's missing in literature?
- 🤖 **AI Report** — Gemini-powered explanation
- 🛡️ **Hallucination Check** — AI accuracy verification
        """)
        st.divider()
        st.markdown("### 🎓 Supported Departments")
        st.markdown("""
- Arts & Science
- English Literature & ELT
- Engineering (CSE, ECE, EEE, Mech, Civil…)
        """)
        st.caption("© 2024 · AI Research Tools · India")

    st.markdown('<p class="main-header">🔬 AI Research Novelty & Gap Detector</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">For Arts, Science & Engineering PhD Scholars in India · '
        'Powered by AI + Plagiarism Detection + Hallucination Check</p>',
        unsafe_allow_html=True
    )

    # Load engines
    try:
        retriever = load_engines()
        st.sidebar.success(f"✅ {retriever.get_corpus_size()} papers loaded")
    except Exception as e:
        st.error(f"❌ Failed to load engines: {e}")
        return

    # Input form
    with st.form("form"):
        st.subheader("📝 Enter Your Research Details")
        col1,col2 = st.columns([3,1])
        with col1:
            title = st.text_input("Research Title *",
                placeholder="e.g., Deep Learning for Medical Image Segmentation in Indian Hospitals")
        with col2:
            domain = st.selectbox("Domain", [
                "",
                "Sociology","History","Political Science","Economics",
                "Psychology","Environmental Science","Literature","Philosophy",
                "Anthropology","Education","Geography","Women Studies",
                "Commerce","Linguistics","Botany","Zoology",
                "Chemistry","Physics","Mathematics","Statistics",
                "English Literature","English Language Teaching",
                "Postcolonial Studies","Comparative Literature","Translation Studies",
                "Computer Science and Engineering",
                "Electronics and Communication Engineering",
                "Electrical Engineering","Mechanical Engineering",
                "Civil Engineering","Chemical Engineering",
                "Artificial Intelligence","Machine Learning","Deep Learning",
                "Internet of Things","Cybersecurity","Data Science",
                "Robotics and Automation","VLSI Design",
                "Power Systems Engineering","Renewable Energy Engineering",
                "Structural Engineering","Environmental Engineering",
                "Nanotechnology","Materials Science","Biomedical Engineering",
            ])

        abstract = st.text_area("Abstract (recommended)",
            placeholder="Describe your research objectives and methodology…",
            height=120)
        keywords = st.text_input("Keywords",
            placeholder="e.g., deep learning, medical imaging, CNN, India")

        submitted = st.form_submit_button("🚀 Analyse Research", use_container_width=True)

    if submitted:
        if not title.strip():
            st.warning("⚠️ Please enter a research title.")
            return

        with st.spinner("🔍 Running full analysis…"):
            t0     = time.time()
            result = run_analysis(
                retriever     = retriever,
                title         = title.strip(),
                abstract      = abstract.strip(),
                keywords      = keywords.strip(),
                domain        = domain or "",
                top_k         = top_k,
                check_plagiarism = check_plag,
            )
        elapsed = time.time() - t0
        st.success(f"✅ Analysis complete in {elapsed:.1f}s")
        render_results(result)


if __name__ == "__main__":
    main()