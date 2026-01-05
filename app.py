import streamlit as st
import pandas as pd
import json
import os
from src.recommender import CourseRecommender

# Page Configuration
st.set_page_config(
    page_title="Zedny Smart Recommender",
    page_icon="🎓",
    layout="wide"
)

# Minimalist Style
st.markdown("""
<style>
    .course-card {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #1a73e8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .rank-badge {
        background-color: #1a73e8;
        color: white;
        padding: 2px 10px;
        border-radius: 4px;
        font-weight: bold;
        float: right;
    }
    .meta-text {
        font-size: 0.85em;
        color: #5f6368;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Singleton Recommender
@st.cache_resource
def get_rec():
    r = CourseRecommender()
    path = "data/courses.csv"
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}")
        st.stop()
    r.load_courses(path)
    return r

rec = get_rec()

# Sidebar
with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    pre_lvls = ["Any", "White", "Beginner", "Intermediate", "Advanced"]
    pre_cats = ["Any"] + sorted(list(rec.courses_df['category'].unique()))
    
    sel_lvl = st.selectbox("Level (Pre-filter)", pre_lvls)
    sel_cat = st.selectbox("Category (Pre-filter)", pre_cats)
    max_h = int(rec.courses_df['duration_hours'].max()) + 1
    sel_h = st.slider("Max Duration (Hours)", 0, max_h, max_h)
    
    st.markdown("---")
    top_k = st.number_input("Max Results", 5, 100, 30)
    debug_on = st.checkbox("Show Debug Logs")

# Main UI
st.title("🎓 Zedny Smart Course Recommender")
st.markdown("Enter keywords to find relevant courses with 100% precision.")

q_col, b_col = st.columns([4, 1])
with q_col:
    query = st.text_input("What do you want to learn?", placeholder="e.g. Python, SQL, Marketing")
with b_col:
    st.write("")
    st.write("")
    btn = st.button("Search", type="primary", use_container_width=True)

if btn or (query and "res_strict" not in st.session_state):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Filtering dataset..."):
            pre_f = {"level": sel_lvl, "category": sel_cat, "max_duration": sel_h}
            res = rec.recommend(query, top_k=top_k, pre_filters=pre_f)
            st.session_state["res_strict"] = res

# Results
if "res_strict" in st.session_state:
    res = st.session_state["res_strict"]
    data = res.get("results", [])
    dbg = res.get("debug_info", {})
    
    # ERROR MESSAGE
    if dbg.get("error_message"):
        st.error(dbg["error_message"])
    
    if debug_on:
        with st.expander("Debug Details"):
            st.json(dbg)

    if data:
        st.markdown("---")
        # Post-filters
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            p_cat = st.multiselect("Refine Category", sorted(list(set(r['category'] for r in data))))
        with p_c2:
            p_lvl = st.multiselect("Refine Level", sorted(list(set(r['level'] for r in data))))
            
        filtered = data
        if p_cat: filtered = [r for r in filtered if r['category'] in p_cat]
        if p_lvl: filtered = [r for r in filtered if r['level'] in p_lvl]
        
        st.info(f"Showing {len(filtered)} relevant courses")
        
        for c in filtered:
            with st.container():
                st.markdown(f'<span class="rank-badge">Rank: {c["rank"]}/10</span>', unsafe_allow_html=True)
                
                # Title Link
                url = c.get('course_link', '')
                if url:
                    st.markdown(f'<a href="{url}" target="_blank" style="text-decoration:none; font-size:1.4em; font-weight:bold; color:#1a73e8;">{c["title"]}</a>', unsafe_allow_html=True)
                else:
                    st.subheader(c['title'])
                
                st.markdown(f'<div class="meta-text">{c["category"]} | {c["level"]} | {c["instructor"]}</div>', unsafe_allow_html=True)
                st.write(f"{c['description'][:300]}...")
                
                # Skills
                sk = str(c['skills']).split('|')
                st.markdown(" ".join([f"`{s}`" for s in sk if s.strip()]))
                st.markdown("---")
else:
    st.info("👈 Enter a topic and click Search to begin.")
