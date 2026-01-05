import streamlit as st
import pandas as pd
import json
import os
from src.recommender import CourseRecommender

# Page Configuration
st.set_page_config(
    page_title="Smart Course Recommender",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for premium look but WITHOUT images
st.markdown("""
<style>
    .course-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .meta-info {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 12px;
    }
    .rank-badge {
        background-color: #1a73e8;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: 700;
        float: right;
    }
    .inferred-badge {
        background-color: #e8f0fe;
        color: #1967d2;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 500;
        margin-bottom: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Global Singleton Recommender (Auto-load only) ---
@st.cache_resource(show_spinner="Initializing AI Engine...")
def get_recommender():
    rec = CourseRecommender()
    path = "data/courses.csv"
    if not os.path.exists(path):
        st.error(f"Critical Error: {path} not found.")
        st.stop()
    rec.load_courses(path)
    return rec

try:
    rec = get_recommender()
except Exception as e:
    st.error(f"System Load Error: {e}")
    st.stop()

# --- 2. Sidebar Configuration (PRE-RUN FILTERS) ---
with st.sidebar:
    st.title("⚙️ Filter Settings")
    st.markdown("---")
    st.subheader("🎯 Pre-Search (Hard Filters)")
    
    pre_lvls = ["Any", "White", "Beginner", "Intermediate", "Advanced"]
    cats = ["Any"] + sorted(list(rec.courses_df['category'].unique()))
    
    sel_level = st.selectbox("Pre-Level", pre_lvls)
    sel_cat = st.selectbox("Pre-Category", cats)
    
    max_h = int(rec.courses_df['duration_hours'].max()) + 1
    sel_dur = st.slider("Pre-Max Duration (Hours)", 0, max_h, max_h)
    
    st.markdown("---")
    top_k = st.number_input("Candidates", 5, 100, 30)
    debug_mode = st.checkbox("Show Development Logs")

# --- 3. Main Content ---
st.title("🎓 Smart Course Recommender")
st.markdown("Discover courses tailored to your expertise using semantic AI.")

query_col, btn_col = st.columns([4, 1])
with query_col:
    user_query = st.text_input("Search for topics...", placeholder="e.g., Python for ML")
with btn_col:
    st.write("") # padding
    st.write("") # padding
    go_btn = st.button("Get Recommendations", type="primary", use_container_width=True)

# --- 4. Logic ---
if go_btn or (user_query and "res" not in st.session_state):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("AI analyzing dataset..."):
            pre_f = {"level": sel_level, "category": sel_cat, "max_duration": sel_dur}
            pkg = rec.recommend(user_query, top_k=top_k, pre_filters=pre_f)
            st.session_state["res"] = pkg

# --- 5. Display Results ---
if "res" in st.session_state:
    res_pkg = st.session_state["res"]
    raw_results = res_pkg.get("results", [])
    debug = res_pkg.get("debug_info", {})
    
    # LEVEL INFERENCE BADGE
    st.markdown(f'<div class="inferred-badge">Inferred Level: {debug.get("inferred_level", "Unknown")}</div>', unsafe_allow_html=True)
    
    if debug.get("keyword_warning"):
        st.error(f"⚠️ {debug['keyword_warning']}")
        
    if debug_mode:
        with st.expander("🛠️ Debug Logs"):
            st.json(debug)

    if not raw_results:
        if not debug.get("keyword_warning"):
            st.info("No courses match your pre-filters. Try broadening the level or category.")
    else:
        # --- 6. POST-RUN FILTERS ---
        st.markdown("---")
        st.subheader("🔍 Refine these results (Post-Filters)")
        post_col1, post_col2, post_col3 = st.columns(3)
        
        with post_col1:
            post_cats = st.multiselect("Filter by Category", sorted(list(set(r['category'] for r in raw_results))))
        with post_col2:
            post_lvls = st.multiselect("Filter by Level", sorted(list(set(r['level'] for r in raw_results))))
        with post_col3:
            st.write("") # align
            filt_inst = st.checkbox("Only with Instructor Name")

        # Application
        filtered = raw_results
        if post_cats:
            filtered = [r for r in filtered if r['category'] in post_cats]
        if post_lvls:
            filtered = [r for r in filtered if r['level'] in post_lvls]
        if filt_inst:
            filtered = [r for r in filtered if r.get('instructor') and r['instructor'] != 'Unknown']

        st.write(f"Showing {len(filtered)} results")
        
        # Download
        st.download_button("📥 Download Results (JSON)", json.dumps(filtered, indent=2), "recommendations.json", "application/json")
        st.markdown("---")

        # Cards
        for c in filtered:
            with st.container():
                # RANK BADGE
                st.markdown(f'<span class="rank-badge">Rank: {c["rank"]}/10</span>', unsafe_allow_html=True)
                
                # FIXED HYPERLINK TITLE
                title = c['title']
                url = c.get('course_link', '#')
                if url and url != '#':
                    st.markdown(
                        f'<a href="{url}" target="_blank" style="text-decoration:none; font-size:24px; font-weight:700; color:#1a73e8;">{title}</a>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f'<h3 style="margin:0;">{title}</h3>', unsafe_allow_html=True)

                # META & DESCRIPTION
                st.markdown(f"""
                <div class="meta-info">
                    {c['category']} | {c['level']} | {c['instructor']}
                </div>
                <div style="margin-bottom: 20px;">
                    {c['description'][:300]}...
                </div>
                """, unsafe_allow_html=True)
                
                # SKILLS
                sk = str(c['skills']).split('|')
                st.markdown(" ".join([f"`{s}`" for s in sk if s]))
                st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
                st.markdown("---")
else:
    st.info("👈 Use the sidebar for hard filters and enter a topic to start.")
