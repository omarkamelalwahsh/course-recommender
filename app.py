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

# Custom CSS for premium look
st.markdown("""
<style>
    .course-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #dee2e6;
        transition: transform 0.2s;
    }
    .course-card:hover {
        transform: scale(1.0);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .rank-badge {
        background-color: #007bff;
        color: white;
        padding: 2px 8px;
        border-radius: 5px;
        font-weight: bold;
    }
    .level-badge {
        background-color: #28a745;
        color: white;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.9em;
        margin-top: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Global Singleton Recommender (Cached) ---
@st.cache_resource(show_spinner="Initializing AI Engine...")
def get_recommender():
    rec = CourseRecommender()
    target_path = "data/courses.csv"
    if not os.path.exists(target_path):
        st.error(f"Critical Error: {target_path} not found. Please ensure the dataset exists.")
        st.stop()
    rec.load_courses(target_path)
    return rec

try:
    rec = get_recommender()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- 2. Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    st.subheader("🎯 Pre-Search Filters")
    df_ref = rec.courses_df
    
    pre_levels = ["Any", "White", "Beginner", "Intermediate", "Advanced"]
    pre_categories = ["Any"] + sorted(list(df_ref['category'].unique()))
    
    sel_level = st.selectbox("Pre-Level (Manual Override)", pre_levels)
    sel_category = st.selectbox("Pre-Category", pre_categories)
    
    max_dur = int(df_ref['duration_hours'].max()) + 1
    sel_duration = st.slider("Pre-Max Duration (Hours)", 0, max_dur, max_dur)
    
    st.markdown("---")
    st.subheader("🛠️ Advanced Settings")
    top_k = st.number_input("Candidates", 5, 100, 30)
    show_debug = st.checkbox("Show Debug Logs")

# --- 3. Main Content ---
st.title("🎓 Smart Course Recommender")
st.markdown("Find the perfect course for your professional growth powered by semantic search.")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("What would you like to learn today?", placeholder="e.g., Python for Machine Learning")
with col2:
    st.write("") # Spacer
    st.write("") # Spacer
    search_btn = st.button("Recommend", type="primary", use_container_width=True)

# --- 4. Search Logic ---
if search_btn or (query and "query_done" not in st.session_state):
    if not query.strip():
        st.warning("Please enter a query to get recommendations.")
    else:
        with st.spinner("AI is analyzing courses..."):
            pre_filters = {
                "level": sel_level,
                "category": sel_category,
                "max_duration": sel_duration
            }
            results_pkg = rec.recommend(query, top_k=top_k, pre_filters=pre_filters)
            st.session_state["results"] = results_pkg
            st.session_state["query_done"] = query

# --- 5. Results Display ---
if "results" in st.session_state:
    res_pkg = st.session_state["results"]
    results = res_pkg.get("results", [])
    debug = res_pkg.get("debug_info", {})
    
    # Show Inferred Level Badge
    st.markdown(f'<div class="level-badge">Inferred Level: {debug.get("inferred_level", "Unknown")} (based on your query)</div>', unsafe_allow_html=True)
    
    if debug.get("keyword_warning"):
        st.error(f"⚠️ {debug['keyword_warning']}")
    
    if show_debug:
        with st.expander("🛠️ Debug Information"):
            st.json(debug)

    if not results:
        if not debug.get("keyword_warning"):
            st.info("No courses match those filters. Try broadening your criteria.")
    else:
        # --- 6. Post-Filters ---
        st.markdown("---")
        st.subheader("🔍 Refine Results (Post-run)")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            p_cat = st.multiselect("Filter by Category", sorted(list(set(r['category'] for r in results))))
        with c2:
            all_lvls = sorted(list(set(r['level'] for r in results)))
            p_lvl = st.multiselect("Filter by Level", all_lvls)
        with c3:
            has_img = st.checkbox("Only with Preview Image")
            has_inst = st.checkbox("Only with Instructor Name")
            
        # Apply Post-Filters
        filtered = results
        if p_cat:
            filtered = [r for r in filtered if r['category'] in p_cat]
        if p_lvl:
            filtered = [r for r in filtered if r['level'] in p_lvl]
        if has_img:
            filtered = [r for r in filtered if r.get('cover') and str(r['cover']) != 'nan' and r['cover'] != '']
        if has_inst:
            filtered = [r for r in filtered if r.get('instructor') and r['instructor'] != 'Unknown']
            
        st.write(f"Showing {len(filtered)} results")
        
        # Download JSON
        json_str = json.dumps(filtered, indent=2)
        st.download_button(
            label="📥 Download JSON Results",
            data=json_str,
            file_name="recommendations.json",
            mime="application/json"
        )
        
        # Display Cards
        for course in filtered:
            with st.container():
                # Hyperlink if link exists
                title_html = course['title']
                if course.get('course_link'):
                    title_html = f'<a href="{course["course_link"]}" target="_blank" style="text-decoration: none; color: inherit;">{course["title"]}</a>'
                
                st.markdown(f"""
                <div class="course-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h3>{title_html}</h3>
                            <p><strong>Instructor:</strong> {course['instructor']} | <strong>Category:</strong> {course['category']} | <strong>Level:</strong> {course['level']}</p>
                            <p style="color: #666;">{course['description'][:250]}...</p>
                        </div>
                        <div style="text-align: right;">
                            <span class="rank-badge">Rank: {course['rank']}/10</span><br>
                            <small>Score: {course['similarity_score']:.2f}</small>
                        </div>
                    </div>
                    <div style="margin-top: 10px;">
                        <strong>Skills:</strong> {' '.join([f'`{s}`' for s in str(course['skills']).split('|')])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if course.get('cover') and str(course['cover']) != 'nan' and course['cover'] != '':
                    try:
                        st.image(course['cover'], width=240)
                    except:
                        pass
                st.markdown("---")
else:
    st.info("👈 Enter a topic in the search bar and click 'Recommend' or press Enter.")
