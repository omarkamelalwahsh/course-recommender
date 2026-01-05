import streamlit as st
import pandas as pd
import os
from src.recommender import CourseRecommender

# Page Configuration
st.set_page_config(
    page_title="Zedny Smart Course Recommender",
    page_icon="🎓",
    layout="wide"
)

# --- 1. Global Singleton Recommender (Cached) ---
@st.cache_resource(show_spinner="Initializing Course Recommender System...")
def get_recommender():
    """
    Initialize Recommender and Load Data ONCE per app lifecycle.
    """
    rec = CourseRecommender()
    
    # Automatic Data Loading Strategy
    target_path = "data/courses.csv"
    
    # We pass the path; rec.load_courses handles fallback if missing/invalid
    rec.load_courses(target_path)
    
    return rec

# Initialize system
try:
    rec = get_recommender()
except Exception as e:
    st.error(f"Critical System Error: {e}")
    st.stop()

# Determine Status Message
# We check if the loaded data matches fallback data logic or file existence
data_path = "data/courses.csv"
if os.path.exists(data_path) and hasattr(rec, 'courses_df') and not rec.courses_df.empty:
    count = len(rec.courses_df)
    status_msg = f"✅ Loaded {count} courses from data/courses.csv"
    status_type = "success"
else:
    # If using fallback or file missing
    status_msg = "⚠️ Using fallback dataset (data/courses.csv not found or invalid)"
    status_type = "warning"


# --- 2. Sidebar Configuration (Clean - No Upload) ---
with st.sidebar:
    st.header("Search Configuration")
    
    # Status Indicator
    if status_type == "success":
        st.caption(status_msg)
    else:
        st.warning(status_msg)
    
    st.divider()
    
    # Pre-Run Filters using loaded data
    df_ref = rec.courses_df
    
    if df_ref is not None and not df_ref.empty:
        pre_levels = ["Any"] + sorted(list(df_ref['level'].unique())) if 'level' in df_ref.columns else ["Any"]
        pre_categories = ["Any"] + sorted(list(df_ref['category'].unique())) if 'category' in df_ref.columns else ["Any"]
        max_dur_ref = int(df_ref['duration_hours'].max()) + 1 if 'duration_hours' in df_ref.columns else 100
    else:
        pre_levels = ["Any"]
        pre_categories = ["Any"]
        max_dur_ref = 100

    pre_level = st.selectbox("Pre-Level", pre_levels, index=0)
    pre_category = st.selectbox("Pre-Category", pre_categories, index=0)
    
    valid_max_dur = max(1, max_dur_ref)
    pre_max_duration = st.slider("Pre-Max Duration", 0, valid_max_dur, valid_max_dur)
    
    top_k_raw = st.number_input("Top K Candidates", min_value=5, max_value=100, value=30, step=5)
    
    st.divider()
    show_debug = st.checkbox("Show Debug Info", value=False)
    
    
# --- 3. Main Content ---
st.title("🎓 Zedny Smart Course Recommender")
st.markdown("### 🔍 Find a Course")

# Example Queries
cols = st.columns([1, 1, 1, 1, 1, 4])
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

def set_q(txt): st.session_state["query_input"] = txt

if cols[0].button("ML"): set_q("ML")
if cols[1].button("NLP"): set_q("NLP")
if cols[2].button("AWS"): set_q("AWS")
if cols[3].button("Flutter"): set_q("Flutter")
if cols[4].button("BI"): set_q("BI")

query = st.text_input("What do you want to learn?", value=st.session_state["query_input"], placeholder="e.g. Python for Data Science")
search_clicked = st.button("Get Recommendations", type="primary")

# --- 4. Search Logic ---
if search_clicked:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Auto-detect advanced
                final_pre_level = pre_level
                if pre_level == "Any":
                    strong_keywords = ["advanced", "expert", "senior", "deep", "master"]
                    if any(kw in query.lower() for kw in strong_keywords):
                        final_pre_level = "Advanced"
                        st.toast(f"Detected advanced topic '{query}'. Auto-setting Level to Advanced.", icon="🧠")
                
                pre_filters = {
                    "level": final_pre_level,
                    "category": pre_category,
                    "max_duration": pre_max_duration
                }
                
                response = rec.recommend(
                    query, 
                    top_k=top_k_raw,
                    pre_filters=pre_filters,
                    similarity_threshold=0.25
                )
                
                results = response.get("results", [])
                debug_info = response.get("debug_info", {})
                
                st.session_state["last_debug_info"] = debug_info
                
                if debug_info.get("keyword_warning"):
                     st.warning(debug_info["keyword_warning"])
                     st.session_state["raw_results"] = pd.DataFrame()
                elif results:
                    st.session_state["raw_results"] = pd.DataFrame(results)
                else:
                    st.session_state["raw_results"] = pd.DataFrame() 
                    st.warning("No strong matches found. Try changing your query or relax filters.")
                    
            except Exception as e:
                st.error(f"An error occurred during search: {e}")

# --- 5. Debug Panel ---
if show_debug and st.session_state.get("last_debug_info"):
    with st.expander("🛠️ Debug Information", expanded=True):
        d_info = st.session_state["last_debug_info"]
        st.write(f"**Original Query:** `{d_info.get('query')}`")
        st.write(f"**Expanded Query:** `{d_info.get('expanded_query')}`")
        st.write(f"**Courses after pre-filter:** `{d_info.get('pre_filter_count')}` / `{d_info.get('total_courses')}`")
        
        scores = d_info.get('top_raw_scores', [])
        if scores:
            st.write(f"**Top 5 Raw Similarity Scores:** `{[round(s, 4) for s in scores]}`")
        else:
            st.write("**Top 5 Raw Similarity Scores:** None")
            
        if d_info.get("keyword_warning"):
            st.error(f"**Guardrail Warning:** {d_info.get('keyword_warning')}")


# --- 6. Results & Post-Filters ---
if "raw_results" in st.session_state and st.session_state["raw_results"] is not None and not st.session_state["raw_results"].empty:
    st.divider()
    st.header("Refine Results")
    
    df = st.session_state["raw_results"].copy()
    
    # Post-Filters
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        post_levels = ["Any"] + sorted(list(df['level'].unique())) if 'level' in df.columns else ["Any"]
        post_level = st.selectbox("Post-Level", post_levels, key="post_lvl")
        
    with c2:
        post_categories = ["Any"] + sorted(list(df['category'].unique())) if 'category' in df.columns else ["Any"]
        post_category = st.selectbox("Post-Category", post_categories, key="post_cat")
        
    with c3:
        max_post_dur_val = int(df['duration_hours'].max()) + 1 if 'duration_hours' in df.columns else 100
        post_duration_cap = st.slider("Max Duration", 0, max_post_dur_val, max_post_dur_val, key="post_dur")
        
    with c4:
        res_count = len(df)
        post_top_n = st.slider("Show Results", 1, res_count, min(5, res_count), key="post_topn")
    
    # Apply Post Filters
    filtered_df = df.copy()
    if post_level != "Any":
        filtered_df = filtered_df[filtered_df['level'] == post_level]
    if post_category != "Any":
        filtered_df = filtered_df[filtered_df['category'] == post_category]
    if 'duration_hours' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['duration_hours'] <= post_duration_cap]
        
    display_df = filtered_df.head(post_top_n)
    
    st.subheader(f"Showing {len(display_df)} results")
    
    if display_df.empty:
        st.warning("No courses match your POST-filters.")
    else:
        for idx, row in display_df.iterrows():
            with st.container():
                rank_display = f"{row.get('rank', 0)}/10"
                
                col_main, col_meta = st.columns([3, 1])
                with col_main:
                    st.markdown(f"### {row['title']}")
                    st.markdown(f"**Description:** {row['description']}")
                    st.markdown(f"**Skills:** `{row['skills']}`")
                with col_meta:
                    st.metric(label="Relevance Fit", value=rank_display)
                    st.caption(f"**Category:** {row['category']}")
                    st.caption(f"**Level:** {row['level']}")
                    st.caption(f"**Duration:** {row['duration_hours']}h")
                
                st.divider()
elif "raw_results" not in st.session_state or st.session_state["raw_results"] is None:
    st.info("👈 Use the Search filters and click 'Get Recommendations' to start.")
