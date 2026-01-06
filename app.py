
import streamlit as st
import pandas as pd
import time
from src.pipeline import CourseRecommenderPipeline
from src.schemas import RecommendRequest
from src.config import TOP_K_DEFAULT

# --- Page Config ---
st.set_page_config(
    page_title="Zedny Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Validations ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

@st.cache_resource
def get_pipeline():
    return CourseRecommenderPipeline()

# --- Custom CSS for Cards ---
st.markdown("""
<style>
    .course-card {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .course-card:hover {
        transform: scale(1.02);
        border-color: #4CAF50;
    }
    .course-title {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50 !important;
        text-decoration: none;
        margin-bottom: 5px;
        display: block;
    }
    .course-meta {
        color: #888;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .course-desc {
        color: #ddd;
        font-size: 15px;
        line-height: 1.5;
    }
    .score-badge {
        background-color: #2c2c2c;
        color: #4CAF50;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 12px;
        margin-left: 10px;
        font-weight: bold;
    }
    .why-section {
        margin-top: 10px;
        padding: 10px;
        background-color: #262626;
        border-radius: 5px;
        font-size: 13px;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.title("🎓 Zedny Smart Course Recommender")
    st.caption("AI-Powered Semantic Search | v2.1 - Strict & Stable")

    # Init Pipeline
    if st.session_state.pipeline is None:
        with st.spinner("Initializing AI Engine..."):
            try:
                st.session_state.pipeline = get_pipeline()
            except Exception as e:
                st.error(f"Failed to initialize system: {e}")
                st.stop()

    pipeline = st.session_state.pipeline

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Search Filters")
    
    # Extract categories
    categories = ["Any"]
    levels = ["Any"]
    
    if pipeline.courses_df is not None:
        cats = sorted(pipeline.courses_df['category'].dropna().unique().tolist())
        levs = sorted(pipeline.courses_df['level'].dropna().unique().tolist())
        categories += cats
        levels += levs

    sel_category = st.sidebar.selectbox("Category", categories)
    sel_level = st.sidebar.selectbox("Level", levels)
    top_k = st.sidebar.slider("Number of Results", 5, 50, TOP_K_DEFAULT)
    
    enable_rerank = st.sidebar.checkbox("Enable Deep Re-ranking (Slower)", value=False)
    show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption("v2.1 - Production")

    # --- Search Input ---
    query = st.text_input("What do you want to learn today?", placeholder="e.g. Python, Marketing, Leadership...")

    # --- Logic ---
    if query:
        if len(query.strip()) < 2:
            st.warning("Please enter a valid search query (at least 2 chars).")
            return

        with st.spinner("Analyzing and searching..."):
            try:
                # Prepare Filter
                filters = {}
                if sel_category != "Any": filters['category'] = sel_category
                if sel_level != "Any": filters['level'] = sel_level

                request = RecommendRequest(
                    query=query,
                    top_k=top_k,
                    filters=filters,
                    rerank=enable_rerank
                )
                
                # Run Pipeline
                response = pipeline.recommend(request)

                # --- Display Results ---
                if response.total_found == 0:
                    st.warning("⚠️ No relevant courses found.")
                    st.info("Tip: Try more general keywords or check your spelling. Our system is strict to ensure relevance.")
                else:
                    st.success(f"Found {response.total_found} relevant courses!")
                    
                    for res in response.results:
                        why_html = ""
                        if show_debug and res.match_reasons:
                            reasons = " • ".join(res.match_reasons)
                            kws = ", ".join(res.matched_keywords)
                            why_html = f"""
                            <div class='why-section'>
                                <strong>💡 Why this course?</strong><br>
                                Reasons: {reasons}<br>
                                Matched Keywords: {kws}
                            </div>
                            """
                        
                        card_html = f"""
                        <div class="course-card">
                            <a href="{res.url}" target="_blank" class="course-title">
                                #{res.rank} {res.title}
                            </a>
                            <div class="course-meta">
                                <span class="score-badge">Relevance: {int(res.score * 100)}%</span>
                                | Category: {res.category} | Level: {res.level}
                            </div>
                            <div class="course-desc">
                                {res.debug_info.get('desc_snippet', '')}...
                            </div>
                            {why_html}
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        
                    if show_debug:
                        with st.expander("🛠️ Full Technical Data (JSON)"):
                            st.json(response.dict())

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                
if __name__ == "__main__":
    main()
