from src.recommender import CourseRecommender
import json

rec = CourseRecommender()
rec.load_courses("data/courses.csv")

query = "Advanced JS"
res = rec.recommend(query)

print(f"Query: {query}")
print(f"Debug Info: {json.dumps(res['debug_info'], indent=2)}")
print(f"Results Count: {len(res['results'])}")
if len(res['results']) == 0:
    # Check if filters are blocking
    all_text = " ".join(rec.courses_df['combined_text'].tolist()).lower()
    for kw in res['debug_info']['strong_keywords']:
        print(f"Keyword '{kw}' in all_text: {kw in all_text}")
    
    # Check matching with level
    lvl = res['debug_info']['inferred_level']
    filtered = rec.courses_df[rec.courses_df['level'] == lvl]
    print(f"Courses at level '{lvl}': {len(filtered)}")
    if len(filtered) > 0:
        filt_text = " ".join(filtered['combined_text'].tolist()).lower()
        for kw in res['debug_info']['strong_keywords']:
            print(f"Keyword '{kw}' in filtered level text: {kw in filt_text}")
