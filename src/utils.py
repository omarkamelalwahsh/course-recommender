import re

# Stopwords List (English + Arabic)
STOPWORDS = {
    'the', 'is', 'in', 'for', 'where', 'when', 'to', 'at', 'be', 'this', 'that',
    'how', 'what', 'a', 'an', 'of', 'and', 'or', 'with', 'by', 'from',
    'need', 'looking', 'for', 'in', 'on', 'of', 'and', 'with', 'please', 'pls',
    'programming', 'language', 'basics', 'course', 'tutorial',
    'انا', 'عاوز', 'عايز', 'محتاج', 'اتعلم', 'كورس', 'دورة', 'دروس',
    'شرح', 'من', 'فضلك', 'في', 'على', 'عن', 'كيف', 'ما', 'هو', 'ماهو', 'تعليم',
    'اساسيات', 'مقدمة', 'احتراف', 'كامل', 'شامل', 'عربي', 'بالعربي'
}

def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))

def normalize_query(query: str) -> str:
    """
    Normalize query: lowercase, remove punctuation, remove extra spaces.
    Expand common abbreviations: ML, NLP, AWS, BI, CV.
    Translate Arabic Tech Terms -> English (Strict Mapping).
    Remove stopwords.
    """
    # 1. Lowercase
    q = query.lower()
    
    # 2. Comprehensive Arabic -> English Mapping
    # This ensures "جافا" becomes "java" and enforces strict matching later.
    ar_to_en = {
        # Languages
        'بايثون': 'python', 'جافا': 'java', 'سي شارب': 'c#', 'سي بلس بلس': 'c++', 
        'جافاسكريبت': 'javascript', 'جافا سكربت': 'javascript', 'بي اتش بي': 'php',
        'روبي': 'ruby', 'سويفت': 'swift', 'كوتلن': 'kotlin', 'دارت': 'dart',
        'راست': 'rust', 'جو': 'golang', 'جولانج': 'golang', 'اس كيو ال': 'sql',
        
        # Frameworks & Libs
        'فلاتر': 'flutter', 'رياكت': 'react', 'انجلر': 'angular', 'فيو': 'vue',
        'جانجو': 'django', 'فلاكس': 'flask', 'سبرينج': 'spring', 'لارفيل': 'laravel',
        'دوت نت': '.net', 'نود': 'node', 'اكسبريس': 'express', 'باندز': 'pandas',
        'تنسرفلو': 'tensorflow', 'بايتورش': 'pytorch', 'كيراس': 'keras',
        
        # Domains
        'ذكاء': 'intelligence', 'اصطناعي': 'artificial', 'تعلم': 'learning', 'الالة': 'machine',
        'بيانات': 'data', 'تحليل': 'analysis', 'علم': 'science', 'رؤية': 'vision',
        'حاسوب': 'computer', 'شبكات': 'networks', 'امن': 'security', 'سبراني': 'cyber',
        'سحابة': 'cloud', 'ويب': 'web', 'مواقع': 'web', 'تطبيقات': 'apps', 'موبائل': 'mobile',
        'اندرويد': 'android', 'ايفون': 'ios', 'العاب': 'games', 'يونيقي': 'unity',
        
        # Soft Skills / Business
        'تسويق': 'marketing', 'ادارة': 'management', 'مشاريع': 'project', 'اعمال': 'business',
        'قيادة': 'leadership', 'تواصل': 'communication', 'مبيعات': 'sales', 
        'محاسبة': 'accounting', 'مالية': 'finance', 'موارد': 'hr', 'بشرية': 'human',
        
        # Tools
        'اكسل': 'excel', 'وورد': 'word', 'بوربوينت': 'powerpoint', 'فوتوشوب': 'photoshop',
        'اليستريتور': 'illustrator', 'فيجما': 'figma', 'لينكس': 'linux', 'ديكر': 'docker',
        'كوبيرنيتس': 'kubernetes', 'جيت': 'git', 'جيت هب': 'github'
    }
    
    # Replace sorted by length to avoid partial replacements (e.g. 'جافاسكريبت' before 'جافا')
    sorted_dict = dict(sorted(ar_to_en.items(), key=lambda item: len(item[0]), reverse=True))
    
    for ar, en in sorted_dict.items():
        if ar in q:
            q = q.replace(ar, en)

    # 3. Expand common abbreviations & Fix Common Typos
    abbr_map = {
        'ml': 'machine learning',
        'nlp': 'natural language processing',
        'aws': 'amazon web services',
        'bi': 'business intelligence',
        'cv': 'computer vision',
        'ai': 'artificial intelligence',
        'ds': 'data science',
        'java script': 'javascript'  # Fix common typo
    }
    
    # Use regex for word boundary replacement for abbreviations
    for abbr, full in abbr_map.items():
        pattern = re.compile(r'\b' + re.escape(abbr) + r'\b')
        q = pattern.sub(full, q)
        
    # 4. Remove Punctuation (Keep # + .) for C#, .NET, C++, Node.js
    q = re.sub(r'[^\w\s\u0600-\u06FF\+\#\.]', ' ', q)
    
    # 5. Remove Stopwords & Clean
    tokens = q.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    return " ".join(tokens).strip()
