"""
Cross-Lingual Information Retrieval (CLIR) Query Processor
Complete pipeline: Language Detection → Normalization → Translation → Expansion → Entity Mapping → Search
"""

import re
import json
import math
from pathlib import Path
from collections import Counter
import warnings
from typing import Dict, List, Tuple, Optional, Iterable

warnings.filterwarnings('ignore')

try:
    from langdetect import detect, LangDetectException
except ImportError:
    print("langdetect not installed. Install with: pip install langdetect")

try:
    from googletrans import Translator
except ImportError:
    print("googletrans not installed. Install with: pip install googletrans==4.0.2")

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    print("nltk not installed. Install with: pip install nltk")

try:
    import spacy
    nlp = None
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("spacy model not loaded (optional for NER)")
except ImportError:
    print("spacy not installed (optional)")


# ==================== 1. LANGUAGE DETECTION ====================

def detect_language(query: str) -> str:
    """Detect whether the query is in Bangla or English."""
    if not query or not query.strip():
        return 'unknown'
    
    try:
        lang_code = detect(query)
        if lang_code in ['bn', 'hi']:
            return 'bn'
        elif lang_code == 'en':
            return 'en'
        else:
            return _character_based_detection(query)
    except:
        return _character_based_detection(query)


def _character_based_detection(text: str) -> str:
    """Fallback: Bangla Unicode 0x0980-0x09FF vs ASCII"""
    bangla_count = sum(1 for char in text if 0x0980 <= ord(char) <= 0x09FF)
    english_count = sum(1 for char in text if 0x0000 <= ord(char) <= 0x007F)
    
    if bangla_count > english_count:
        return 'bn'
    elif english_count > bangla_count:
        return 'en'
    return 'unknown'


# ==================== 2. NORMALIZATION ====================

def normalize_query(query: str, language: Optional[str] = None, remove_stopwords: bool = False) -> str:
    """Normalize: lowercase, whitespace, optional stopwords."""
    if not query:
        return ""
    
    if language is None:
        language = detect_language(query)
    
    normalized = query.lower()
    
    if language == 'bn':
        normalized = _normalize_bangla(normalized)
    else:
        normalized = _normalize_english(normalized)
    
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    if remove_stopwords:
        normalized = _remove_stopwords(normalized, language)
    
    return normalized


def _normalize_english(text: str) -> str:
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_bangla(text: str) -> str:
    # Remove diacritics
    for diacritic in ['\u0981', '\u0982', '\u0983', '\u09BC', '\u09CD']:
        text = text.replace(diacritic, '')
    text = re.sub(r'[^\u0980-\u09FF\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _remove_stopwords(text: str, language: str) -> str:
    try:
        if language == 'en':
            stop_words = set(stopwords.words('english'))
        elif language == 'bn':
            stop_words = _get_bangla_stopwords()
        else:
            return text
        
        words = text.split()
        return ' '.join([w for w in words if w.lower() not in stop_words])
    except:
        return text


def _get_bangla_stopwords() -> set:
    return {
        'এর', 'এবং', 'একটি', 'যা', 'সে', 'তার', 'তাদের', 'তারা',
        'হয়েছে', 'হয়', 'থাকে', 'থাকা', 'আছে', 'ছিল', 'করা', 'করে',
        'করেছে', 'করব', 'করুন', 'করতে', 'দেওয়া', 'দিয়েছে', 'দিন',
        'দিয়ে', 'বলা', 'বলে', 'বলেছে', 'দ্বারা', 'থেকে', 'যদি', 'তখন',
        'এখন', 'কী', 'কে', 'কোন', 'যে', 'এই', 'এটি', 'ওই', 'না', 'নয়'
    }


# ==================== 3. QUERY TRANSLATION ====================

def translate_query(query: str, target_language: str) -> Optional[str]:
    """Translate query using Google Translate (free)."""
    if not query or not query.strip():
        return None
    
    try:
        translator = Translator()
        source_lang = detect_language(query)
        
        if source_lang == target_language:
            return query
        
        lang_map = {'en': 'en', 'bn': 'bn'}
        if source_lang not in lang_map or target_language not in lang_map:
            return None
        
        translation = translator.translate(query, src=lang_map[source_lang], dest=lang_map[target_language])
        
        # Handle both sync and async responses
        if hasattr(translation, '__await__'):
            # If it's a coroutine, we need to handle it differently
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            translation = loop.run_until_complete(translation)
        
        # Extract text from translation object
        if hasattr(translation, 'text'):
            return translation.text
        elif isinstance(translation, str):
            return translation
        else:
            return str(translation)
    except Exception as e:
        print(f"Translation error: {e}")
        return None


# ==================== 4. QUERY EXPANSION ====================

def expand_query(query: str, language: Optional[str] = None) -> List[str]:
    """Expand with synonyms/morphological variants."""
    if not query:
        return []
    
    if language is None:
        language = detect_language(query)
    
    expanded = [query]
    
    if language == 'en':
        expanded.extend(_expand_english_query(query))
    elif language == 'bn':
        expanded.extend(_expand_bangla_query(query))
    
    return list(dict.fromkeys(expanded))  # Remove duplicates, preserve order


def _expand_english_query(query: str) -> List[str]:
    try:
        from nltk.corpus import wordnet
        expansions = []
        words = query.split()
        
        for word in words:
            synsets = wordnet.synsets(word)
            for synset in synsets[:2]:
                for lemma in synset.lemmas()[:2]:
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word:
                        expansions.append(synonym)
        
        # Morphological variants
        if query.endswith('ing'):
            expansions.append(query[:-3])
        if query.endswith('ed'):
            expansions.append(query[:-2])
        if query.endswith('s'):
            expansions.append(query[:-1])
        
        return list(set(expansions))
    except:
        return []


def _expand_bangla_query(query: str) -> List[str]:
    expansions = []
    bangla_variants = {
        'করা': ['করে', 'করেছে', 'করবে', 'করুন', 'করতে'],
        'দেওয়া': ['দিয়েছে', 'দিন', 'দিয়ে', 'দিতে'],
        'বলা': ['বলে', 'বলেছে', 'বলুন', 'বলতে'],
        'যাওয়া': ['যায়', 'গেছে', 'যাচ্ছে', 'যাবে'],
        'আসা': ['আসে', 'এসেছে', 'আসছে', 'আসবে'],
        'হওয়া': ['হয়', 'হয়েছে', 'হচ্ছে', 'হবে'],
    }
    
    for root, variants in bangla_variants.items():
        if root in query:
            for variant in variants:
                expansions.append(query.replace(root, variant))
    
    return list(set(expansions))


# ==================== 5. NAMED ENTITY MAPPING ====================

def map_named_entities(query: str, source_lang: Optional[str] = None, 
                       target_lang: Optional[str] = None) -> Dict:
    """Extract NE and map across languages."""
    if not query:
        return {'entities': [], 'source_lang': 'unknown', 'target_lang': 'unknown', 'mappings': {}}
    
    if source_lang is None:
        source_lang = detect_language(query)
    if target_lang is None:
        target_lang = 'bn' if source_lang == 'en' else 'en'
    
    entities = _extract_named_entities(query, source_lang)
    
    mappings = {}
    for entity in entities:
        translated = translate_query(entity, target_lang)
        if translated:
            mappings[entity] = [translated]
    
    # Add hardcoded mappings
    common_mappings = _get_common_entity_mappings()
    for entity in entities:
        if entity in common_mappings:
            if source_lang == 'en' and target_lang == 'bn':
                mappings.setdefault(entity, []).extend(common_mappings[entity]['bn'])
            elif source_lang == 'bn' and target_lang == 'en':
                mappings.setdefault(entity, []).extend(common_mappings[entity]['en'])
    
    return {
        'entities': entities,
        'source_lang': source_lang,
        'target_lang': target_lang,
        'mappings': mappings
    }


def _extract_named_entities(text: str, language: str) -> List[str]:
    if language == 'en':
        return _extract_english_entities(text)
    elif language == 'bn':
        return _extract_bangla_entities(text)
    return []


def _extract_english_entities(text: str) -> List[str]:
    entities = []
    try:
        if nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'GPE', 'ORG', 'FAC']:
                    entities.append(ent.text)
    except:
        pass
    
    # Fallback: capitalized words
    if not entities:
        words = text.split()
        entities = [w for w in words if w and w[0].isupper() and len(w) > 2]
    
    return list(set(entities))


def _extract_bangla_entities(text: str) -> List[str]:
    entities = []
    words = text.split()
    for word in words:
        if word.endswith('দেশ') or word.endswith('গ্রাম') or word.endswith('শহর'):
            entities.append(word)
    return list(set(entities))


def _get_common_entity_mappings() -> Dict:
    return {
        'Bangladesh': {'bn': ['বাংলাদেশ'], 'en': ['Bangladesh']},
        'বাংলাদেশ': {'en': ['Bangladesh'], 'bn': ['বাংলাদেশ']},
        'Dhaka': {'bn': ['ঢাকা'], 'en': ['Dhaka']},
        'ঢাকা': {'en': ['Dhaka'], 'bn': ['ঢাকা']},
        'Sheikh Hasina': {'bn': ['শেখ হাসিনা'], 'en': ['Sheikh Hasina']},
        'শেখ হাসিনা': {'en': ['Sheikh Hasina'], 'bn': ['শেখ হাসিনা']},
    }


# ==================== 6. CORPUS SEARCH ====================

class SimpleSearchIndex:
    """TF-IDF search over JSONL articles."""
    
    def __init__(self, articles: List[Dict]):
        self.articles = articles
        self.doc_terms: List[Counter] = []
        self.df: Counter = Counter()
        self.N = len(articles)
        self._build()
    
    def _build(self):
        for art in self.articles:
            tokens = _tokenize_for_index(
                art.get('title', '') + ' ' + art.get('body', ''),
                art.get('language', 'en')
            )
            term_counts = Counter(tokens)
            self.doc_terms.append(term_counts)
            for term in term_counts.keys():
                self.df[term] += 1
    
    def search(self, query: str, language: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        if not query.strip():
            return []

        if language is None:
            language = detect_language(query)

        q_tokens = _tokenize_for_index(query, language)
        scores = []
        
        for idx, doc_counts in enumerate(self.doc_terms):
            score = 0.0
            for term in q_tokens:
                if term not in self.df:
                    continue
                tf = doc_counts.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log((self.N + 1) / (self.df[term] + 1)) + 1
                score += tf * idf
            
            if score > 0:
                scores.append((score, idx))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, idx in scores[:top_k]:
            art = self.articles[idx]
            results.append({
                'title': art.get('title'),
                'url': art.get('url'),
                'language': art.get('language'),
                'score': round(score, 3),
                'snippet': _build_snippet(art.get('body', ''))
            })
        return results


def _tokenize_for_index(text: str, language: str) -> List[str]:
    normalized = normalize_query(text, language=language, remove_stopwords=True)
    return normalized.split()


def _build_snippet(text: str, limit: int = 200) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:limit] + ('…' if len(text) > limit else '')


def load_articles(file_paths: Iterable[str]) -> List[Dict]:
    """Load from JSONL files."""
    articles = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    articles.append(json.loads(line))
                except:
                    continue
    return articles


def search_corpus(query: str, language: Optional[str] = None, corpus_paths: Optional[List[str]] = None, top_k: int = 5) -> Dict:
    """Search news corpus (language-aware)."""
    if language is None:
        language = detect_language(query)

    if corpus_paths is None:
        if language == 'bn':
            corpus_paths = ["data/processed/bn.jsonl"]
        elif language == 'en':
            corpus_paths = ["data/processed/en.jsonl"]
        else:
            corpus_paths = ["data/processed/en.jsonl", "data/processed/bn.jsonl"]
    
    articles = load_articles(corpus_paths)
    if not articles:
        return {'matches': [], 'total_articles': 0}
    
    index = SimpleSearchIndex(articles)
    matches = index.search(query, language=language, top_k=top_k)
    return {'matches': matches, 'total_articles': len(articles)}


# ==================== COMPLETE CLIR PIPELINE ====================

def clir_pipeline(query: str, target_language: str = None) -> Dict:
    """
    Complete 5-step CLIR pipeline.
    
    Args:
        query: Input query
        target_language: 'en' or 'bn' (auto-detect opposite if None)
        
    Returns:
        Dict with all 5 processing steps + search results
    """
    results = {'original_query': query, 'steps': {}}
    
    # STEP 1: Language Detection
    detected_lang = detect_language(query)
    results['steps']['1_language_detection'] = {
        'detected': detected_lang,
        'description': 'Bangla' if detected_lang == 'bn' else 'English' if detected_lang == 'en' else 'Unknown'
    }
    
    # STEP 2: Normalization
    normalized = normalize_query(query, language=detected_lang, remove_stopwords=False)
    normalized_no_stop = normalize_query(query, language=detected_lang, remove_stopwords=True)
    results['steps']['2_normalization'] = {
        'normalized': normalized,
        'normalized_without_stopwords': normalized_no_stop
    }
    
    # Determine target
    if target_language is None:
        target_language = 'bn' if detected_lang == 'en' else 'en'
    
    # STEP 3: Translation (force translate to the opposite corpus for cross-lingual search)
    translation = None
    if detected_lang != target_language:
        translation = translate_query(query, target_language)

    # If translation failed, keep track of that fact
    if translation:
        results['steps']['3_translation'] = {
            'source_lang': detected_lang,
            'target_lang': target_language,
            'translated_query': translation
        }
    else:
        results['steps']['3_translation'] = {
            'note': 'No translation available (will still search both corpora)',
            'source_lang': detected_lang,
            'target_lang': target_language
        }
    
    # STEP 4: Query Expansion
    expanded_original = expand_query(query, detected_lang)
    expanded_translation = expand_query(translation, target_language) if translation else []
    
    results['steps']['4_query_expansion'] = {
        'original_expanded': expanded_original[:15],
        'translation_expanded': expanded_translation[:15]
    }
    
    # STEP 5: Named Entity Mapping
    entity_map = map_named_entities(query, detected_lang, target_language)
    results['steps']['5_entity_mapping'] = entity_map
    
    # BONUS: Search both corpora (original language + opposite language via translation/fallback)
    search_queries = [(query, detected_lang)]

    # Try the translated query in the opposite corpus; if translation failed, still try the original text there
    if translation:
        search_queries.append((translation, target_language))
    else:
        if detected_lang in ('en', 'bn'):
            search_queries.append((query, target_language))
    
    # Search both corpora and keep results separated by language
    en_matches = []
    bn_matches = []
    
    for sq, sq_lang in search_queries:
        matches = search_corpus(sq, language=sq_lang, top_k=5)
        corpus_matches = matches.get('matches', [])
        
        # Separate by language
        for match in corpus_matches:
            if match.get('language') == 'en':
                en_matches.append(match)
            elif match.get('language') == 'bn':
                bn_matches.append(match)
    
    # Deduplicate within each language
    def dedupe_by_url(matches):
        seen = set()
        unique = []
        for m in sorted(matches, key=lambda x: x.get('score', 0), reverse=True):
            if m['url'] not in seen:
                seen.add(m['url'])
                unique.append(m)
        return unique
    
    en_matches = dedupe_by_url(en_matches)
    bn_matches = dedupe_by_url(bn_matches)
    
    results['search_results'] = {
        'english': en_matches[:3],
        'bangla': bn_matches[:3],
        'total_english': len(en_matches),
        'total_bangla': len(bn_matches)
    }
    
    return results


# ==================== INTERACTIVE MODE ====================

def interactive_query():
    """Interactive CLIR query processor."""
    print("=" * 80)
    print("CLIR Query Processor - Cross-Lingual Information Retrieval")
    print("=" * 80)
    print("Type 'exit' or 'quit' to stop\n")
    
    # Load corpus
    corpus_paths = ["data/processed/en.jsonl", "data/processed/bn.jsonl"]
    articles = load_articles(corpus_paths)
    print(f"✓ Loaded {len(articles)} articles from corpus\n")
    
    while True:
        print("\n" + "=" * 80)
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['exit', 'quit', '']:
            print("Goodbye!")
            break
        
        # Ask target language
        print("\nWhich language do you want to work with?")
        print("  1. English (en)")
        print("  2. Bangla (bn)")
        print("  3. Auto-detect and use opposite (default)")
        lang_choice = input("Enter choice (1/2/3): ").strip()
        
        target_lang = None
        if lang_choice == '1':
            target_lang = 'en'
        elif lang_choice == '2':
            target_lang = 'bn'
        
        print("\n" + "=" * 80)
        print("PROCESSING PIPELINE")
        print("=" * 80)
        
        # Run pipeline
        results = clir_pipeline(query, target_language=target_lang)
        
        # Display results
        print(f"\nOriginal Query: {results['original_query']}")
        
        # Step 1
        step1 = results['steps']['1_language_detection']
        print(f"\n[STEP 1] Language Detection")
        print(f"  ✓ Detected: {step1['detected'].upper()} ({step1['description']})")
        
        # Step 2
        step2 = results['steps']['2_normalization']
        print(f"\n[STEP 2] Normalization")
        print(f"  ✓ Normalized: {step2['normalized']}")
        print(f"  ✓ Without Stopwords: {step2['normalized_without_stopwords']}")
        
        # Step 3
        step3 = results['steps']['3_translation']
        print(f"\n[STEP 3] Query Translation")
        if 'translated_query' in step3:
            print(f"  ✓ Translating from {step3['source_lang'].upper()} to {step3['target_lang'].upper()}")
            print(f"  ✓ Original: {results['original_query']}")
            print(f"  ✓ Translated: {step3['translated_query']}")
            print(f"\n  → Translation complete! Proceeding to query expansion...")
        else:
            print(f"  ✓ {step3['note']}")
        
        # Step 4
        step4 = results['steps']['4_query_expansion']
        print(f"\n[STEP 4] Query Expansion")
        if step4['original_expanded']:
            print(f"  ✓ Original Variants: {', '.join(step4['original_expanded'][:5])}...")
        if step4['translation_expanded']:
            print(f"  ✓ Translation Variants: {', '.join(step4['translation_expanded'][:5])}...")
        
        # Step 5
        step5 = results['steps']['5_entity_mapping']
        print(f"\n[STEP 5] Named Entity Mapping")
        if step5['entities']:
            print(f"  ✓ Entities: {', '.join(step5['entities'])}")
            for entity, mappings in step5['mappings'].items():
                print(f"    - {entity} → {', '.join(mappings)}")
        else:
            print(f"  ✓ No named entities detected")
        
        # Search Results
        print(f"\n[SEARCH RESULTS] Top Matches from Both Corpora")
        print("-" * 80)
        
        search_res = results['search_results']
        
        # English Results
        print(f"\n📰 ENGLISH NEWS ({search_res['total_english']} total matches)")
        print("-" * 80)
        if search_res['english']:
            for i, match in enumerate(search_res['english'], 1):
                print(f"\n{i}. {match['title'][:70]}...")
                print(f"   Score: {match['score']}")
                print(f"   Snippet: {match['snippet'][:120]}...")
        else:
            print("  No English matches found.")
        
        # Bangla Results
        print(f"\n📰 BANGLA NEWS ({search_res['total_bangla']} total matches)")
        print("-" * 80)
        if search_res['bangla']:
            for i, match in enumerate(search_res['bangla'], 1):
                print(f"\n{i}. {match['title'][:70]}...")
                print(f"   Score: {match['score']}")
                print(f"   Snippet: {match['snippet'][:120]}...")
        else:
            print("  No Bangla matches found.")


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("=" * 80)
        print("CLIR Test Mode")
        print("=" * 80)
        
        test_queries = [
            ("Kathak dance Bangladesh", None),
            ("শেখ হাসিনা", 'en'),
            ("cinema halls", 'bn'),
        ]
        
        for query, target in test_queries:
            print(f"\nQuery: {query} (target: {target or 'auto'})")
            print("-" * 80)
            results = clir_pipeline(query, target)
            print(f"Detected: {results['steps']['1_language_detection']['detected']}")
            if 'translated_query' in results['steps']['3_translation']:
                print(f"Translation: {results['steps']['3_translation']['translated_query']}")
            
            search_res = results['search_results']
            print(f"English matches: {search_res['total_english']}, Bangla matches: {search_res['total_bangla']}")
            if search_res['english']:
                print(f"Top EN: {search_res['english'][0]['title'][:50]}")
            if search_res['bangla']:
                print(f"Top BN: {search_res['bangla'][0]['title'][:50]}")
    else:
        interactive_query()
