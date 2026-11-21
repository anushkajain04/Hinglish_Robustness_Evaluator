"""
NLP Utilities & Custom Scikit-Learn Transformers for Hinglish Sentiment Analysis
Includes:
- HINGLISH_CANONICAL_MAP: Comprehensive lexicon mapping Indian texting shorthand and
  phonetic word variants to standardized canonical lemmas.
- LEXICAL_PERTURBATIONS: Reverse mapping for generating realistic Hinglish texting variants.
- HinglishNormalizer: character elongation reduction & canonical lexicon normalization.
- HinglishNegationTagger: clause-bounded bidirectional negation scope tagging (SOV-aware).
- Standalone helper functions for text normalization & negation tagging.
"""

import re
from sklearn.base import BaseEstimator, TransformerMixin

# Comprehensive dictionary mapping common Hinglish shorthand and phonetic spelling
# variants to canonical standard forms.
HINGLISH_CANONICAL_MAP = {
    # Pronouns
    'm': 'main',
    'me': 'main',
    'mein': 'main',
    'mei': 'main',
    'mai': 'main',
    'mjhe': 'mujhe',
    'muje': 'mujhe',
    'mujhko': 'mujhe',
    'mjko': 'mujhe',
    'tm': 'tum',
    'tume': 'tum',
    'ap': 'aap',
    'ye': 'yeh',
    'yh': 'yeh',
    'wo': 'woh',
    'vo': 'woh',
    'voh': 'woh',
    'hm': 'hum',

    # Verbs & Connectors
    'h': 'hai',
    'he': 'hai',
    'hn': 'hai',
    'hain': 'hai',
    'hyn': 'hai',
    'ky': 'kya',
    'kyaa': 'kya',
    'kia': 'kya',
    'kyuki': 'kyunki',
    'kyoki': 'kyunki',
    'qki': 'kyunki',
    'bi': 'bhi',
    'bhii': 'bhi',
    'or': 'aur',
    'kch': 'kuch',
    'kuchh': 'kuch',

    # Intensifiers
    'bht': 'bahut',
    'bohot': 'bahut',
    'boht': 'bahut',
    'bahoot': 'bahut',
    'bahot': 'bahut',
    'bhut': 'bahut',
    'blkl': 'bilkul',
    'bilkool': 'bilkul',
    'bilkull': 'bilkul',
    'ekdam': 'ekdum',
    'ekdm': 'ekdum',
    'ikdam': 'ekdum',
    'jyada': 'zyada',
    'jada': 'zyada',
    'zyaada': 'zyada',
    'zyda': 'zyada',
    'thora': 'thoda',
    'thda': 'thoda',

    # Sentiment Words
    'acha': 'achha',
    'accha': 'achha',
    'achi': 'achha',
    'acchi': 'achha',
    'achhi': 'achha',
    'achchi': 'achha',
    'theek': 'thik',
    'thk': 'thik',
    'tk': 'thik',
    'bkwas': 'bakwas',
    'bakwaas': 'bakwas',
    'bkwass': 'bakwas',
    'bakvas': 'bakwas',
    'bekaar': 'bekar',
    'bkr': 'bekar',
    'bkaar': 'bekar',
    'gatiya': 'ghatiya',
    'ghatia': 'ghatiya',
    'ghatiyya': 'ghatiya',
    'mst': 'mast',
    'masst': 'mast',
    'zbrdst': 'zabardast',
    'jabardast': 'zabardast',
    'jbrdst': 'zabardast',

    # Fillers & Slang
    'yr': 'yaar',
    'yar': 'yaar',
    'yaara': 'yaar',
    'bhaai': 'bhai',
    'bhay': 'bhai',

    # Questions, Interrogatives & Quantifiers
    'sb': 'sab',
    'sabb': 'sab',
    'kb': 'kab',
    'kbb': 'kab',
    'kha': 'kaha',
    'kahan': 'kaha',
    'kahaan': 'kaha',
    'kidhar': 'kaha',
    'kidhr': 'kaha',
    'kdhr': 'kaha',
    'kse': 'kaise',
    'kese': 'kaise',
    'kaisse': 'kaise',
    'kyu': 'kyun',
    'kyon': 'kyun',
    'kyn': 'kyun',
    'q': 'kyun',
    'kon': 'kaun',
    'kn': 'kaun',

    # Time, Adverbs & Auxiliaries
    'abi': 'abhi',
    'abhii': 'abhi',
    'kl': 'kal',
    'aj': 'aaj',
    'kr': 'kar',
    'kro': 'karo',
    'krna': 'karna',
    'rha': 'raha',
    'rhi': 'rahi',
    'rhe': 'rahe',

    # Negations
    'nhi': 'nahi',
    'nai': 'nahi',
    'nahin': 'nahi',
    'nh': 'nahi',
    'nhe': 'nahi',
    'naa': 'na',
    'mt': 'mat',
}

# Reverse mapping: Canonical word -> list of realistic Indian texting shorthand variants
LEXICAL_PERTURBATIONS = {
    # Pronouns
    'main': ['m', 'me', 'mein', 'mei', 'mai'],
    'mujhe': ['mjhe', 'muje', 'mujhko', 'mjko'],
    'tum': ['tm', 'tume'],
    'aap': ['ap'],
    'yeh': ['ye', 'yh'],
    'woh': ['wo', 'vo', 'voh'],
    'hum': ['hm'],

    # Verbs & Connectors
    'hai': ['h', 'he', 'hn', 'hain', 'hyn'],
    'kya': ['ky', 'kyaa', 'kia'],
    'kyunki': ['kyuki', 'kyoki', 'qki'],
    'bhi': ['bi', 'bhii'],
    'aur': ['or'],
    'kuch': ['kch', 'kuchh'],

    # Questions, Quantifiers & Auxiliaries
    'sab': ['sb', 'sabb'],
    'kab': ['kb', 'kbb'],
    'kaha': ['kha', 'kahan', 'kidhar', 'kidhr'],
    'kaise': ['kse', 'kese'],
    'kyun': ['kyu', 'kyon', 'q'],
    'kaun': ['kon', 'kn'],
    'abhi': ['abi'],
    'kal': ['kl'],
    'aaj': ['aj'],
    'kar': ['kr'],
    'karo': ['kro'],
    'karna': ['krna'],
    'raha': ['rha'],
    'rahi': ['rhi'],
    'rahe': ['rhe'],

    # Intensifiers
    'bahut': ['bht', 'bohot', 'boht', 'bahoot', 'bahot', 'bhut'],
    'bilkul': ['blkl', 'bilkool', 'bilkull'],
    'ekdum': ['ekdam', 'ekdm', 'ikdam'],
    'zyada': ['jyada', 'jada', 'zyaada', 'zyda'],
    'thoda': ['thora', 'thda'],

    # Sentiment Words
    'achha': ['acha', 'accha', 'achi', 'acchi', 'achhi', 'achchi'],
    'thik': ['theek', 'thk', 'tk'],
    'bakwas': ['bkwas', 'bakwaas', 'bkwass', 'bakvas'],
    'bekar': ['bekaar', 'bkr', 'bkaar'],
    'ghatiya': ['gatiya', 'ghatia', 'ghatiyya'],
    'mast': ['mst', 'masst'],
    'zabardast': ['zbrdst', 'jabardast', 'jbrdst'],

    # Fillers & Slang
    'yaar': ['yr', 'yar', 'yaara'],
    'bhai': ['bhaai', 'bhay'],

    # Negations
    'nahi': ['nhi', 'nai', 'nahin', 'nh', 'nhe'],
    'na': ['naa'],
    'mat': ['mt'],
}


def normalize_hinglish(text: str) -> str:
    """
    Normalizes excessive character elongations and maps colloquial
    Hinglish texting shorthand and phonetic spelling variants to canonical forms
    using whole-word token matching (ensuring substrings are never corrupted).
    """
    if text is None:
        return ""
    t = str(text).lower()

    # 1. Collapse 3+ consecutive repeating characters to 2 (e.g., 'bahuuuut' -> 'bahuut', 'sooo' -> 'soo')
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)

    # 2. Whole-word token matching against HINGLISH_CANONICAL_MAP
    def _replace_canonical(match):
        w = match.group(0)
        return HINGLISH_CANONICAL_MAP.get(w, w)

    t = re.sub(r'\b[a-zA-Z]+\b', _replace_canonical, t)
    return t


def tag_hinglish_negation(text: str, window: int = 2) -> str:
    """
    Applies clause-bounded bidirectional negation scope tagging.
    Because Hinglish follows SOV (Subject-Object-Verb) word order, adjectives
    and verbs frequently precede negation cues ('thik nahi tha').
    A symmetric window (+/- window tokens) ensures modifiers both before and after
    the cue receive the 'NOT_' prefix within punctuation clause boundaries.
    Applies HINGLISH_CANONICAL_MAP to ensure all negation tokens and surrounding terms
    are standardized.
    """
    if text is None:
        return ""
    norm_text = normalize_hinglish(text)
    neg_cues = {'nahi', 'na', 'mat', 'no', 'not', 'never', 'neither', 'nor', 'bina'}

    # Split text into clauses by punctuation boundaries
    clauses = re.split(r'([.,;!?:\-\n]+)', norm_text)
    result_clauses = []

    for clause in clauses:
        if re.match(r'^[.,;!?:\-\n]+$', clause.strip()):
            result_clauses.append(clause)
            continue

        tokens = clause.split()
        n = len(tokens)
        tagged = [False] * n

        # Find negation cues and mark the +/- window
        for i, token in enumerate(tokens):
            clean_tok = re.sub(r'^\W+|\W+$', '', token).lower()
            clean_tok = HINGLISH_CANONICAL_MAP.get(clean_tok, clean_tok)
            if clean_tok in neg_cues:
                start = max(0, i - window)
                end = min(n, i + window + 1)
                for j in range(start, end):
                    if j != i:
                        tagged[j] = True

        new_tokens = []
        for i, token in enumerate(tokens):
            clean_tok = re.sub(r'^\W+|\W+$', '', token).lower()
            clean_tok = HINGLISH_CANONICAL_MAP.get(clean_tok, clean_tok)
            if tagged[i] and clean_tok not in neg_cues and len(clean_tok) > 0:
                new_tokens.append('NOT_' + token)
            else:
                new_tokens.append(token)
        result_clauses.append(' '.join(new_tokens))

    return ''.join(result_clauses)


class HinglishNormalizer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn Transformer wrapping normalize_hinglish.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_hinglish(x) for x in X]


class HinglishNegationTagger(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn Transformer wrapping tag_hinglish_negation with configurable window.
    """
    def __init__(self, window: int = 2):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [tag_hinglish_negation(x, window=self.window) for x in X]
