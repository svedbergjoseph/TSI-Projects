import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import pandas as pd

# ── Load files ──────────────────────────────────────────────────
with open("transcript1.txt", "r") as f:
    text1 = f.read()

with open("transcript2.txt", "r") as f:
    text2 = f.read()

print("=" * 60)
print("TRANSCRIPT COMPARISON REPORT")
print("=" * 60)

# ── 1. Basic stats ───────────────────────────────────────────────
def word_count(text):
    return len(word_tokenize(text))

print(f"\n📄 Transcript 1 — Words: {word_count(text1)}, Sentences: {len(sent_tokenize(text1))}")
print(f"📄 Transcript 2 — Words: {word_count(text2)}, Sentences: {len(sent_tokenize(text2))}")

# ── 2. TF-IDF Cosine Similarity ──────────────────────────────────
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform([text1, text2])
tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
print(f"\n🔢 TF-IDF Cosine Similarity:      {tfidf_score:.4f}  (0=different, 1=identical)")

# ── 3. Semantic Similarity (Sentence Transformers) ───────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([text1, text2])
semantic_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"🧠 Semantic Similarity Score:      {semantic_score:.4f}  (meaning-based)")

# ── 4. Keyword Overlap ───────────────────────────────────────────
stop_words = set(stopwords.words("english"))

def get_keywords(text):
    tokens = word_tokenize(text.lower())
    return set(t for t in tokens if t.isalpha() and t not in stop_words)

kw1 = get_keywords(text1)
kw2 = get_keywords(text2)
shared = kw1 & kw2
overlap = len(shared) / len(kw1 | kw2)

print(f"\n🔑 Shared Keywords ({len(shared)}): {', '.join(sorted(shared))}")
print(f"📊 Keyword Overlap (Jaccard):      {overlap:.4f}")

# ── 5. Named Entity Comparison (spaCy) ───────────────────────────
nlp = spacy.load("en_core_web_sm")
doc1 = nlp(text1)
doc2 = nlp(text2)

entities1 = {(ent.text, ent.label_) for ent in doc1.ents}
entities2 = {(ent.text, ent.label_) for ent in doc2.ents}
shared_ents = entities1 & entities2

print(f"\n🏷️  Entities in Transcript 1: {entities1 or 'none found'}")
print(f"🏷️  Entities in Transcript 2: {entities2 or 'none found'}")
print(f"✅  Shared Entities:           {shared_ents or 'none'}")

# ── 6. Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
df = pd.DataFrame({
    "Metric": ["TF-IDF Similarity", "Semantic Similarity", "Keyword Jaccard Overlap"],
    "Score":  [f"{tfidf_score:.4f}", f"{semantic_score:.4f}", f"{overlap:.4f}"]
})
print(df.to_string(index=False))
print("=" * 60)