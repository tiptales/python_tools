import spacy

nlp = spacy.load('en_core_web_lg')

def most_similar(word, t):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -t]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:100]

def compute(p, t):
    j_ = ' '.join([i.tag_ for i in nlp(p)])
    k_ = ' '.join([w.lower_ for w in most_similar(nlp.vocab[p], t)])
    return(' '.join([str(i) for i in nlp(k_) if i.tag_ == j_]))
