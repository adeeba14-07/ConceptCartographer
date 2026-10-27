"""Exercise dependency_triples() with hand-built spaCy-like parses (no spaCy needed)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccart.triples import dependency_triples
from ccart.preprocess import Sentence, Mention, tokenize

class T:
    def __init__(s, i, text, idx, pos, dep, lemma=None):
        s.i, s.text, s.idx, s.pos_, s.dep_ = i, text, idx, pos, dep
        s.lower_ = text.lower(); s.lemma_ = lemma or text.lower()
        s.head = None; s.children = []; s.conjuncts = ()
    @property
    def subtree(self):
        out=[self]
        for c in self.children: out.extend(c.subtree)
        return out

class Doc(list):
    pass

def build(words, spec):
    """words: [(text,pos,lemma)], spec: {i: (head_i, dep)}"""
    toks=[]; idx=0
    for i,(w,pos,lem) in enumerate(words):
        toks.append(T(i,w,idx,pos,spec[i][1],lem)); idx+=len(w)+1
    for i,(h,d) in spec.items():
        toks[i].head = toks[h]
        if h!=i: toks[h].children.append(toks[i])
    return Doc(toks)

def sent_for(doc, mentions):
    text=" ".join(t.text for t in doc)
    s=Sentence(0,text,1,1)
    s.tokens=tokenize(text); s.lemmas=[t[0].lower() for t in s.tokens]
    for (a,b,cid) in mentions:
        s.mentions.append(Mention(a,b,cid,doc[a].idx,doc[b-1].idx+len(doc[b-1].text)))
    return s

# 1 active: "tokenization splits text"? use "model uses tokenization"
d=build([("Model","NOUN","model"),("uses","VERB","use"),("tokenization","NOUN","tokenization")],
        {0:(1,"nsubj"),1:(1,"ROOT"),2:(1,"dobj")})
print(dependency_triples(d, sent_for(d,[(0,1,"model"),(2,3,"tok")])))
# 2 passive with agent: "Probability is computed by Model"
d=build([("Probability","NOUN","probability"),("is","AUX","be"),("computed","VERB","compute"),("by","ADP","by"),("Model","NOUN","model")],
        {0:(2,"nsubjpass"),1:(2,"auxpass"),2:(2,"ROOT"),3:(2,"agent"),4:(3,"pobj")})
print(dependency_triples(d, sent_for(d,[(0,1,"prob"),(4,5,"model")])))
# 3 copula type-of: "Laplace is a type of Smoothing"
d=build([("Laplace","PROPN","laplace"),("is","AUX","be"),("a","DET","a"),("type","NOUN","type"),("of","ADP","of"),("Smoothing","NOUN","smoothing")],
        {0:(1,"nsubj"),1:(1,"ROOT"),2:(3,"det"),3:(1,"attr"),4:(3,"prep"),5:(4,"pobj")})
print(dependency_triples(d, sent_for(d,[(0,1,"lap"),(5,6,"smooth")])))
# 4 prep: "Meaning depends on Structure"
d=build([("Meaning","NOUN","meaning"),("depends","VERB","depend"),("on","ADP","on"),("Structure","NOUN","structure")],
        {0:(1,"nsubj"),1:(1,"ROOT"),2:(1,"prep"),3:(2,"pobj")})
print(dependency_triples(d, sent_for(d,[(0,1,"mean"),(3,4,"struct")])))
# 5 passive no agent: "Perplexity is used for Evaluation"
d=build([("Perplexity","NOUN","perplexity"),("is","AUX","be"),("used","VERB","use"),("for","ADP","for"),("Evaluation","NOUN","evaluation")],
        {0:(2,"nsubjpass"),1:(2,"auxpass"),2:(2,"ROOT"),3:(2,"prep"),4:(3,"pobj")})
print(dependency_triples(d, sent_for(d,[(0,1,"perp"),(4,5,"eval")])))
