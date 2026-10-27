# ConceptCartographer
Upload a textbook chapter (PDF/TXT) -> get an interactive concept map, a searchable study card for every concept, and topics.

## Run
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```
(Run these inside the folder that contains `app.py`. Python 3.10-3.12 is best; on 3.13 Gensim may not install - the app then uses scikit-learn LDA automatically.)

## The simple app (`app.py`) - 3 tabs
1. **Map & search** - type any concept or word from your chapter in the search box: the map zooms to it and its neighbours, with its definition, what to learn first and what comes next. If the word is in your PDF but is not a concept yet, it is added to the map automatically.
2. **Learning path** - pick what you know and what you want to learn; get the chain of steps.
3. **Topics** - concepts grouped by LDA (same colours as the map).

Section headings of your PDF always become concepts. Textbook examples ("For example, ...", quoted sentences) are ignored; add leftovers under *Optional settings -> Ignore these words*.

## Definitions and hover cards
Hover (or click) any bubble to see: a short definition, **how it works**, and **how it connects** to other concepts (part of / includes / uses / used for ...). Every connection comes from the document itself (sentences and section headings).

* **Definitions come from your document first.** A formal "X is ..." sentence is used, or a sentence where X is the subject, or the first sentence of X's own section.
* **Only if the document gives no definition**, the app writes one in simple words:
  * with an **AI key** (sidebar -> *AI definitions*, or set the environment variable `ANTHROPIC_API_KEY`): written by AI, marked "Written by AI - this is not in your document";
  * without a key (or if the AI service fails): built offline from the concept's connections, marked "Built from how it connects in your document".
* AI is never asked about a concept the document already defines (this is tested).

## Library, files and links
* **My library (sidebar):** every document you build is saved automatically (folder `library/` next to the app, or the folder in the environment variable `CC_LIBRARY_DIR`). Click it any time to continue studying - it opens instantly. The trash button asks for a second click to confirm. Concepts you add by searching are saved too.
* **Files:** PDF, PowerPoint (`.pptx`, slide titles become sections and speaker notes are read), Word (`.docx`, Heading styles become sections) and text. Old `.ppt` / `.doc` files must be saved as `.pptx` / `.docx` first.
* **Links:** Wikipedia and other article pages (menus, infoboxes, references are removed; headings become sections), links to PDF / PowerPoint / Word files, **Google Slides / Docs / Drive** share links (set sharing to *Anyone with the link can view*), and **YouTube** videos with captions (hand-made or auto-generated; other languages are translated to English by YouTube when possible; needs `pip install youtube-transcript-api`, already in `requirements.txt`). **Canva, Google Books, OneDrive, Scribd** links cannot be read by any program - download the file (Canva: Share -> Download -> PPTX/PDF) and use *Upload file*.
* Limits: pages that need a login or build their text with JavaScript cannot be read (use *Paste text*); a link is saved as a snapshot of the text at the time you added it; YouTube transcripts have no punctuation so they give simpler maps than documents; links to private/local addresses are blocked for safety. If you run the app on a hosting service whose disk is erased on restart, the library is erased too.

## Advanced app (`app_advanced.py`) - optional
Same engine with extra tabs (triples, JSON frames, language-model perplexity, NLP pipeline inspector, exports):
`streamlit run app_advanced.py`

## Syllabus mapping
Predicate-argument structure -> `ccart/triples.py` · Meaning representation -> `ccart/graph.py` · LDA + coherence -> `ccart/topics.py` · N-grams -> `ccart/ngrams.py`, `ccart/langmodel.py`

## Tests (no browser needed)
`python tests/run_all.py` - includes quality checks on four unrelated documents (biology, history, software, economics) and the AI-definition rule.
