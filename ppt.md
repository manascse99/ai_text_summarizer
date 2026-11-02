# Prompt: create a 6-slide presentation

Create a clean, modern, professional **6-slide presentation** for a college Deep Learning project, using the content below. Audience: a teacher and classmates (viva / project review).

## Design instructions
- Minimal and uncluttered: one idea per slide, large headings, at most 5 short bullets per slide, no long paragraphs.
- Dark navy (#0F172A) background with white text and one accent colour (teal #14B8A6). Use the same font and layout on every slide. Light/white version is fine too if it looks cleaner.
- Use simple diagrams (boxes and arrows) instead of text where possible, especially for the pipeline and architecture.
- Add the technology logos on the tech stack slide: Python, PyTorch, Hugging Face. Use the official logos in high resolution if you can; if you cannot, use clean text wordmarks in the same style and I will replace them with the real logos.
- Add small slide numbers. Keep speaker notes short (2 to 3 lines per slide) so I can present from them.
- Do not invent any numbers, results or features beyond what is written here.

---

## Slide 1: Title
**AI Text Summarizer using NLP and Deep Learning**
Subtitle: Abstractive summarization with a pretrained Transformer (T5-small)
Footer: Your Name | Course / Subject | Date
(Visual idea: long text block turning into a short text block with an arrow.)

## Slide 2: Problem and Objective
- Too much text to read: articles, reports, PDFs.
- Goal: take a paragraph, article or PDF and produce a short, meaningful summary.
- Approach: pretrained Transformer, inference only (no training by us).
- Runs from the terminal on a normal laptop CPU. No website or app.
- Two input modes: paste text in the terminal, or give a PDF path.

## Slide 3: Tech Stack
Show as a row of logo cards, each with a one-line role:
- **Python**: main language
- **PyTorch**: deep learning framework, runs the model on CPU or GPU
- **Hugging Face Transformers**: loads the tokenizer and pretrained model, provides `generate()`
- **T5-small** (`google-t5/t5-small`): pretrained encoder-decoder, about 60M parameters
- **rouge-score**: measures summary quality
- **Pandas** and **pypdf**: results table and PDF text extraction

## Slide 4: How It Works (Pipeline diagram)
Draw as a left-to-right flow of boxes:

Input text or PDF -> Preprocess (clean, add `summarize:` prefix) -> Tokenizer (token IDs + attention mask, max 512 tokens) -> **Encoder** (reads the input with self-attention) -> **Decoder** (writes the summary one token at a time) -> Decode to text -> Summary (or 3 bullets)

Under the diagram, 3 short callouts:
- **Attention mask**: 1 = real token, 0 = padding.
- **Beam search** (4 beams) vs **greedy**: beam search is usually more fluent.
- Long text and PDFs: split into 300-word chunks, summarize each, join.

## Slide 5: Code Structure and Evaluation
Left side, project files:
- `summarizer.py`: loads the model, `summarize()`, PDF and paste input
- `evaluate.py`: ROUGE evaluation on sample articles
- `sample_articles.json`: 3 demo articles with reference summaries
- `requirements.txt`, `README.md`

Right side, evaluation (real run on 3 demo samples):
| Metric | Average F1 |
|---|---|
| ROUGE-1 | 0.268 |
| ROUGE-2 | 0.107 |
| ROUGE-L | 0.226 |

Note under the table: "Demo only (3 samples), not a benchmark. ROUGE measures word overlap, not factual correctness."
(Visual idea: simple bar chart of the three scores.)

## Slide 6: Limitations, Future Work and Conclusion
**Limitations**
- t5-small is small: summaries can be generic, copy sentences, or hallucinate.
- Input limit is 512 tokens per chunk (about 350 words); summaries are about 45 words by default.
- Scanned (image-only) PDFs are not supported.

**Future work**
- Fine-tune on CNN/DailyMail using `Seq2SeqTrainer`.
- Use a larger model (t5-base, bart-large-cnn).
- Second summarizing pass for long documents; OCR for scanned PDFs.

**Conclusion:** A working end-to-end summarizer built with a pretrained encoder-decoder Transformer and evaluated with ROUGE.

Closing line: "Thank you. Questions?"
