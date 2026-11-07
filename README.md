# AI Text Summarizer (NLP + Deep Learning)

Summarizes a paragraph or article using a pretrained Transformer (`google-t5/t5-small`).
Pure ML/DL project: plain Python scripts, no web app or API.

## Technologies
Python, PyTorch, Hugging Face Transformers, `rouge-score` (ROUGE metric), Pandas (results table), pypdf (PDF text).

## Structure
```
ai-text-summarizer/
├── data/sample_articles.json   # 3 demo articles + reference summaries (NOT a benchmark)
├── src/summarizer.py           # load model, summarize(text)
├── src/evaluate.py             # ROUGE evaluation on the samples
├── results/                    # rouge_scores.csv is written here
├── example.md                  # sample paragraphs for manual testing
├── flowchart.md                # comprehensive architecture and flowcharts
├── ppt.md                      # presentation slides blueprint
├── viva.md                     # viva defense guide
├── requirements.txt
└── README.md
```

## Installation
The first run downloads the model (~240 MB) and needs internet. After that it runs offline on CPU.
```bash
cd ai-text-summarizer
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the summarizer
```bash
python src/summarizer.py                      # paste text, press Enter, then Ctrl-D
python src/summarizer.py "Your text here"     # text as an argument
python src/summarizer.py path/to/file.pdf     # summarize a PDF
```
PDFs and other long texts are split into ~300-word chunks, each chunk is summarized, and the results are joined.
Scanned (image-only) PDFs have no text layer and are not supported.
Or from Python (run from `src/`):
```python
from summarizer import summarize
summarize(text, max_length=60, min_length=15, num_beams=4, bullets=False)
```
- `max_length` / `min_length`: summary length in tokens.
- `num_beams=1`: greedy decoding. `num_beams=4`: beam search.
- `bullets=True`: returns up to 3 bullet points.
- Empty input raises `ValueError`; input longer than 512 tokens is truncated.

## Run evaluation
```bash
python src/evaluate.py
```
Prints ROUGE-1/2/L for each sample plus averages, and saves `results/rouge_scores.csv`.
The model is used as-is (no training), so there is no train/test leakage.
**ROUGE only measures word overlap with a reference. It does not check factual correctness.**
Scores on 3 samples are only a demo, not a benchmark.

> Note: `rouge-score` is used directly instead of the `evaluate` library, because a file named
> `src/evaluate.py` would shadow the `evaluate` package when run as a script.

## How it works (pipeline)
1. **Preprocess**: collapse whitespace; reject empty text.
2. **Tokenize**: text becomes token IDs (SentencePiece subwords). The `summarize:` prefix tells T5 the task.
3. **Attention mask**: 1 for real tokens, 0 for padding, so the model ignores padding.
4. **Truncate**: input capped at 512 tokens (T5-small's training length).
5. **Encoder** reads the full input with self-attention and builds a representation.
6. **Decoder** generates the summary one token at a time, attending to the encoder output (cross-attention).
7. **Decoding strategy**: greedy picks the best token each step (fast, can be dull); beam search keeps the best `num_beams` partial sentences (usually more fluent). `no_repeat_ngram_size=3` stops repetition.
8. **Decode** token IDs back to text. `model.eval()` + `torch.inference_mode()` disable dropout and gradient tracking.

**Abstractive vs extractive:** extractive methods copy existing sentences; T5 is abstractive, it writes new
sentences, which can be more natural but can also **hallucinate** (state things not in the source).

## Limitations and improvements
- t5-small is tiny: summaries can be generic, repetitive, or factually wrong.
- `summarize()` itself reads only the first 512 tokens; `summarize_long()` chunks, but chunk summaries are joined without a second pass, and the bullet output in the terminal uses only the start of long texts.
- The bullet mode just splits the generated summary into sentences.
- Evaluation set is only 3 samples; use a real dataset (e.g. CNN/DailyMail via Hugging Face Datasets) for meaningful scores.
- Fine-tuning later: load a dataset, tokenize articles as inputs and summaries as `labels`, and train with `Seq2SeqTrainer`. Use a larger model (`t5-base`, `bart-large-cnn`) for better quality.

## Viva questions
1. **Why a Transformer instead of an RNN?** Self-attention sees all tokens at once, handles long-range dependencies, and trains in parallel.
2. **What is an attention mask?** A 0/1 tensor marking real tokens vs padding.
3. **Encoder vs decoder?** Encoder understands the input; decoder generates output token by token using cross-attention to the encoder.
4. **Greedy vs beam search?** Greedy takes the top token each step; beam search tracks several candidate sequences and picks the best overall.
5. **What does ROUGE measure?** N-gram (ROUGE-1/2) and longest-common-subsequence (ROUGE-L) overlap with a reference; not meaning or truth.
6. **What is hallucination?** Fluent output containing facts not supported by the input.
7. **Why `torch.inference_mode()`?** No gradients are needed, which saves memory and time.
8. **Extractive vs abstractive?** Extractive selects source sentences; abstractive generates new text.
