# Viva Guide: AI Text Summarizer

## 1. One-line explanation
We give a paragraph, article or PDF to a **pretrained Transformer (T5-small)**. It reads the text and **writes a new short summary**. We also measure quality with **ROUGE**.

Type: **abstractive text summarization**, using **pretrained model inference** (no training done by us).

---

## 2. Tech stack (what and why)

| Technology | What it is | Why we use it |
|---|---|---|
| **Python** | Programming language | Standard language for ML |
| **PyTorch** (`torch`) | Deep learning framework (tensors, neural networks, GPU support) | T5 runs on it; gives `inference_mode()` and CUDA/CPU selection |
| **Hugging Face Transformers** | Library of pretrained models + tokenizers | Loads T5 and tokenizer in 2 lines; provides `generate()` |
| **google-t5/t5-small** | Pretrained encoder-decoder Transformer (~60M parameters) | Small enough for a laptop CPU, can do summarization |
| **rouge-score** | Implementation of the ROUGE metric | Measures summary quality against a reference |
| **Pandas** | Table library | Holds evaluation results and saves CSV |
| **pypdf** | PDF reader | Extracts text from PDF files |

Not used: web frameworks, databases, APIs (the brief says pure ML/DL).

---

## 3. Project structure
```
ai-text-summarizer/
├── data/sample_articles.json   demo articles + reference summaries
├── src/summarizer.py           the model + summarize function + terminal input
├── src/evaluate.py             ROUGE evaluation
├── results/rouge_scores.csv    evaluation output (created on run)
├── requirements.txt            libraries to install
├── README.md                   how to install/run
└── viva.md                     this file
```

---

## 4. How each file works

### `src/summarizer.py` (the main file)
1. **Imports** torch, Transformers, pypdf.
2. **Loads once**: tokenizer and model `google-t5/t5-small`; picks `cuda` if a GPU exists, else `cpu`; calls `model.eval()` (turns off dropout).
3. **`summarize(text, max_length, min_length, num_beams, bullets)`**
   - Cleans whitespace; raises an error if the text is empty.
   - Adds the prefix `"summarize: "` (T5 is multi-task; the prefix selects the task).
   - Tokenizer converts text into **token IDs** and an **attention mask**; cuts input at **512 tokens**.
   - `model.generate()` inside `torch.inference_mode()` produces output token IDs.
   - `tokenizer.decode()` turns IDs back into text.
   - If `bullets=True`, splits the summary into up to 3 sentences with `-` bullets.
4. **`summarize_long(text)`**: for long text/PDFs, splits into ~300-word chunks, summarizes each, joins them.
5. **`read_input(source)`**: if the argument ends with `.pdf` it reads the PDF; if there is an argument it uses it as text; otherwise it asks you to paste text.
6. **`if __name__ == "__main__"`**: runs when you execute the file; prints original, summary and bullets.

### `src/evaluate.py`
1. Loads `data/sample_articles.json` (articles + human reference summaries).
2. Calls the **same `summarize()`** on every article.
3. Compares prediction with reference using ROUGE-1, ROUGE-2, ROUGE-L (F-measure).
4. Prints per-sample and average scores, saves `results/rouge_scores.csv`.
5. No training happens, so there is **no test-set leakage**.

### `data/sample_articles.json`
3 short articles with reference summaries written by us. Labelled as demo samples, **not a benchmark**.

### `requirements.txt`
`torch, transformers, rouge-score, pandas, pypdf`.

### Why `rouge-score` and not the `evaluate` library?
Our file is named `src/evaluate.py`; running it as a script would make Python import our own file instead of the `evaluate` package (name clash). ROUGE itself is unchanged.

---

## 5. Full pipeline (step by step)
```
Input text / PDF
   -> preprocessing (clean whitespace, check empty, PDF text extraction)
   -> add "summarize: " prefix
   -> tokenization (text -> token IDs) + attention mask, truncate to 512
   -> ENCODER (self-attention reads the whole input)
   -> DECODER (generates one token at a time, cross-attention to encoder)
   -> decoding strategy (greedy or beam search)
   -> decode IDs -> summary text (optionally 3 bullets)
   -> (evaluation) compare with reference using ROUGE
```

---

## 6. Key concepts

- **Tokenization**: splitting text into subword pieces (SentencePiece) and mapping each to an integer ID. The model only understands numbers.
- **Attention mask**: tensor of 1s (real tokens) and 0s (padding) so the model ignores padding.
- **Transformer encoder-decoder**: encoder understands input; decoder generates output. Uses **self-attention** (every token looks at all others) and **cross-attention** (decoder looks at encoder output).
- **Pretrained inference**: the weights were already learned on huge text data (C4 dataset); we only use them for prediction.
- **Greedy vs beam search**: greedy (`num_beams=1`) takes the best token at each step; beam search (`num_beams=4`) keeps 4 candidate sentences and picks the best overall. Beam is usually more fluent but slower.
- **`no_repeat_ngram_size=3`**: blocks repeating the same 3-word sequence.
- **Max input / output tokens**: input limit 512 (model was trained with it); output limited by `max_length` (60) and `min_length` (15).
- **`model.eval()`**: disables dropout. **`torch.inference_mode()`**: disables gradient tracking, saving memory and time.
- **Extractive vs abstractive**: extractive copies existing sentences; abstractive (T5) generates new wording.
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: overlap of single words / word pairs / longest common subsequence between prediction and reference.
- **Hallucination**: fluent output containing facts not in the source.

---

## 7. Results (actual run)
Average ROUGE F1 on the 3 demo samples: **ROUGE-1 0.268, ROUGE-2 0.107, ROUGE-L 0.226**.
Only 3 samples, so this is a demo, not a benchmark. ROUGE measures word overlap only, not truth.

---

## 8. Limitations and improvements
- t5-small is tiny: summaries can be generic, copy sentences, or be wrong. In our test it tends to copy sentences from the input (partly extractive in behaviour).
- Input above 512 tokens is truncated in `summarize()`; `summarize_long()` chunks but does not do a second summarizing pass.
- Scanned (image-only) PDFs are not supported (no text layer; would need OCR).
- Evaluation set is tiny.
- **Improvements**: fine-tune on CNN/DailyMail with `Seq2SeqTrainer`; use a bigger model (`t5-base`, `bart-large-cnn`); add a second summarizing pass for long documents; add OCR; add factual-consistency checks.
- **How fine-tuning would work**: tokenize articles as inputs and summaries as `labels`, train with cross-entropy loss, evaluate on a held-out test set.

---

## 9. Viva questions and answers

**Basics**
1. **What does your project do?** Takes text or a PDF and generates a short summary using a pretrained Transformer.
2. **Is it extractive or abstractive?** Abstractive, as T5 generates text. (Small models often reuse source phrases.)
3. **Did you train a model?** No. We use pretrained weights for inference. Fine-tuning is a future extension.
4. **Why T5-small?** About 60M parameters, runs on a CPU laptop, and supports summarization out of the box.
5. **What is T5?** "Text-to-Text Transfer Transformer": every task is text in, text out, selected by a prefix like `summarize:`.

**Architecture**
6. **Why Transformer instead of RNN/LSTM?** Self-attention handles long-range dependencies and processes tokens in parallel; RNNs are sequential and forget over long text.
7. **Encoder vs decoder?** Encoder builds a representation of the input; decoder generates the output token by token.
8. **What is self-attention?** Each token computes weights over all other tokens to decide what is relevant.
9. **What is cross-attention?** Decoder tokens attend to the encoder's output to stay grounded in the input.
10. **What is an attention mask?** 1 for real tokens, 0 for padding.
11. **What is a token ID?** Integer index of a subword in the vocabulary.

**Code**
12. **What does `model.generate()` do?** Runs the decoder repeatedly to produce output tokens using the chosen decoding strategy.
13. **Why `model.eval()`?** Turns off dropout so output is deterministic and consistent.
14. **Why `torch.inference_mode()`?** No gradients needed; faster and uses less memory.
15. **How do you choose GPU/CPU?** `torch.cuda.is_available()`; use CUDA if true, else CPU.
16. **Why the `summarize:` prefix?** T5 was trained on many tasks; the prefix tells it which one.
17. **How is empty input handled?** `summarize()` raises `ValueError`; the terminal script prints "No text provided."
18. **How do you handle long text?** Truncation to 512 tokens in `summarize()`; chunking in `summarize_long()`.
19. **Greedy vs beam search?** Greedy: one best token per step. Beam: tracks several sequences, chooses the best overall.
20. **What do `max_length` and `min_length` do?** Bound the summary length in tokens.

**Evaluation**
21. **What is ROUGE?** Metric comparing generated summary with a reference by n-gram overlap.
22. **ROUGE-1, -2, -L?** Unigram overlap, bigram overlap, longest common subsequence.
23. **Precision, recall, F-measure?** Precision = overlap / words in prediction; recall = overlap / words in reference; F = their harmonic mean.
24. **Limitation of ROUGE?** It checks wording, not meaning or factual correctness; a correct paraphrase can score low.
25. **Why are your scores low?** Tiny model, only 3 samples, abstractive paraphrasing, short single reference.
26. **Is there test-set leakage?** No: the model is pretrained and not trained on our samples.

**Limitations**
27. **What is hallucination?** The model states things not in the source.
28. **Biggest limitation of your project?** Small model quality and 512-token input limit.
29. **How would you improve it?** Fine-tune, bigger model, hierarchical summarization, OCR, factual checks.
30. **How would you fine-tune?** Load a summarization dataset, tokenize inputs and labels, train with `Seq2SeqTrainer`, evaluate on a held-out set with ROUGE.

---

## 10. Commands to demo
```bash
cd "/Users/manassrivastava/Desktop/dl ca/ai-text-summarizer"
source venv/bin/activate
python src/summarizer.py                  # paste text, Enter, Ctrl-D
python src/summarizer.py "/path/to/file.pdf"
python src/evaluate.py
```
