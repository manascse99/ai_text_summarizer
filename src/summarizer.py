import re
import sys

import torch
from pypdf import PdfReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "google-t5/t5-small"
MAX_INPUT_TOKENS = 512  # T5-small was trained with 512-token inputs

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()  # turn off dropout: we only do inference


def summarize(text, max_length=60, min_length=15, num_beams=4, bullets=False):
    """Abstractive summary. num_beams=1 is greedy decoding, >1 is beam search."""
    text = " ".join(text.split())  # preprocessing: collapse whitespace/newlines
    if not text:
        raise ValueError("Input text is empty.")

    # Text -> token IDs + attention mask (1 = real token, 0 = padding).
    # T5 is multi-task, so the "summarize:" prefix tells it which task to do.
    inputs = tokenizer("summarize: " + text, return_tensors="pt",
                       truncation=True, max_length=MAX_INPUT_TOKENS).to(device)

    # Encoder reads the input once; decoder then generates output tokens one by one.
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_length=max_length, min_length=min_length,
                                    num_beams=num_beams, no_repeat_ngram_size=3)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if bullets:
        sentences = re.split(r"(?<=[.!?])\s+", summary)[:3]
        return "\n".join("- " + s for s in sentences)
    return summary


def summarize_long(text, words_per_chunk=300, **kwargs):
    """Texts longer than the 512-token limit (e.g. PDFs): summarize each chunk, then join."""
    words = text.split()
    chunks = [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return " ".join(summarize(chunk, **kwargs) for chunk in chunks)


def read_input(source):
    """source is a .pdf path, or '' to paste text into the terminal."""
    if source.lower().endswith(".pdf"):
        return " ".join(page.extract_text() or "" for page in PdfReader(source).pages)
    if source:
        return source
    print("Paste text, then press Enter and Ctrl-D (Ctrl-Z on Windows):")
    return sys.stdin.read()


if __name__ == "__main__":
    text = read_input(" ".join(sys.argv[1:]).strip())
    if not text.strip():
        sys.exit("No text provided.")
    print("ORIGINAL:\n", text[:1000], "...\n" if len(text) > 1000 else "\n")
    print("SUMMARY:\n", summarize_long(text), "\n")
    print("BULLETS:\n", summarize(text, bullets=True))
