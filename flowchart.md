# Project Flowcharts: AI Text Summarizer (NLP + Deep Learning)

This document provides visual flowcharts and architecture diagrams explaining how the **AI Text Summarizer** project works across all scripts, data files, models, and evaluation routines.

---

## Workspace Structure & Component Overview

```
ai-text-summarizer/
├── example.md                          # Sample paragraphs (Technology, Education, Sports) for testing
├── flowchart.md                        # This system flowchart documentation
├── requirements.txt                    # Dependencies (torch, transformers, rouge-score, pandas, pypdf)
├── README.md                           # Documentation, pipeline details, and viva guide summary
├── viva.md                             # Detailed viva preparation guide
├── ppt.md                              # Slide presentation design and script
├── data/
│   └── sample_articles.json            # 3 demo articles + human reference summaries
├── results/
│   └── rouge_scores.csv                # Evaluation results table (ROUGE-1, ROUGE-2, ROUGE-L)
└── src/
    ├── summarizer.py                   # Core inference engine (T5 model, PDF parsing, CLI)
    └── evaluate.py                     # ROUGE scoring benchmark harness
```

---

## 1. High-Level System Architecture

The following diagram illustrates the overall system architecture, showing how input files, core processing scripts, the deep learning model, and output artifacts connect.

```mermaid
flowchart TD
    classDef inputStyle fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef coreStyle fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e;
    classDef modelStyle fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#6b21a8;
    classDef outputStyle fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#15803d;

    subgraph Inputs ["Input Sources"]
        I1["Terminal Paste (sys.stdin)"]:::inputStyle
        I2["CLI Text Argument"]:::inputStyle
        I3["PDF Document (.pdf)"]:::inputStyle
        I4["example.md (Sample Inputs)"]:::inputStyle
        I5["data/sample_articles.json (Benchmark Samples)"]:::inputStyle
    end

    subgraph CoreEngine ["Core Engine (src/)"]
        S1["src/summarizer.py<br/>(Inference & Input Handling)"]:::coreStyle
        E1["src/evaluate.py<br/>(Evaluation Pipeline)"]:::coreStyle
    end

    subgraph ModelHarness ["Deep Learning Framework"]
        M1["Hugging Face Transformers"]:::modelStyle
        M2["Pretrained 'google-t5/t5-small'<br/>(Encoder-Decoder Seq2Seq)"]:::modelStyle
        M3["PyTorch Backend<br/>(Inference Mode / CUDA or CPU)"]:::modelStyle
    end

    subgraph Outputs ["System Outputs"]
        O1["Terminal Output<br/>(Full Summary)"]:::outputStyle
        O2["Terminal Output<br/>(3 Key Bullet Points)"]:::outputStyle
        O3["results/rouge_scores.csv<br/>(ROUGE-1, ROUGE-2, ROUGE-L Table)"]:::outputStyle
    end

    I1 --> S1
    I2 --> S1
    I3 --> S1
    I4 -.->|Manual Test Data| S1
    I5 --> E1

    S1 <--> M1
    M1 <--> M2
    M2 <--> M3

    E1 -->|Calls summarize function| S1

    S1 --> O1
    S1 --> O2
    E1 --> O3
```

---

## 2. Main Execution & Inference Flow (`src/summarizer.py`)

This flowchart details how `src/summarizer.py` starts, handles various input types, performs chunking for long documents, invokes model inference, and formats the output.

```mermaid
flowchart TD
    classDef startStyle fill:#e2e8f0,stroke:#475569,stroke-width:2px;
    classDef procStyle fill:#e0f2fe,stroke:#0284c7,stroke-width:2px;
    classDef branchStyle fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef outStyle fill:#dcfce7,stroke:#16a34a,stroke-width:2px;
    classDef errStyle fill:#fee2e2,stroke:#dc2626,stroke-width:2px;

    Start(["Execution Started: python src/summarizer.py [args]"]):::startStyle --> InitEnv["Initialize Device (cuda if available, else cpu)<br/>Load AutoTokenizer & AutoModelForSeq2SeqLM<br/>Set model.eval()"]:::procStyle

    InitEnv --> ParseInput{"Check Input Arguments (sys.argv)"}:::branchStyle

    ParseInput -->|"Ends with .pdf"| ReadPDF["read_input(): pypdf PdfReader extracts text from all pages"]:::procStyle
    ParseInput -->|"Text string argument"| ReadArg["read_input(): Use text from sys.argv"]:::procStyle
    ParseInput -->|"No arguments provided"| ReadStdin["read_input(): Prompt user and read from sys.stdin"]:::procStyle

    ReadPDF --> CheckEmpty{"Is text empty or whitespace?"}:::branchStyle
    ReadArg --> CheckEmpty
    ReadStdin --> CheckEmpty

    CheckEmpty -->|"Yes"| ExitErr["sys.exit('No text provided.')"]:::errStyle
    CheckEmpty -->|"No"| DisplayOrig["Print ORIGINAL (First 1000 characters)"]:::procStyle

    DisplayOrig --> LongCheck{"Is text long?<br/>(summarize_long)"}:::branchStyle

    LongCheck -->|"Split into chunks of 300 words"| Chunking["Divide text into 300-word segments"]:::procStyle
    Chunking --> ChunkLoop["Iterate each chunk through summarize()"]:::procStyle
    ChunkLoop --> JoinChunks["Join chunk summaries with spaces"]:::procStyle
    JoinChunks --> PrintSum["Print SUMMARY"]:::outStyle

    LongCheck -->|"Direct Call with bullets=True"| CallBullets["summarize(text, bullets=True)"]:::procStyle
    CallBullets --> PrintBullets["Print BULLETS (up to 3 items)"]:::outStyle

    PrintSum --> Done(["Execution Finished"]):::startStyle
    PrintBullets --> Done
```

---

## 3. Deep Learning Summarization Pipeline (`summarize()` function)

The internal processing inside `summarize()` converts raw text into deep learning representations, runs Seq2Seq generation with beam search, and decodes the predictions.

```mermaid
flowchart TD
    classDef procStyle fill:#e0f2fe,stroke:#0284c7,stroke-width:2px;
    classDef branchStyle fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef dlStyle fill:#f3e8ff,stroke:#9333ea,stroke-width:2px;
    classDef outStyle fill:#dcfce7,stroke:#16a34a,stroke-width:2px;
    classDef errStyle fill:#fee2e2,stroke:#dc2626,stroke-width:2px;

    InputText(["Raw Input String"]):::procStyle --> Preproc["Preprocessing:<br/>1. text.split() and ' '.join() to collapse extra whitespaces and newlines<br/>2. Validate text is not empty"]:::procStyle

    Preproc --> CheckValid{"Is cleaned text empty?"}:::branchStyle
    CheckValid -->|"Yes"| RaiseError["Raise ValueError('Input text is empty.')"]:::errStyle
    CheckValid -->|"No"| AddPrefix["Prepend T5 Task Prefix:<br/>'summarize: ' + text"]:::procStyle

    AddPrefix --> Tokenize["SentencePiece Tokenizer:<br/>- truncation=True<br/>- max_length=512 tokens<br/>- return_tensors='pt'<br/>- send tensors to device (CPU/CUDA)"]:::dlStyle

    Tokenize --> GenTensors["Generate Model Inputs:<br/>1. input_ids: Matrix of subword token IDs<br/>2. attention_mask: 1 for real tokens, 0 for padding"]:::dlStyle

    GenTensors --> InfContext["Enter torch.inference_mode():<br/>- Disables gradient calculations<br/>- Minimizes memory usage"]:::dlStyle

    InfContext --> ModelGen["model.generate():<br/>- num_beams=4 (Beam Search)<br/>- max_length=60 (Max summary tokens)<br/>- min_length=15 (Min summary tokens)<br/>- no_repeat_ngram_size=3 (Prevents 3-gram repetition)"]:::dlStyle

    ModelGen --> Decode["tokenizer.decode(output_ids[0], skip_special_tokens=True):<br/>Converts token IDs back into human-readable string"]:::procStyle

    Decode --> CheckBullets{"Is bullets=True?"}:::branchStyle

    CheckBullets -->|"False"| ReturnSummary["Return raw summary string"]:::outStyle
    CheckBullets -->|"True"| RegexSplit["Split by sentence endings using regex:<br/>re.split(r'(?<=[.!?])\s+', summary)[:3]"]:::procStyle
    RegexSplit --> FormatBullets["Format as bullet points:<br/>'\n'.join('- ' + s for s in sentences)"]:::outStyle
    FormatBullets --> ReturnSummary
```

---

## 4. T5-Small Transformer Architecture & Generation Mechanics

This diagram shows what happens inside the `google-t5/t5-small` model during inference: the interaction between the 6-layer Encoder and the 6-layer Decoder with Beam Search.

```mermaid
flowchart LR
    classDef encStyle fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px;
    classDef decStyle fill:#fef3c7,stroke:#b45309,stroke-width:2px;
    classDef searchStyle fill:#f3e8ff,stroke:#7e22ce,stroke-width:2px;
    classDef dataStyle fill:#f1f5f9,stroke:#475569,stroke-width:1.5px;

    subgraph EncoderSide ["T5 Encoder (6 Blocks)"]
        direction TB
        E_In["Token IDs + Positional Encodings"]:::dataStyle
        E_Attn["Bidirectional Self-Attention<br/>(Tokens attend to all tokens)"]:::encStyle
        E_FFN["Feed-Forward Dense Layers + LayerNorm"]:::encStyle
        E_Out["Encoder Hidden States (Context Representation)"]:::encStyle
        E_In --> E_Attn --> E_FFN --> E_Out
    end

    subgraph CrossAttention ["Seq2Seq Bridge"]
        E_Out -->|"Encoder Key & Value Vectors"| D_Cross
    end

    subgraph DecoderSide ["T5 Decoder (6 Blocks)"]
        direction TB
        D_In["Target Token Prefix (Start Token)"]:::dataStyle
        D_Self["Masked Autoregressive Self-Attention<br/>(Tokens only attend to past tokens)"]:::decStyle
        D_Cross["Cross-Attention Layer<br/>(Attends to Encoder Hidden States)"]:::decStyle
        D_FFN["Feed-Forward Dense Layers + LayerNorm"]:::decStyle
        D_LM["LM Head (Linear Projection to Vocab Size 32,128)"]:::decStyle
        D_In --> D_Self --> D_Cross --> D_FFN --> D_LM
    end

    subgraph BeamSearchModule ["Decoding Strategy (Beam Search: num_beams=4)"]
        direction TB
        B_Cand["Select Top-4 Candidate Sequences by Log-Likelihood"]:::searchStyle
        B_Ngram["Apply no_repeat_ngram_size=3 Filter<br/>(Zero out logits of repeated 3-grams)"]:::searchStyle
        B_Term{"Check Terminus:<br/>Token = &lt;/s&gt; OR Length &gt;= 60?"}:::searchStyle
        D_LM --> B_Cand --> B_Ngram --> B_Term
        B_Term -->|"No and Length &lt; 60"| D_In
        B_Term -->|"Yes or Finished"| BestBeam["Return Highest Scoring Sequence"]:::searchStyle
    end
```

---

## 5. Evaluation & ROUGE Benchmarking Flow (`src/evaluate.py`)

This flowchart outlines the quantitative evaluation workflow that validates summary accuracy against human reference summaries.

```mermaid
flowchart TD
    classDef initStyle fill:#e2e8f0,stroke:#475569,stroke-width:2px;
    classDef procStyle fill:#e0f2fe,stroke:#0284c7,stroke-width:2px;
    classDef loopStyle fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef metricStyle fill:#f3e8ff,stroke:#9333ea,stroke-width:2px;
    classDef outStyle fill:#dcfce7,stroke:#16a34a,stroke-width:2px;

    Start(["Start Evaluation: python src/evaluate.py"]):::initStyle --> LoadData["Read data/sample_articles.json<br/>Extract 3 sample records (article, human summary)"]:::procStyle

    LoadData --> InitScorer["Initialize rouge_scorer.RougeScorer:<br/>- Metrics: rouge1, rouge2, rougeL<br/>- Stemmer: Porter Stemmer enabled (use_stemmer=True)"]:::procStyle

    InitScorer --> Loop["For Each Sample in Dataset"]:::loopStyle

    Loop --> CallSum["Run summarize(sample['article'])<br/>(Pretrained model inference only, no training)"]:::procStyle

    CallSum --> ComputeROUGE["scorer.score(reference, prediction):<br/>- ROUGE-1: Unigram overlap F1<br/>- ROUGE-2: Bigram overlap F1<br/>- ROUGE-L: Longest Common Subsequence F1"]:::metricStyle

    ComputeROUGE --> AppendRow["Store prediction string and F1-measures in rows list"]:::procStyle

    AppendRow --> CheckMore{"More samples left?"}:::loopStyle
    CheckMore -->|"Yes"| Loop
    CheckMore -->|"No"| CreateDF["Convert rows list into Pandas DataFrame"]:::procStyle

    CreateDF --> SaveCSV["Save DataFrame to results/rouge_scores.csv"]:::outStyle

    SaveCSV --> DisplayStats["Console Display:<br/>1. Print individual sample ROUGE-1/2/L scores<br/>2. Print Average ROUGE F1 summary<br/>3. Print Demo dataset disclaimer"]:::outStyle

    DisplayStats --> End(["Evaluation Finished"]):::initStyle
```

---

## 6. Data Representation & Tensor Transformation Lifecycle

This state transition flowchart tracks the exact transformation of data types and tensor shapes as an article travels through the system.

```mermaid
flowchart LR
    classDef dataBox fill:#ffffff,stroke:#334155,stroke-width:1.5px;
    classDef tensorBox fill:#ede9fe,stroke:#7c3aed,stroke-width:2px;

    T1["Raw Input<br/><b>Type:</b> str / PDF stream<br/><b>Content:</b> 'Solar power capacity...'"]:::dataBox
    --> T2["Cleaned & Prefixed String<br/><b>Type:</b> str<br/><b>Content:</b> 'summarize: Solar power...'"]:::dataBox
    --> T3["Tokenized Tensors<br/><b>Type:</b> torch.Tensor<br/><b>Shape:</b> [1, input_len &lt;= 512]<br/><b>Tensors:</b> input_ids, attention_mask"]:::tensorBox
    --> T4["Encoder Hidden States<br/><b>Type:</b> torch.FloatTensor<br/><b>Shape:</b> [1, input_len, 512]"]:::tensorBox
    --> T5["Generated Output IDs<br/><b>Type:</b> torch.LongTensor<br/><b>Shape:</b> [1, output_len &lt;= 60]"]:::tensorBox
    --> T6["Decoded Text<br/><b>Type:</b> str<br/><b>Content:</b> 'solar power capacity has grown...'"]:::dataBox
    --> T7["Output Formatting<br/><b>Type:</b> str / List[str] / CSV<br/><b>Outputs:</b> summary, bullets, rouge_scores.csv"]:::dataBox
```

---

## 7. File Mapping & Technical Responsibility Matrix

| File Path | Primary Function | Input | Output | Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| `src/summarizer.py` | Model loading, text/PDF ingestion, tokenization, beam search inference, bullet formatting. | Stdin, CLI string, or `.pdf` file path | Terminal summary text & 3 bullet points | `torch`, `transformers`, `pypdf` |
| `src/evaluate.py` | Benchmark evaluation loop, ROUGE metric calculation, results tabulation. | `data/sample_articles.json` | Console scores & `results/rouge_scores.csv` | `pandas`, `rouge-score`, `src/summarizer.py` |
| `data/sample_articles.json` | Ground-truth reference dataset containing 3 real-world test cases. | JSON text file | Articles and reference summaries for evaluation | Standard library `json` |
| `results/rouge_scores.csv` | Output metric storage for evaluation results. | Pandas DataFrame output from `evaluate.py` | CSV file with columns: `prediction`, `rouge1`, `rouge2`, `rougeL` | `pandas` |
| `example.md` | Playground / test paragraphs (Technology, Education, Sports) for testing CLI manual inputs. | Markdown text | Copy-paste input for manual testing | None |
| `viva.md` | Oral exam preparation documentation covering concepts, limitations, and Q&A. | Markdown text | Study guide | None |
| `ppt.md` | Presentation design guidelines and 6-slide script for DL project demonstration. | Markdown text | Presentation blueprint | None |

---

## 8. Runtime Sequence Diagram: Summarization Flow

This sequence diagram depicts the chronological message and control flow between components when executing `src/summarizer.py`.

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Terminal
    participant CLI as summarizer.py (CLI)
    participant Ingestion as read_input() / pypdf
    participant Engine as summarize() / summarize_long()
    participant Tok as AutoTokenizer (T5)
    participant Torch as PyTorch / T5-Small Model
    participant Post as Regex / Formatting

    User->>CLI: Run python src/summarizer.py [args]
    Note over CLI: Initialize model & tokenizer once globally
    CLI->>Ingestion: read_input(sys.argv)
    alt Argument is a .pdf file
        Ingestion->>Ingestion: PdfReader(source).pages.extract_text()
    else Argument is raw string
        Ingestion-->>CLI: Return raw string
    else No argument provided
        Ingestion->>User: Prompt for text (Enter + Ctrl-D)
        User-->>Ingestion: Interactive multiline input
    end
    Ingestion-->>CLI: Cleaned source text

    alt Text is empty
        CLI-->>User: Exit with "No text provided."
    else Text valid
        CLI->>User: Print original text snippet (first 1000 chars)
        
        opt Full Summary via summarize_long()
            CLI->>Engine: summarize_long(text, words_per_chunk=300)
            loop For each 300-word chunk
                Engine->>Engine: summarize(chunk)
                Engine->>Tok: tokenize("summarize: " + chunk, max_len=512)
                Tok-->>Engine: input_ids, attention_mask
                Engine->>Torch: model.generate(**inputs, num_beams=4)
                Torch-->>Engine: output_ids
                Engine->>Tok: decode(output_ids[0], skip_special_tokens=True)
                Tok-->>Engine: chunk summary string
            end
            Engine-->>CLI: Concatenated summary
            CLI->>User: Display SUMMARY
        end

        opt Bullet Points Summary
            CLI->>Engine: summarize(text, bullets=True)
            Engine->>Torch: model.generate(**inputs, num_beams=4)
            Torch-->>Engine: output_ids
            Engine->>Tok: decode(output_ids[0])
            Tok-->>Engine: raw summary string
            Engine->>Post: re.split(r"(?<=[.!?])\s+", summary)[:3]
            Post-->>Engine: Bullet formatted string
            Engine-->>CLI: Bullet lines
            CLI->>User: Display BULLETS
        end
    end
```

---

## 9. Runtime Sequence Diagram: Evaluation Pipeline

This sequence diagram details the interaction during benchmark evaluation in `src/evaluate.py`.

```mermaid
sequenceDiagram
    autonumber
    actor Evaluator as Developer / Examiner
    participant EvalScript as src/evaluate.py
    participant DataFile as data/sample_articles.json
    participant Engine as src/summarizer.py (summarize)
    participant Scorer as rouge_score (RougeScorer)
    participant Pandas as pandas (DataFrame)
    participant ResultsFile as results/rouge_scores.csv

    Evaluator->>EvalScript: Run python src/evaluate.py
    EvalScript->>DataFile: Load JSON samples
    DataFile-->>EvalScript: 3 demo samples (article, human reference)
    
    EvalScript->>Scorer: Initialize RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    loop For each sample record
        EvalScript->>Engine: summarize(sample["article"])
        Note over Engine: Pretrained inference (No training / weights frozen)
        Engine-->>EvalScript: prediction string
        EvalScript->>Scorer: score(reference, prediction)
        Scorer-->>EvalScript: precision, recall, fmeasure
        EvalScript->>EvalScript: Append prediction + F1 scores to rows list
    end

    EvalScript->>Pandas: pd.DataFrame(rows)
    Pandas-->>EvalScript: Structured DataFrame
    EvalScript->>ResultsFile: results.to_csv("results/rouge_scores.csv", index=False)
    ResultsFile-->>EvalScript: File saved confirmation
    
    EvalScript->>Evaluator: Print individual scores & mean F1 summary to console
```

---

## 10. Edge Case & Error Handling Decision Tree

This diagram outlines how the codebase handles various failure modes and edge cases gracefully.

```mermaid
flowchart TD
    classDef checkStyle fill:#fef3c7,stroke:#d97706,stroke-width:2px;
    classDef safeStyle fill:#dcfce7,stroke:#16a34a,stroke-width:2px;
    classDef alertStyle fill:#fee2e2,stroke:#dc2626,stroke-width:2px;

    InputScenario{"Input Scenario"}:::checkStyle

    InputScenario -->|"Empty String / Whitespace"| E1{"Checked by summarize()"}:::checkStyle
    E1 -->|"text.split() yields empty"| A1["Raises ValueError: 'Input text is empty.'"]:::alertStyle

    InputScenario -->|"No CLI arguments provided"| E2{"Checked by read_input()"}:::checkStyle
    E2 -->|"Interactive Stdin mode"| S2["Prompts user: 'Paste text, then press Enter and Ctrl-D'"]:::safeStyle
    S2 --> E3{"User provides no text"}:::checkStyle
    E3 --> A2["CLI exits with 'No text provided.'"]:::alertStyle

    InputScenario -->|"Document > 512 tokens"| E4{"summarize() vs summarize_long()"}:::checkStyle
    E4 -->|"Direct summarize()"| S3["Tokenizer truncates to 512 tokens (truncation=True)"]:::safeStyle
    E4 -->|"summarize_long()"| S4["Splits into 300-word chunks, summarizes sequentially, and joins"]:::safeStyle

    InputScenario -->|"Repetitive text generation"| E5{"Decoder configuration"}:::checkStyle
    E5 --> S5["no_repeat_ngram_size=3 prevents 3-gram loops"]:::safeStyle

    InputScenario -->|"Hardware acceleration"| E6{"Device Detection"}:::checkStyle
    E6 -->|"torch.cuda.is_available() == True"| S6["Uses CUDA GPU"]:::safeStyle
    E6 -->|"torch.cuda.is_available() == False"| S7["Gracefully falls back to CPU"]:::safeStyle

    InputScenario -->|"Script name collision"| E7{"evaluate.py vs evaluate library"}:::checkStyle
    E7 --> S8["Uses rouge-score directly to avoid Python package shadowing"]:::safeStyle
```

---

## 11. Concrete Example Trace (Solar Power Sample)

An authentic trace from `data/sample_articles.json` through the system into `results/rouge_scores.csv`:

```mermaid
flowchart TD
    classDef stepStyle fill:#e0f2fe,stroke:#0284c7,stroke-width:1.5px;
    classDef resStyle fill:#fef3c7,stroke:#b45309,stroke-width:1.5px;

    subgraph Step1 ["1. Input Article (61 words)"]
        A1["'Solar power capacity has grown rapidly over the past decade as panel prices fell by more than 80 percent. Many countries now install more solar capacity each year than any other source of electricity. However, solar output varies with weather and time of day, so grids need batteries or other backup to stay reliable. Researchers are working on cheaper storage and smarter grid software to handle this variability.'"]:::stepStyle
    end

    subgraph Step2 ["2. T5 Task Prep & Tokenization"]
        A2["Prepended: 'summarize: Solar power capacity...'<br/>Tokenized: [13959, 10, 4843, 624, 2772, 65, 3450, 4015, ...] (77 tokens)"]:::stepStyle
    end

    subgraph Step3 ["3. Beam Search Output (34 words)"]
        A3["'solar power capacity has grown rapidly over the past decade as panel prices fell by more than 80 percent. many countries now install more solar capacity each year than any other source of electricity.'"]:::resStyle
    end

    subgraph Step4 ["4. Ground Truth Human Summary (20 words)"]
        A4["'Solar capacity has surged because panels became much cheaper, but its variable output means grids need better storage and smarter software.'"]:::stepStyle
    end

    subgraph Step5 ["5. Evaluation (ROUGE Overlap)"]
        A5["• ROUGE-1 F1: 0.145 (Unigram matching)<br/>• ROUGE-2 F1: 0.075 (Bigram matching)<br/>• ROUGE-L F1: 0.145 (Longest common subsequence)"]:::resStyle
    end

    Step1 --> Step2 --> Step3
    Step3 --> Step5
    Step4 --> Step5
```

---

## 12. Viva / Defense Quick Reference Guide

Use this chart when explaining the system during oral examination or code defense:

```mermaid
flowchart LR
    classDef qStyle fill:#ede9fe,stroke:#6b21a8,stroke-width:2px;
    classDef aStyle fill:#dcfce7,stroke:#16a34a,stroke-width:2px;

    Q1["Why Transformer over RNN?"]:::qStyle --> A1["Self-attention captures long-range context in parallel, no vanishing gradients"]:::aStyle
    Q2["Extractive or Abstractive?"]:::qStyle --> A2["T5 is an abstractive Seq2Seq model; writes new sentences"]:::aStyle
    Q3["What is the Attention Mask?"]:::qStyle --> A3["Binary tensor (1 = real subword token, 0 = padding)"]:::aStyle
    Q4["Why Beam Search (4 beams)?"]:::qStyle --> A4["Explores multiple sentence trajectories to optimize global fluency"]:::aStyle
    Q5["Why torch.inference_mode()?"]:::qStyle --> A5["Disables autograd tracking, drastically reducing memory & latency"]:::aStyle
    Q6["What is ROUGE?"]:::qStyle --> A6["Statistical n-gram/LCS overlap against human reference summaries"]:::aStyle
```


