# Full results

Macro F1 over all consecutive paragraph pairs. The PAN 2024 test sets come without ground truth, so the
validation splits serve as the held-out evaluation.

## Sentence-Transformers: pair construction approach

| Model | Approach | Easy train | Easy val | Medium train | Medium val | Hard train | Hard val |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| all-MiniLM-L12-v2 | 1 · consecutive | 0.986 | 0.974 | 0.891 | 0.869 | 0.853 | **0.827** |
| all-MiniLM-L12-v2 | 2 · grouped | 0.989 | **0.981** | 0.908 | **0.871** | 0.862 | 0.801 |
| all-mpnet-base-v2 | 1 · consecutive | 0.992 | **0.989** | 0.907 | **0.8804** | 0.861 | **0.839** |
| all-mpnet-base-v2 | 2 · grouped | 0.994 | 0.986 | 0.902 | 0.8798 | 0.866 | 0.835 |

Approach 2 raises training F1 in 5 of 6 cases. On validation it helps MiniLM on easy and medium but costs
2.6 points on hard. For mpnet the two approaches are within 0.5 points everywhere. The larger encoder
gains nothing from the extra pairs. The smaller one overfits them when topic gives no help.

## Sentence-Transformers vs CL-MAWSA

| Model | Easy train | Easy val | Medium train | Medium val | Hard train | Hard val |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| CL-MAWSA | 0.931 | 0.897 | 0.853 | 0.807 | 0.738 | 0.695 |
| all-MiniLM-L12-v2 (1) | 0.986 | 0.974 | 0.891 | 0.869 | 0.853 | 0.827 |
| all-mpnet-base-v2 (1) | 0.992 | 0.989 | 0.907 | 0.880 | 0.861 | 0.839 |

The gap widens with difficulty (9 points on easy, 14 on hard). CL-MAWSA starts from a general-purpose
DeBERTa and must learn pair similarity from scratch. The Sentence-Transformers were pre-trained for
similarity on over a billion sentence pairs, and fine-tuning only has to shift them from *meaning* towards
*style*.

## Prompted LLMs (validation only, no training)

| Model | Easy | Medium | Hard |
|---|:---:|:---:|:---:|
| gpt-3.5-turbo | **0.94044** | **0.871** | **0.8197** |
| llama-3-70b-instruct | 0.92213 | 0.8362 | 0.7682 |
| mixtral-8x7b-instruct | 0.89206 | 0.7976 | 0.7374 |

Notes from running them:
- Outputs had to be post-processed into PAN's `changes` lists. Typical failures: prose before or after the
  JSON, a second JSON object, labels as strings, and lists one element too short or too long.
  `parse_changes` handles the recoverable cases and rejects the rest.
- Mixtral broke the format most often, which is part of its lower score.
- Even at temperature 0, repeated calls on the same document sometimes returned different labels, so every
  answer is cached and runs can be re-scored without new calls.

## Fine-tuned small models vs prompted LLMs

| | Fine-tuned bi-encoder | Prompted LLM |
|---|---|---|
| Best hard F1 | 0.839 (all-mpnet-base-v2) | 0.820 (gpt-3.5-turbo) |
| Setup effort | training per dataset | prompt design on 50 documents |
| Inference | one forward pass per paragraph, local | one API call per document |
| Determinism | identical output every run | can vary between runs |
| Output format | always valid | needs parsing; some answers unusable |

LLMs are the faster way to a reasonable system, with no training data needed. A small fine-tuned encoder is
more accurate on this task, much cheaper to run, and reproducible.
