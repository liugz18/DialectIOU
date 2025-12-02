# DialectIOU

Abstract
--------
Recent advances in deep learning have improved Automatic Speech Recognition (ASR) and Speech Large Language Models (Speech LLMs). However, Chinese dialectal speech recognition and understanding are hindered by Dialectal Expressions (DEs)—diverse, widely-used, but under-resourced lexical and phrasal items unique to certain dialects. DEs pose distinct challenges due to their pivotal role in sentence comprehension and lack of standardized representation.

This work focuses on two key phenomena: Unwritten Dialectal Syllables (UDS) in DEs, and Dialectal Pronunciation Variants (DPV). We propose a fine-grained and interpretable evaluation toolkit featuring:

1. A novel metric Sino-Dialect CER (SDCER) for UDS-aware ASR evaluation.
2. A task Dialectal Expression Detection (DED) detecting DEs directly from audio.
3. A multiple-choice benchmark Dialect Understanding via MCQs (DMCQ) for contextual dialect understanding.

We release a new dataset and evaluate multiple Speech LLMs to address the challenge of DEs.


Dataset
-------
The dataset released with this toolkit contains audio clips paired with transcriptions and annotations tailored for dialectal expressions. Annotations include:
- DE spans (Dialectal Expressions) within transcripts (marked within 【】)
- UDS tags for syllables lacking standard, authoritative Chinese character representation (marked within <>)

See the `data/` directory (if present) for data files and format. If the data is hosted separately, please place it under `data/` and update paths in `config.py` accordingly.


Quick Start
-----------
1. Ensure required dependencies are installed (Python 3.8+ and common ML libraries). You may want to create a virtual environment.

2. Configure `config.py` to point to your local model and data paths. Key fields:
- `SELECTED_MODEL`: choose from classes within `models/`, e.g. `ParaformerLlmApiModel`
- `AUDIO_BASE_PATH`: base folder for audio dataset.
- `TEXT_FILE_PATH`: path to text annotations, relative to `AUDIO_BASE_PATH`.
- `MODEL_CONFIGS`: model-specific settings.
- `USE_WORD_COMPARISON`: default False, do not change.
- `QUIZZES_PATH`: path to DMCQ quiz json.
- `USE_EXTERNAL_SEGMENT_EVALUATOR`: default True to use SDCER metric, do not change.
- `EXTERNAL_EVALUATOR_FILE`: path to SDCER evaluator, need to clone repo from [here](https://github.com/liugz18/power-asr/tree/Chinese)
- `EXTERNAL_EVALUATOR_CLASS`: default `ChineseSegmentEvaluator`, do not change.
- `USE_DIALECT_EXPLANATIONS`: whether to use DE paraphrases in DMCQ task.

3. Run evaluation scripts in `DialectIOU/`:

# Example usage
# 1. Print current config
python config.py

# 2. Run UDS-aware ASR evaluation along with DE Detection from audio
python main.py

# 3. Run with checkpoint
python eval_w_checkpoint.py

# 4. Run DMCQ benchmark
python answer.py

Contact
-------
For questions or contributions, open an issue or contact the authors via the repository.
