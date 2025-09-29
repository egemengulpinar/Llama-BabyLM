# BabyLM LLaMA 

This directory contains two BabyLM challenge models based on the LLaMA architecture:

## Models

### Model A: BabyLM-LLaMA-26M
- **Parameters**: 26 million
- **Training Data**: 2MB (Simple Wikipedia, BNC Spoken, Open Subtitles)
- **BLiMP Accuracy**: 57.2%
- **Model Card**: [A/modelcard.md](A/modelcard.md)

### Model B: BabyLM-LLaMA-66M
- **Parameters**: 66 million
- **Training Data**: 10MB (Simple Wikipedia, BNC Spoken, Open Subtitles)
- **BLiMP Accuracy**: 61.6%
- **Model Card**: [B/modelcard.md](B/modelcard.md)

## Architecture Comparison

| Feature | Model A (26M) | Model B (66M) |
|---------|---------------|---------------|
| Hidden Size | 384 | 512 |
| Layers | 6 | 8 |
| Attention Heads | 6 | 8 |
| Vocabulary | ~16K | ~32K |
| Training Data | 2MB | 10MB |

## Training Data Sources

Both models were trained on a curated subset of the BabyLM dataset consisting of three equally-represented sources:

1. **Simple Wikipedia** - Educational content with simplified language
2. **BNC Spoken** - British National Corpus spoken dialogues
3. **Open Subtitles** - Movie and TV subtitles for conversational patterns

The training data was prepared using a round-robin mixing strategy with equal byte allocation:
- **Model A**: ~667KB per source (total: 2MB)
- **Model B**: ~3.33MB per source (total: 10MB)

## Evaluation Results

Both models were evaluated on the BLiMP (Benchmark of Linguistic Minimal Pairs) dataset:

- **67 linguistic phenomena** tested
- **~2,000 sentence pairs** evaluated
- **Pseudo-log-likelihood ranking** used for minimal pair comparison

### Key Findings

- Model B shows consistent improvement over Model A across most phenomena
- Both models excel at binding constraints (Principle A)
- Areas for improvement include complex syntactic constructions and ellipsis phenomena

## Usage

### Loading Models

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model A
tokenizer_A = AutoTokenizer.from_pretrained("path/to/model/A/final")
model_A = AutoModelForCausalLM.from_pretrained("path/to/model/A/final")

# Model B
tokenizer_B = AutoTokenizer.from_pretrained("path/to/model/B/final")
model_B = AutoModelForCausalLM.from_pretrained("path/to/model/B/final")
```

### Text Generation

```python
input_text = "The linguist discovered that"
inputs = tokenizer_A(input_text, return_tensors="pt")
outputs = model_A.generate(**inputs, max_length=50, temperature=0.7)
generated_text = tokenizer_A.decode(outputs[0], skip_special_tokens=True)
```

## Files Structure

```
work/model/
├── A/
│   ├── final/           # Model A saved model
│   ├── modelcard.md     # Model A documentation
│   └── losses.csv       # Training losses
├── B/
│   ├── final/           # Model B saved model
│   ├── modelcard.md     # Model B documentation
│   └── losses.csv       # Training losses
└── README.md           # This file
```

## Research Impact

These models demonstrate that even small language models can achieve meaningful performance on linguistic tasks when trained on carefully curated, linguistically-relevant data. The comparison between Model A and Model B shows the benefits of:

1. **Increased model capacity** for better linguistic understanding
2. **More training data** for improved vocabulary coverage
3. **Balanced data mixing** for diverse linguistic patterns

## Citation

If you use these models in your research, please cite the BabyLM Challenge:

```bibtex
@inproceedings{babylm2023,
  title={BabyLM Challenge: Efficiency in Language Modeling},
  author={Various},
  booktitle={BabyLM Challenge Workshop at EACL 2024},
  year={2024}
}
```

## License

All models are released under the MIT License.
