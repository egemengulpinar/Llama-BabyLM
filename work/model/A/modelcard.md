---
language: en
license: mit
tags:
- babylm
- llama
- causal-lm
- small-model
- linguistics
model-index:
- name: babylm-llama-26m
  results:
  - task:
      type: text-generation
      name: BLiMP Benchmark
    dataset:
      name: BLiMP (Benchmark of Linguistic Minimal Pairs)
      type: linguistic-minimal-pairs
    metrics:
    - name: Accuracy
      type: acc
      value: 0.572
      verified: false
---

# BabyLM-LLaMA-26M (Model A)

## Model Description

BabyLM-LLaMA-26M is a compact causal language model based on the LLaMA architecture, specifically designed for the BabyLM Challenge. This model represents the smaller variant (Model A) in our comparative study, trained on approximately 2MB of multilingual text data.

### Model Architecture

- **Architecture Type**: Causal Language Model (Decoder-only Transformer)
- **Hidden Size**: 384
- **Number of Layers**: 6
- **Number of Attention Heads**: 6
- **Intermediate Size**: 1,536 (4× hidden size)
- **Context Length**: 512 tokens
- **Vocabulary Size**: ~16,000 tokens (ByteLevel-BPE)
- **Parameters**: ~26 million

## Training Data

The model was trained on a carefully curated 2MB subset of the BabyLM training data, consisting of:

- **Simple Wikipedia** (filtered educational content)
- **BNC Spoken** (British National Corpus spoken dialogues)
- **Open Subtitles** (movie and TV subtitles)

The training corpus was created using a round-robin mixing strategy to ensure balanced representation from all three sources, with equal byte allocation (approximately 667KB per source) to maintain source diversity.

## Training Procedure

### Training Hyperparameters

- **Learning Rate**: 3e-4
- **Training Epochs**: 5
- **Batch Size**: 16 per device (with gradient accumulation ×4)
- **Warmup Steps**: 200
- **Optimizer**: AdamW
- **LR Scheduler**: Cosine annealing
- **Precision**: FP16 (when CUDA available)
- **Gradient Checkpointing**: Enabled

### Training Infrastructure

- **Framework**: HuggingFace Transformers
- **Backend**: PyTorch
- **GPU**: NVIDIA GPU with CUDA support
- **Memory Optimization**: Gradient checkpointing for reduced memory usage

## Evaluation

### BLiMP Benchmark Results

The model was evaluated on the BLiMP (Benchmark of Linguistic Minimal Pairs) dataset, which tests syntactic and semantic understanding through minimal pair comparisons.

- **Overall Accuracy**: 57.2%
- **Test Set Size**: ~2,000 sentence pairs across 67 linguistic phenomena
- **Evaluation Metric**: Pseudo-log-likelihood ranking accuracy

### Phenomenon-Specific Performance

The model shows varying performance across different linguistic phenomena:

- **Strong Performance**: Principle A binding constraints (91-100% accuracy)
- **Moderate Performance**: Subject-verb agreement, quantifier scope
- **Areas for Improvement**: Complex syntactic constructions, ellipsis phenomena

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_path = "babylm-llama-26m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Generate text
input_text = "The cat sat on the"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Limitations and Biases

- **Data Limitation**: Trained on only 2MB of text, which may limit vocabulary coverage and world knowledge
- **Domain Specificity**: Training data is primarily from Wikipedia, spoken transcripts, and subtitles, potentially introducing domain-specific biases
- **Scale Limitation**: As a 26M parameter model, performance may be limited compared to larger language models
- **Evaluation Scope**: Evaluated primarily on syntactic phenomena; semantic and pragmatic capabilities may vary

## Ethical Considerations

This model is intended for research purposes in the BabyLM Challenge. Users should be aware of potential biases inherent in the training data and consider appropriate use cases for linguistic research rather than production applications requiring high accuracy or safety guarantees.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{babylm-llama-26m,
  title={BabyLM-LLaMA-26M: A Compact Language Model for Linguistic Research},
  author={BabyLM Challenge Participant},
  year={2024},
  url={https://babylm.github.io/}
}
```

## License

This model is released under the MIT License.
