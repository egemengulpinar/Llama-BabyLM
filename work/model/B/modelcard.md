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
- name: babylm-llama-66m
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
      value: 0.616
      verified: false
---

# BabyLM-LLaMA-66M (Model B)

## Model Description

BabyLM-LLaMA-66M is a medium-sized causal language model based on the LLaMA architecture, developed for the BabyLM Challenge. This model represents the larger variant (Model B) in our comparative study, trained on approximately 10MB of multilingual text data.

### Model Architecture

- **Architecture Type**: Causal Language Model (Decoder-only Transformer)
- **Hidden Size**: 512
- **Number of Layers**: 8
- **Number of Attention Heads**: 8
- **Intermediate Size**: 2,048 (4× hidden size)
- **Context Length**: 512 tokens
- **Vocabulary Size**: ~32,000 tokens (ByteLevel-BPE)
- **Parameters**: ~66 million

## Training Data

The model was trained on a curated 10MB subset of the BabyLM training data, consisting of:

- **Simple Wikipedia** (filtered educational content)
- **BNC Spoken** (British National Corpus spoken dialogues)
- **Open Subtitles** (movie and TV subtitles)

The training corpus was created using a round-robin mixing strategy to ensure balanced representation from all three sources, with equal byte allocation (approximately 3.33MB per source) to maintain source diversity, providing 5× more training data compared to the smaller Model A variant.

## Training Procedure

### Training Hyperparameters

- **Learning Rate**: 2e-4 (lower than Model A to accommodate larger size)
- **Training Epochs**: 5
- **Batch Size**: 16 per device (with gradient accumulation ×8)
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

- **Overall Accuracy**: 61.6%
- **Test Set Size**: ~2,000 sentence pairs across 67 linguistic phenomena
- **Evaluation Metric**: Pseudo-log-likelihood ranking accuracy

### Phenomenon-Specific Performance

The model demonstrates improved performance compared to the smaller Model A variant:

- **Strong Performance**: Principle A binding constraints (85-100% accuracy)
- **Good Performance**: Subject-verb agreement, quantifier scope, passive constructions
- **Areas for Improvement**: Complex syntactic constructions, certain ellipsis phenomena

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_path = "babylm-llama-66m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Generate text
input_text = "The researchers conducted an experiment to"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## Model Comparison

This model (Model B) shows improved performance compared to Model A:

| Metric | Model A (26M) | Model B (66M) |
|--------|---------------|---------------|
| Parameters | 26M | 66M |
| Training Data | 2MB | 10MB |
| BLiMP Accuracy | 57.2% | 61.6% |

The larger model benefits from:
- Increased parameter count for better representation capacity
- More extensive training data for improved vocabulary coverage
- Enhanced syntactic understanding across linguistic phenomena

## Limitations and Biases

- **Data Limitation**: While larger than Model A, still trained on limited data compared to full-scale language models
- **Domain Specificity**: Training data sources may introduce domain-specific patterns and biases
- **Scale Consideration**: As a 66M parameter model, performance may still be limited for complex reasoning tasks
- **Evaluation Scope**: Primarily evaluated on syntactic phenomena; broader NLP capabilities may vary

## Ethical Considerations

This model is intended for research purposes in the BabyLM Challenge. The increased size and training data provide better performance but also require consideration of computational costs and potential biases. Users should evaluate appropriateness for their specific use cases.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{babylm-llama-66m,
  title={BabyLM-LLaMA-66M: A Medium-Sized Language Model for Linguistic Research},
  author={BabyLM Challenge Participant},
  year={2024},
  url={https://babylm.github.io/}
}
```

## License

This model is released under the MIT License.
