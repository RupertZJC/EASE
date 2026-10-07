# Third-party components

The experiment depends on model checkpoints and detector methods released by
their respective authors. Before publishing a fork or redistributing model
weights, review each upstream model card and license.

- Adversarial Paraphrasing: `https://github.com/chengez/Adversarial-Paraphrasing`
- Fast-DetectGPT: `https://github.com/baoguangsheng/fast-detect-gpt`
- DetectGPT: `https://github.com/eric-mitchell/detect-gpt`
- DNA-GPT: `https://github.com/Xianjun-Yang/DNA-GPT`
- Binoculars: `https://github.com/ahans30/Binoculars`
- MAGE: Hugging Face model `yaful/MAGE`
- RADAR: Hugging Face model `TrustSafeAI/RADAR-Vicuna-7B`
- OpenAI RoBERTa detectors: Hugging Face models under `openai-community`

`src/ease/paraphrase.py` is adapted from the Apache-2.0-licensed Adversarial
Paraphrasing release. The remaining detector wrappers implement the evaluation
equations and fixed model configuration used in the experiments; checkpoints
are downloaded from their upstream locations at runtime and are not bundled.
