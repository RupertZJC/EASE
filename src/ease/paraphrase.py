"""
adversarial_paraphrasing.py
Core implementation of the Adversarial Paraphrasing (AP) competitor, migrated
and adapted from the upstream Adversarial-Paraphrasing project to our data.

Architecture:
- LLMModel: singleton wrapper that loads Qwen3-8B once onto GPU 0.
- Paraphraser: runs instruction-following paraphrase decoding while using a
  guidance classifier to score partial outputs in real time.
- OpenAIRoberta: RoBERTa guidance classifier placed on a second GPU.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ======================= LLM Model (loaded once) =======================

class LLMModel:
    """Qwen3-8B model wrapper (singleton) placed on GPU 0."""

    _instance = None

    @torch.no_grad()
    def __init__(self, name="Qwen/Qwen3-8B", device="cuda:0",
                 temperature=1.0, top_p=0.99, top_k=50):
        if LLMModel._instance is not None:
            raise RuntimeError("LLMModel is a singleton. Use get_instance() instead.")
        self.name = name
        self.device = device
        from transformers import pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.name,
            model_kwargs={"torch_dtype": torch.float16},
            device=device,
        )
        self.model = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        # Keep decoding parameters explicit.  AP's upstream defaults target
        # Llama-3; experiments using another generator must record and pass
        # their intended values instead of silently inheriting those defaults.
        self.model.generation_config.top_p = float(top_p)
        self.model.generation_config.top_k = int(top_k)
        self.model.generation_config.temperature = float(temperature)
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        self.model.eval()
        print(f"  [LLMModel] Loaded {self.name} on {device}")

    @classmethod
    def get_instance(cls, name="Qwen/Qwen3-8B", device="cuda:0",
                     temperature=1.0, top_p=0.99, top_k=50):
        """Return the singleton instance without reloading the LLM."""
        if cls._instance is None:
            cls._instance = cls(name, device, temperature, top_p, top_k)
        else:
            actual = cls._instance.model.generation_config
            requested = (float(temperature), float(top_p), int(top_k))
            loaded = (float(actual.temperature), float(actual.top_p), int(actual.top_k))
            if requested != loaded:
                raise RuntimeError(
                    f"LLMModel singleton decoding mismatch: requested={requested}, loaded={loaded}"
                )
        return cls._instance

    @classmethod
    def clear_instance(cls):
        """Free the singleton instance and its GPU memory."""
        if cls._instance is not None:
            del cls._instance.model
            del cls._instance.pipeline
            del cls._instance
            cls._instance = None
            torch.cuda.empty_cache()
            print("  [LLMModel] Instance cleared, GPU memory freed")


# ======================= Detectors =======================

class MAGEDetector:
    """MAGE detector (yaful/MAGE)."""

    @torch.no_grad()
    def __init__(self, device="cuda"):
        model_dir = "yaful/MAGE"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.device = device

    @torch.no_grad()
    def get_scores(self, texts):
        """AI probability scores (higher = more AI-like)."""
        from ease.detectors.mage_preprocess import preprocess
        texts_ = [preprocess(text) for text in texts]
        toks = self.tokenizer(texts_, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model(**toks)
        scores = F.softmax(outputs.logits, dim=-1)
        scores = [score[0].item() for score in scores]
        return np.stack(scores)


class OpenAIRoberta:
    """OpenAI RoBERTa guidance classifier."""

    @torch.no_grad()
    def __init__(self, model_name="openai_roberta_base", device="cuda:1"):
        size = model_name.split("_")[-1]
        model_dir = f"openai-community/roberta-{size}-openai-detector"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        self.device = device
        print(f"  [OpenAIRoberta] Loaded {model_name} on {device}")

    @torch.no_grad()
    def get_scores(self, texts):
        """Fake score; lower score means more human-like (AP convention)."""
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        logits = self.model(**tokenized).logits
        scores = torch.nn.Softmax(dim=1)(logits)  # [[fake, real], ...]
        fake_scores = scores[:, 0].cpu().numpy()  # lower = more human
        return fake_scores


# ======================= Paraphraser =======================

class Paraphraser:
    """Core AP Paraphraser.

    Uses an instruction-following LLM and steers decoding with a guidance
    classifier: for every candidate token it scores the corresponding partial
    text and prefers candidates that look less machine-like.
    """

    def __init__(self, llm_model=None, classifier=None):
        if llm_model is None:
            llm_model = LLMModel.get_instance()
        self.model = llm_model.model
        self.tokenizer = llm_model.tokenizer
        self.classifier = classifier
        print(f"  [Paraphraser] Initialized with classifier={classifier}")

    def paraphrase(self, contents, batch_size=10, max_new_tokens=512, top_p=0.9,
                   adversarial=1.0, deterministic=True,
                   classifier_batch_size=32):
        """Paraphrase texts while guiding generation by the detector score.

        Params:
            contents: list of texts.
            batch_size: generation batch size.
            max_new_tokens: max number of generated tokens.
            top_p: nucleus sampling threshold.
            adversarial: >0 enables adversarial guidance, 0 disables it.
            Guided decoding always uses the released option=2 semantics:
            language-model probability plus proxy-detector score.
            deterministic: True=always pick the lowest-score token, False=sample.
        """
        system_prompt = (
            "You are a rephraser. Given any input text, you are supposed to "
            "rephrase the text without changing its meaning and content, while "
            "maintaining the text quality. Also, it is important for you to output "
            "a rephrased text that has a different style from the input text. You "
            "can not just make a few changes to the input text. The input text is "
            "given below. Print your rephrased output text between tags <TAG> and "
            "</TAG>."
        )
        self.model.generation_config.top_p = top_p

        responses = []
        for b in range(0, len(contents), batch_size):
            inputs = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": content}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                for content in contents[b:b+batch_size]
            ]
            inputs = [inp + "<TAG> " for inp in inputs]

            tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.model.device)
            input_ids = tokenized_inputs["input_ids"]
            attention_mask = tokenized_inputs["attention_mask"]
            past_key_values = None
            finished = torch.zeros(len(input_ids), dtype=torch.bool, device=self.model.device)
            generated_tokens = [[] for _ in range(len(input_ids))]

            for t in range(max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                probs = F.softmax(logits.float() / self.model.generation_config.temperature, dim=-1)

                # Top-p masking.
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                mask = probs_sum - probs_sort > self.model.generation_config.top_p
                probs_sort[mask] = 0.0

                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

                next_tokens, prob_scores = [], []
                for i in range(len(probs_idx)):
                    candidate_tokens = probs_idx[i][probs_sort[i] > 0.0].cpu().detach().numpy().tolist()
                    candidate_probs = probs_sort[i][probs_sort[i] > 0.0].cpu().detach().numpy().tolist()
                    pairs = list(zip(candidate_tokens, candidate_probs))
                    if t == 0:
                        pairs = [pair for pair in pairs if pair[0] != self.tokenizer.eos_token_id]
                    pairs = pairs[:self.model.generation_config.top_k]
                    if not pairs:
                        raise RuntimeError("no valid AP candidate token after EOS filtering")
                    next_tokens.append([pair[0] for pair in pairs])
                    prob_scores.append([pair[1] for pair in pairs])

                # Adversarial guidance: score candidate continuations.
                sampled_tokens = []
                if bool(adversarial) and self.classifier is not None:
                    for i in range(len(next_tokens)):
                        if len(next_tokens[i]) == 1:
                            sampled_tokens.append(next_tokens[i][0])
                            continue

                        # Build the partial text that appends each candidate token.
                        toks = torch.tensor(next_tokens[i], device=self.model.device).unsqueeze(1)
                        inps = torch.tensor(generated_tokens[i], dtype=input_ids.dtype).unsqueeze(0).expand(toks.shape[0], -1).to(self.model.device)
                        next_toks = torch.cat([inps, toks], dim=-1)
                        next_words = self.tokenizer.batch_decode(next_toks, skip_special_tokens=True)

                        # Batch classifier scoring.
                        adv_scores = []
                        for j in range(0, len(next_words), classifier_batch_size):
                            adv_scores.extend(self.classifier.get_scores(next_words[j:j+classifier_batch_size]))

                        # Fixed AP option=2: retain both the source-model
                        # probability and proxy-detector signal.
                        adv_scores = -np.array(prob_scores[i]) + float(adversarial) * np.array(adv_scores)

                        if deterministic:
                            idx = np.argmin(adv_scores)  # lower fake = more human
                            sampled_tokens.append(next_tokens[i][idx])
                        else:
                            adv_scores = -np.array(adv_scores)
                            adv_scores += -adv_scores.min() + 1e-9
                            adv_scores[adv_scores < 0] = 0.0
                            adv_scores /= adv_scores.sum()
                            idx = np.random.choice(len(next_tokens[i]), p=adv_scores)
                            sampled_tokens.append(next_tokens[i][idx])
                else:
                    # Plain sampling without guidance.
                    for i in range(len(next_tokens)):
                        tok = np.random.choice(next_tokens[i], p=prob_scores[i]/np.sum(prob_scores[i]))
                        sampled_tokens.append(tok)

                # Update generation state.
                for i in range(len(input_ids)):
                    if not finished[i]:
                        generated_tokens[i].append(sampled_tokens[i])
                        if sampled_tokens[i] == self.tokenizer.eos_token_id:
                            finished[i] = True
                if finished.all():
                    break

                input_ids = torch.tensor(sampled_tokens, dtype=input_ids.dtype).unsqueeze(1).to(self.model.device)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((len(input_ids), 1), dtype=torch.long, device=self.model.device)],
                    dim=1,
                )

            for i in range(len(input_ids)):
                response_text = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
                response_text = response_text.replace("<TAG>", "").replace("</TAG>", "").strip()
                # Strip the boilerplate remarks occasionally emitted by the model.
                weirds = ["Note: I rephrased", "Note: I've rephrased",
                          "Note: I have rephrased", "(Note:"]
                for weird in weirds:
                    if weird in response_text:
                        response_text = response_text.split(weird)[0].strip()
                        break
                responses.append(response_text)

        return responses
