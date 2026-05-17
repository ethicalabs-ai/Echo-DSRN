"""
echo_dsrn/modeling_generative_clf.py
────────────────────────────────────────────────────────────────────────────
EchoForGenerativeClassification
────────────────────────────────────────────────────────────────────────────
A sequence classifier built on top of EchoForCausalLM using *constrained
scoring* instead of a linear head.

Why a new class instead of EchoForSequenceClassification
─────────────────────────────────────────────────────────
EchoForSequenceClassification uses a single nn.Linear layer seeded from the
lm_head rows. This works perfectly for adapters with single-token labels (e.g.
NSFW: "0" / "1"). For multi-token labels like "weather_query" or
"iot_hue_lightchange" the mean-pooling approximation loses too much signal.

EchoForGenerativeClassification instead computes, for each candidate label L:

    score(L | x) = Σ_t  log P(token_t | x, token_1..t-1)

i.e. the sum of log-probabilities of each token in L conditioned on the input
and the previously generated label tokens. These 60 scores form the logits
returned by forward(), making the model a drop-in AutoModelForSequenceClassification.

HuggingFace API compatibility
──────────────────────────────
• Registered with AutoModelForSequenceClassification via __init__.py
• forward() returns SequenceClassifierOutputWithPast (logits, loss, hidden_states)
• classify(text, tokenizer) convenience method mirrors EchoForSequenceClassification
• from_causal_lm() factory for converting a merged CausalLM checkpoint
• config.auto_map updated by the factory to point here
• No new weights are added — the model is the base lm with the adapter merged

Scoring algorithm
─────────────────
For each label string L_i (from config.id2label):
  1. Tokenise L_i (no BOS/EOS, no special tokens)
  2. Concatenate [input_ids, label_tokens] along seq_len
  3. Run a single forward pass on the combined sequence
  4. Sum log-softmax values at the positions where label tokens are predicted
     (i.e. positions [n_input, n_input+1, ..., n_input+len(label)-1])

This is O(n_labels) forward passes — for 60 MASSIVE intents on short utterances
(avg 7 tokens) it runs in ~60ms on GPU, which is acceptable for inference.
Training is NOT required — the adapter's generative knowledge is used directly.
"""

from __future__ import annotations

import typing
from typing import List, Optional, Tuple, Union

if typing.TYPE_CHECKING:
    # Force HF trust_remote_code to bundle nested dependencies
    pass

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from .configuration_echo import EchoConfig
from .modeling_echo import EchoForCausalLM


class EchoForGenerativeClassification(EchoForCausalLM):
    """
    Intent / multi-label classifier using constrained generative scoring.

    The model is identical to EchoForCausalLM plus:
    • A cache of tokenised label sequences built from config.id2label
    • A forward() that scores all labels and returns (B, num_labels) logits
    • No new trainable parameters

    Usage
    ─────
        from echo_dsrn import EchoForGenerativeClassification
        from transformers import AutoTokenizer

        model = EchoForGenerativeClassification.from_pretrained(
            "ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen",
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(
            "ethicalabs/Echo-SmolTools-114M-Intent-CLF-Gen",
            trust_remote_code=True,
        )
        label, probs = model.classify("What time is it in Tokyo?", tok)
        # → ("datetime_query", tensor([...]))
    """

    # Let HF serialisation know this is a classification model
    _no_split_modules = []

    def __init__(self, config: EchoConfig):
        super().__init__(config)
        # Label token cache — populated lazily on first call that needs it
        # (requires a tokenizer, which we don't have at __init__ time)
        self._label_token_ids: Optional[List[List[int]]] = None

    # ------------------------------------------------------------------
    # Label token cache
    # ------------------------------------------------------------------

    def _build_label_cache(self, tokenizer) -> List[List[int]]:
        """
        Tokenise every label string in config.id2label and cache the result.
        Called lazily on the first forward/classify call.
        """
        cache: List[List[int]] = []
        n = self.config.num_labels
        for idx in range(n):
            label_str = self.config.id2label[idx]
            # Encode without special tokens — the label is a continuation,
            # not a standalone sentence.
            tids = tokenizer.encode(label_str, add_special_tokens=False)
            if not tids:
                raise ValueError(
                    f"Label '{label_str}' (id={idx}) tokenises to an empty sequence. "
                    "Please check your tokenizer and label strings."
                )
            cache.append(tids)
        self._label_token_ids = cache
        return cache

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _score_labels(
        self,
        input_ids: torch.Tensor,  # (B, S)
        attention_mask: torch.Tensor,  # (B, S)
        label_cache: List[List[int]],
    ) -> torch.Tensor:
        """
        Return a (B, num_labels) tensor of log-probability scores.

        For each sample b and label L_i:
            score[b, i] = Σ_t log P(L_i[t] | input_ids[b], L_i[:t])
        """
        device = input_ids.device
        B, S = input_ids.shape
        num_labels = len(label_cache)
        scores = torch.full((B, num_labels), float("-inf"), device=device)

        # 1. Process the prompt once and cache the state
        base_out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        prompt_pkv = base_out.past_key_values
        # Logits predicting the first label token (last token of the prompt)
        prompt_last_logits = base_out.logits[:, -1:, :]  # (B, 1, V)

        # Helper to clone the custom EchoCache safely
        def clone_pkv(pkv):
            if pkv is None:
                return None
            if hasattr(pkv, "states"):
                from .modeling_echo import EchoCache

                new_states = [tuple(t.clone() for t in state_tuple) for state_tuple in pkv.states]
                return EchoCache(new_states)
            elif isinstance(pkv, (list, tuple)):
                return [tuple(t.clone() for t in state_tuple) for state_tuple in pkv]
            return pkv

        for i, label_tids in enumerate(label_cache):
            L = len(label_tids)
            label_t = (
                torch.tensor(label_tids, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
            )  # (B, L)

            # 2. Forward pass ONLY the label tokens using the cloned cache
            label_out = super().forward(
                input_ids=label_t,
                attention_mask=None,  # EchoModel handles this causally for recurrent steps
                past_key_values=clone_pkv(prompt_pkv),
                use_cache=False,
            )

            # 3. Splice the logits together.
            # The first label token is predicted by prompt_last_logits.
            # The remaining label tokens are predicted by label_out.logits[:, :-1, :]
            if L == 1:
                label_logits = prompt_last_logits
            else:
                label_logits = torch.cat(
                    [prompt_last_logits, label_out.logits[:, :-1, :]], dim=1
                )  # (B, L, V)

            log_probs = F.log_softmax(label_logits, dim=-1)  # (B, L, V)

            # Gather log-prob of the correct next token at each label position
            label_t_expanded = label_t.unsqueeze(-1)  # (B, L, 1)
            token_log_probs = log_probs.gather(dim=-1, index=label_t_expanded).squeeze(-1)  # (B, L)

            # Sum log-probs across label tokens → scalar score per sample
            scores[:, i] = token_log_probs.sum(dim=-1)  # (B,)

        return scores  # (B, num_labels) — these are log-prob sums (higher = better)

    # ------------------------------------------------------------------
    # forward()
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # Tokenizer is needed for label scoring; injected via classify() or
        # set once via set_tokenizer().
        _tokenizer=None,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        """
        Returns SequenceClassifierOutputWithPast with:
          • logits  : (B, num_labels) log-prob sums — higher = more likely intent
          • loss    : cross-entropy loss if `labels` is provided, else None

        Note: kwargs are accepted but ignored (past_key_values, use_cache, etc.)
        to maintain drop-in compatibility with the HF pipeline.
        """
        tokenizer = _tokenizer or self._tokenizer_ref
        if tokenizer is None:
            raise RuntimeError(
                "EchoForGenerativeClassification.forward() requires a tokenizer for "
                "label scoring. Either call model.set_tokenizer(tok) once after loading, "
                "or use the classify() convenience method."
            )

        if self._label_token_ids is None:
            self._build_label_cache(tokenizer)

        logits = self._score_labels(input_ids, attention_mask, self._label_token_ids)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
        )

    # ------------------------------------------------------------------
    # Tokenizer binding
    # ------------------------------------------------------------------

    _tokenizer_ref = None  # class-level default; overridden per-instance

    def set_tokenizer(self, tokenizer) -> "EchoForGenerativeClassification":
        """
        Bind a tokenizer so forward() can score label strings.
        Call this once after loading the model:

            model = EchoForGenerativeClassification.from_pretrained(...)
            model.set_tokenizer(tokenizer)
        """
        self._tokenizer_ref = tokenizer
        if self._label_token_ids is None:
            self._build_label_cache(tokenizer)
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Override from_pretrained to automatically load and bind the tokenizer
        from the same checkpoint, so that forward() works out of the box when
        loaded via pipeline() or AutoModelForSequenceClassification without a
        manual set_tokenizer() call.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=kwargs.get("trust_remote_code", False),
            )
            model.set_tokenizer(tok)
        except Exception:
            # Best-effort: if tokenizer loading fails, the user can still call
            # set_tokenizer() manually before running forward().
            pass
        return model

    # ------------------------------------------------------------------
    # High-level inference API
    # ------------------------------------------------------------------

    # Default user-message template; matches what the Intent PEFT was trained on.
    _DEFAULT_USER_TEMPLATE = "Classify the intent of the following request: {utt}"

    def _format_prompts(
        self, texts: List[str], tokenizer, system_prompt: Optional[str], user_template: str
    ) -> List[str]:
        """
        Apply the chat template to a list of raw utterances.
        Falls back to raw text if the tokenizer has no chat template.
        """
        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            return texts  # bare tokenizer — score raw text directly

        formatted = []
        for utt in texts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_template.format(utt=utt)})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted.append(prompt)
        return formatted

    @torch.inference_mode()
    def classify(
        self,
        text: Union[str, List[str]],
        tokenizer,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        max_length: int = 256,
    ) -> Union[
        Tuple[str, torch.Tensor],
        Tuple[List[str], torch.Tensor],
    ]:
        """
        Classify text (or a list of texts) into one of the intent classes.

        The input is formatted with the same chat template used during adapter
        training. The system_prompt and user_template default to the values
        baked into config at merge time.

        Args:
            text          : Raw utterance(s) — no prompt wrapping needed.
            tokenizer     : HF tokenizer for the model.
            system_prompt : Override the system message (optional).
            user_template : Override the user message template (optional).
                            Must contain ``{utt}`` as the utterance placeholder.
            max_length    : Max tokenised length (default 256 to fit chat template).

        Returns (single input):
            (label_str, probs_tensor)   — probs shape (num_labels,)

        Returns (batch input):
            (label_list, probs_tensor)  — probs shape (B, num_labels)
        """
        self.set_tokenizer(tokenizer)
        single = isinstance(text, str)
        texts = [text] if single else text

        # Resolve prompt components: args > config > defaults
        sys_prompt = system_prompt or getattr(self.config, "classification_system_prompt", None)
        usr_template = user_template or getattr(
            self.config, "classification_user_template", self._DEFAULT_USER_TEMPLATE
        )

        # Apply the chat template the adapter was trained with
        formatted = self._format_prompts(texts, tokenizer, sys_prompt, usr_template)

        enc = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(self.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            _tokenizer=tokenizer,
        )
        probs = torch.softmax(out.logits, dim=-1)  # (B, num_labels)

        if single:
            pred_idx = probs[0].argmax().item()
            label_str = self.config.id2label[pred_idx]
            return label_str, probs[0]
        else:
            pred_idxs = probs.argmax(dim=-1).tolist()
            label_strs = [self.config.id2label[i] for i in pred_idxs]
            return label_strs, probs

    # ------------------------------------------------------------------
    # Factory: from_causal_lm
    # ------------------------------------------------------------------

    @classmethod
    def from_causal_lm(
        cls,
        causal_lm_model: EchoForCausalLM,
        num_labels: int,
        id2label: dict,
        label2id: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> "EchoForGenerativeClassification":
        """
        Construct an EchoForGenerativeClassification from a (possibly
        adapter-merged) EchoForCausalLM instance.

        No new weights are added — all parameters come from causal_lm_model.
        The config is updated in-place to record classification metadata.

        Args:
            causal_lm_model : A loaded (and optionally merged) EchoForCausalLM.
            num_labels       : Number of intent classes (e.g. 60 for MASSIVE).
            id2label         : Mapping {int_idx: label_str}.
            label2id         : Optional reverse mapping; auto-derived if None.
            system_prompt    : System message to bake into config (used by classify()).
            user_template    : User message template to bake into config.
                               Must contain ``{utt}`` placeholder.

        Returns:
            EchoForGenerativeClassification ready for inference.
        """
        if label2id is None:
            label2id = {v: int(k) for k, v in id2label.items()}

        config = causal_lm_model.config
        config.num_labels = num_labels
        config.id2label = {int(k): v for k, v in id2label.items()}
        config.label2id = {v: int(k) for k, v in id2label.items()}

        # Bake prompt components into config so the model is self-contained
        if system_prompt is not None:
            config.classification_system_prompt = system_prompt
        if user_template is not None:
            config.classification_user_template = user_template

        # Point auto_map to this module so HF can find it with trust_remote_code
        config.auto_map = {
            **getattr(config, "auto_map", {}),
            "AutoModelForSequenceClassification": (
                "modeling_generative_clf.EchoForGenerativeClassification"
            ),
        }

        # Reuse all weights — no state_dict copy needed, just change the class
        gen_clf = cls(config)
        gen_clf.load_state_dict(causal_lm_model.state_dict())

        # Cast to same dtype as source
        src_dtype = next(causal_lm_model.parameters()).dtype
        gen_clf.to(src_dtype)

        return gen_clf
