import os
import copy
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from transformers import __version__
from transformers.generation import (
    GenerationMixin, 
    GenerateDecoderOnlyOutput, 
    GenerateEncoderDecoderOutput, 
    GenerateBeamDecoderOnlyOutput, 
    GenerateBeamEncoderDecoderOutput,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.utils import PushToHubMixin, is_torchdynamo_compiling, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module

from kvarena.methods.KVDict.cache_utils import KVDictCacheConfig, KVDictCache

GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]

METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")
NEEDS_CACHE_CONFIG = {}
NEED_SETUP_CACHE_CLASSES_MAPPING = {}
QUANT_BACKEND_CLASSES_MAPPING = {}
ALL_CACHE_IMPLEMENTATIONS = []

if is_torch_available():
    from transformers.cache_utils import (
        Cache,
        HQQQuantizedCache,
        HybridCache,
        MambaCache,
        OffloadedStaticCache,
        QuantizedCacheConfig,
        QuantoQuantizedCache,
        SlidingWindowCache,
        StaticCache,
        StaticCacheConfig,
    )
    from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor, WatermarkLogitsProcessor

    NEEDS_CACHE_CONFIG["quantized"] = QuantizedCacheConfig
    NEEDS_CACHE_CONFIG["static"] = StaticCacheConfig
    NEEDS_CACHE_CONFIG["kvdict"] = KVDictCacheConfig
    NEED_SETUP_CACHE_CLASSES_MAPPING = {
        "static": StaticCache,
        "offloaded_static": OffloadedStaticCache,
        "sliding_window": SlidingWindowCache,
        "hybrid": HybridCache,
        "mamba": MambaCache,
    }
    QUANT_BACKEND_CLASSES_MAPPING = {"quanto": QuantoQuantizedCache, "HQQ": HQQQuantizedCache}
    ALL_CACHE_IMPLEMENTATIONS = list(NEED_SETUP_CACHE_CLASSES_MAPPING.keys()) + list(NEEDS_CACHE_CONFIG.keys())

class GenerationConfig(PushToHubMixin):
    # no-format
    """
    Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
    for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *contrastive search* if `penalty_alpha>0.` and `top_k>1`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-search decoding* if `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`
        - *dola decoding* if `dola_layers` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

    <Tip>

    A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
    the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
    description of the possible manipulations, as well as examples of their usage.

    </Tip>

    Arg:
        > Parameters that control the length of the output

        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        min_length (`int`, *optional*, defaults to 0):
            The minimum length of the sequence to be generated. Corresponds to the length of the input prompt +
            `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
        min_new_tokens (`int`, *optional*):
            The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        max_time (`float`, *optional*):
            The maximum amount of time you allow the computation to run for in seconds. generation will still finish
            the current pass after allocated time has been passed.
        stop_strings (`str or List[str]`, *optional*):
            A string or a list of strings that should terminate generation if the model outputs them.

        > Parameters that control the generation strategy used

        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        penalty_alpha (`float`, *optional*):
            The values balance the model confidence and the degeneration penalty in contrastive search decoding.
        dola_layers (`str` or `List[int]`, *optional*):
            The layers to use for DoLa decoding. If `None`, DoLa decoding is not used. If a string, it must
            be one of "low" or "high", which means using the lower part or higher part of the model layers, respectively.
            "low" means the first half of the layers up to the first 20 layers, and "high" means the last half of the
            layers up to the last 20 layers.
            If a list of integers, it must contain the indices of the layers to use for candidate premature layers in DoLa.
            The 0-th layer is the word embedding layer of the model. Set to `'low'` to improve long-answer reasoning tasks,
            `'high'` to improve short-answer tasks. Check the [documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/generation_strategies.md)
            or [the paper](https://arxiv.org/abs/2309.03883) for more details.

        > Parameters that control the cache

        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        cache_implementation (`str`, *optional*, default to `None`):
            Name of the cache class that will be instantiated in `generate`, for faster decoding. Possible values are:
            {ALL_CACHE_IMPLEMENTATIONS}. We support other cache types, but they must be manually instantiated and
            passed to `generate` through the `past_key_values` argument. See our
            [cache documentation](https://huggingface.co/docs/transformers/en/kv_cache) for further information.
        cache_config (`CacheConfig` or `dict`, *optional*, default to `None`):
            Arguments used in the key-value cache class can be passed in `cache_config`. Can be passed as a `Dict` and
            it will be converted to its repsective `CacheConfig` internally.
            Otherwise can be passed as a `CacheConfig` class matching the indicated `cache_implementation`.
        return_legacy_cache (`bool`, *optional*, default to `True`):
            Whether to return the legacy or new format of the cache when `DynamicCache` is used by default.

        > Parameters for manipulation of the model output logits

        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
            the 0.99-0.8 range (use the opposite of normal `top_p` values).
        typical_p (`float`, *optional*, defaults to 1.0):
            Local typicality measures how similar the conditional probability of predicting a target token next is to
            the expected conditional probability of predicting a random token next, given the partial text already
            generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
            add up to `typical_p` or higher are kept for generation. See [this
            paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
        epsilon_cutoff (`float`, *optional*, defaults to 0.0):
            If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
            `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
            size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
        eta_cutoff (`float`, *optional*, defaults to 0.0):
            Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
            0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) *
            exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token
            probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3,
            depending on the size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
        diversity_penalty (`float`, *optional*, defaults to 0.0):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        encoder_repetition_penalty (`float`, *optional*, defaults to 1.0):
            The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the
            original input. 1.0 means no penalty.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        bad_words_ids (`List[List[int]]`, *optional*):
            List of list of token ids that are not allowed to be generated. Check
            [`~generation.NoBadWordsLogitsProcessor`] for further documentation and examples.
        force_words_ids (`List[List[int]]` or `List[List[List[int]]]`, *optional*):
            List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple list of
            words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`, this
            triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one
            can allow different forms of each word.
        renormalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors (including the custom
            ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits
            are normalized but some logit processors break the normalization.
        constraints (`List[Constraint]`, *optional*):
            Custom constraints that can be added to the generation to ensure that the output will contain the use of
            certain tokens as defined by `Constraint` objects, in the most sensible way possible.
        forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
            multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
            language token.
        forced_eos_token_id (`int` or List[int]`, *optional*, defaults to `model.config.forced_eos_token_id`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
        remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
            Note that using `remove_invalid_values` can slow down generation.
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where
            penalty starts and `decay_factor` represents the factor of exponential decay
        suppress_tokens (`List[int]`, *optional*):
            A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their
            log probs to `-inf` so that they are not sampled.
        begin_suppress_tokens  (`List[int]`, *optional*):
            A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit
            processor will set their log probs to `-inf` so that they are not sampled.
        forced_decoder_ids (`List[List[int]]`, *optional*):
            A list of pairs of integers which indicates a mapping from generation indices to token indices that will be
            forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token
            of index 123.
        sequence_bias (`Dict[Tuple[int], float]`, *optional*)):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. Check
            [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples.
        token_healing (`bool`, *optional*, defaults to `False`):
            Heal tail tokens of prompts by replacing them with their appropriate extensions.
            This enhances the quality of completions for prompts affected by greedy tokenization bias.
        guidance_scale (`float`, *optional*):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        low_memory (`bool`, *optional*):
            Switch to sequential beam search and sequential topk for contrastive search to reduce peak memory.
            Used with beam search and contrastive search.
        watermarking_config (`BaseWatermarkingConfig` or `dict`, *optional*):
            Arguments used to watermark the model outputs by adding a small bias to randomly selected set of "green"
            tokens. See the docs of [`SynthIDTextWatermarkingConfig`] and [`WatermarkingConfig`] for more
            details. If passed as `Dict`, it will be converted to a `WatermarkingConfig` internally.

        > Parameters that define the output variables of generate

        num_return_sequences (`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`, *optional*):
            Whether or not to return the unprocessed prediction logit scores. See `logits` under returned tensors for
            more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`], as opposed to returning exclusively the generated
            sequence. This flag must be set to `True` to return the generation cache (when `use_cache` is `True`)
            or optional outputs (see flags starting with `output_`)

        > Special tokens that can be used at generation time

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        > Generation parameters exclusive to encoder-decoder models

        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        decoder_start_token_id (`int` or `List[int]`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token or a list of length
            `batch_size`. Indicating a list enables different start ids for each element in the batch
            (e.g. multilingual models with different target languages in one batch)

        > Generation parameters exclusive to assistant generation
        is_assistant (`bool`, *optional*, defaults to `False`):
            Whether the model is an assistant (draft) model.
        num_assistant_tokens (`int`, *optional*, defaults to 20):
            Defines the number of _speculative tokens_ that shall be generated by the assistant model before being
            checked by the target model at each iteration. Higher values for `num_assistant_tokens` make the generation
            more _speculative_ : If the assistant model is performant larger speed-ups can be reached, if the assistant
            model requires lots of corrections, lower speed-ups are reached.
        num_assistant_tokens_schedule (`str`, *optional*, defaults to `"constant"`):
            Defines the schedule at which max assistant tokens shall be changed during inference.
            - `"heuristic"`: When all speculative tokens are correct, increase `num_assistant_tokens` by 2 else
              reduce by 1. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model.
            - `"heuristic_transient"`: Same as `"heuristic"` but `num_assistant_tokens` is reset to its initial value after each generation call.
            - `"constant"`: `num_assistant_tokens` stays unchanged during generation
        assistant_confidence_threshold (`float`, *optional*, defaults to 0.4):
            The confidence threshold for the assistant model. If the assistant model's confidence in its prediction for the current token is lower
            than this threshold, the assistant model stops the current token generation iteration, even if the number of _speculative tokens_
            (defined by `num_assistant_tokens`) is not yet reached. It is an unsupervised version of the dynamic speculation lookahead
            from Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models <https://arxiv.org/abs/2405.04304>.
        prompt_lookup_num_tokens (`int`, *optional*, default to `None`):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*, default to `None`):
            The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.

        > Wild card

        generation_kwargs:
            Additional generation kwargs will be forwarded to the `generate` function of the model. Kwargs that are not
            present in `generate`'s signature will be used in the model forward pass.
    """

    extra_output_flags = ("output_attentions", "output_hidden_states", "output_scores", "output_logits")

    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)

        # Parameters that control the cache
        self.use_cache = kwargs.pop("use_cache", True)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)
        if self.cache_implementation is not None and self.cache_implementation in NEEDS_CACHE_CONFIG:
            cache_config_class = NEEDS_CACHE_CONFIG[self.cache_implementation]
            if self.cache_config is None:
                self.cache_config = cache_config_class()
            elif isinstance(self.cache_config, dict):
                self.cache_config = cache_config_class.from_dict(self.cache_config)
        self.return_legacy_cache = kwargs.pop("return_legacy_cache", None)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.constraints = kwargs.pop("constraints", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", False)
        self.guidance_scale = kwargs.pop("guidance_scale", None)
        self.low_memory = kwargs.pop("low_memory", None)
        watermarking_config = kwargs.pop("watermarking_config", None)
        if watermarking_config is None:
            self.watermarking_config = None
        elif isinstance(watermarking_config, BaseWatermarkingConfig):
            self.watermarking_config = watermarking_config
        else:
            self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.is_assistant = False
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 20)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", "constant")
        self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", 0.4)

        # Prompt lookup decoding
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        self_without_metadata = self.to_json_string(use_diff=False, ignore_metadata=True)
        other_without_metadata = other.to_json_string(use_diff=False, ignore_metadata=True)
        return self_without_metadata == other_without_metadata

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"

    def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
        # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
        # property and part of the `__repr__`
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams == 1:
            if self.do_sample is False:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if self.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif self.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Assisted generation may extend some generation modes
        if assistant_model is not None or self.prompt_lookup_num_tokens is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )

        # DoLa generation may extend some generation modes
        if self.dola_layers is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.DOLA_GENERATION
            else:
                raise ValueError(
                    "You've set `dola_layers`, which triggers DoLa generate. Currently, DoLa generate "
                    "is only supported with Greedy Search and Sample."
                )
        return generation_mode

    def validate(self, is_init=False):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters not validated here are best validated at generate runtime, as they may depend on
        other inputs and/or the model, such as parameters related to the generation length.

        Arg:
            is_init (`bool`, *optional*, defaults to `False`):
                Whether the validation is performed during the initialization of the instance.
        """

        # Validation of individual attributes
        if self.early_stopping not in {True, False, "never"}:
            raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"`max_new_tokens` must be greater than 0, but is {self.max_new_tokens}.")
        if self.pad_token_id is not None and self.pad_token_id < 0:
            warnings.warn(
                f"`pad_token_id` should be positive but got {self.pad_token_id}. This will cause errors when batch "
                "generating, if there is padding. Please set `pad_token_id` explicitly as "
                "`model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation"
            )

        # Validation of attribute relations:
        fix_location = ""
        if is_init:
            fix_location = (
                " This was detected when initializing the generation config instance, which means the corresponding "
                "file may hold incorrect parameterization and should be fixed."
            )

        # 1. detect sampling-only parameterization when not in sampling mode
        if self.do_sample is False:
            greedy_wrong_parameter_msg = (
                "`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only "
                "used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."
                + fix_location
            )
            if self.temperature is not None and self.temperature != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="temperature", flag_value=self.temperature),
                    UserWarning,
                )
            if self.top_p is not None and self.top_p != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="top_p", flag_value=self.top_p),
                    UserWarning,
                )
            if self.min_p is not None:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="min_p", flag_value=self.min_p),
                    UserWarning,
                )
            if self.typical_p is not None and self.typical_p != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="typical_p", flag_value=self.typical_p),
                    UserWarning,
                )
            if (
                self.top_k is not None and self.top_k != 50 and self.penalty_alpha is None
            ):  # contrastive search uses top_k
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="top_k", flag_value=self.top_k),
                    UserWarning,
                )
            if self.epsilon_cutoff is not None and self.epsilon_cutoff != 0.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="epsilon_cutoff", flag_value=self.epsilon_cutoff),
                    UserWarning,
                )
            if self.eta_cutoff is not None and self.eta_cutoff != 0.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(flag_name="eta_cutoff", flag_value=self.eta_cutoff),
                    UserWarning,
                )

        # 2. detect beam-only parameterization when not in beam mode
        if self.num_beams is None:
            warnings.warn("`num_beams` is set to None - defaulting to 1.", UserWarning)
            self.num_beams = 1

        if self.num_beams == 1:
            single_beam_wrong_parameter_msg = (
                "`num_beams` is set to 1. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used "
                "in beam-based generation modes. You should set `num_beams>1` or unset `{flag_name}`." + fix_location
            )
            if self.early_stopping is not False:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="early_stopping", flag_value=self.early_stopping),
                    UserWarning,
                )
            if self.num_beam_groups is not None and self.num_beam_groups != 1:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="num_beam_groups", flag_value=self.num_beam_groups
                    ),
                    UserWarning,
                )
            if self.diversity_penalty is not None and self.diversity_penalty != 0.0:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="diversity_penalty", flag_value=self.diversity_penalty
                    ),
                    UserWarning,
                )
            if self.length_penalty is not None and self.length_penalty != 1.0:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="length_penalty", flag_value=self.length_penalty),
                    UserWarning,
                )
            if self.constraints is not None:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(flag_name="constraints", flag_value=self.constraints),
                    UserWarning,
                )

        # 3. detect incorrect paramaterization specific to advanced beam modes
        else:
            # constrained beam search
            if self.constraints is not None or self.force_words_ids is not None:
                constrained_wrong_parameter_msg = (
                    "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. However, "
                    "`{flag_name}` is set to `{flag_value}`, which is incompatible with this generation mode. Set "
                    "`constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue." + fix_location
                )
                if self.do_sample is True:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(flag_name="do_sample", flag_value=self.do_sample)
                    )
                if self.num_beam_groups is not None and self.num_beam_groups != 1:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(
                            flag_name="num_beam_groups", flag_value=self.num_beam_groups
                        )
                    )
            # group beam search
            if self.diversity_penalty != 0.0 or self.num_beam_groups != 1:
                group_error_prefix = (
                    "`diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. In "
                    "this generation mode, "
                )
                if self.do_sample is True:
                    raise ValueError(group_error_prefix + "`do_sample` must be set to `False`")
                if self.num_beams % self.num_beam_groups != 0:
                    raise ValueError(group_error_prefix + "`num_beams` should be divisible by `num_beam_groups`")
                if self.diversity_penalty == 0.0:
                    raise ValueError(
                        group_error_prefix
                        + "`diversity_penalty` should be greater than `0.0`, otherwise your groups will be identical."
                    )
            # DoLa generation
            if self.dola_layers is not None and (self.repetition_penalty is None or self.repetition_penalty < 1.2):
                warnings.warn(
                    "`dola_layers` is set to trigger DoLa decoding, but `repetition_penalty` is set to a value of "
                    f"{self.repetition_penalty}, which could induce unwanted repetition. The recommended value for "
                    "DoLa decoding is `repetition_penalty>=1.2`.",
                    UserWarning,
                )

        # 4. check `num_return_sequences`
        if self.num_return_sequences != 1:
            if self.num_beams == 1:
                if self.do_sample is False:
                    raise ValueError(
                        "Greedy methods without beam search do not support `num_return_sequences` different than 1 "
                        f"(got {self.num_return_sequences})."
                    )
            elif self.num_return_sequences > self.num_beams:
                raise ValueError(
                    f"`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` "
                    f"({self.num_beams})."
                )

        # 5. check cache-related arguments
        if self.cache_implementation is not None and self.cache_implementation not in ALL_CACHE_IMPLEMENTATIONS:
            raise ValueError(
                f"Invalid `cache_implementation` ({self.cache_implementation}). Choose one of: "
                f"{ALL_CACHE_IMPLEMENTATIONS}"
            )
        if self.cache_config is not None:
            cache_class = NEEDS_CACHE_CONFIG.get(self.cache_implementation)
            if cache_class is None:
                raise ValueError(
                    "You provided a `cache_config` but the cache implementation you are using "
                    f"({self.cache_implementation}) does not require any config. Make sure to use the "
                    "correct cache implementation matching your cache config."
                )
            if not isinstance(self.cache_config, cache_class):
                self.cache_config = cache_class.from_dict(self.cache_config)
            self.cache_config.validate()
        if self.use_cache is False:
            # In this case, all cache-related arguments should be unset. However, since `use_cache=False` is often used
            # passed to `generate` directly to hot-fix cache issues, let's raise a warning instead of an error
            # (otherwise a user might need to overwrite several parameters).
            no_cache_warning = (
                "You have set `use_cache` to `False`, but {cache_arg} is set to {cache_arg_value}. {cache_arg} will "
                "have no effect."
            )
            for arg_name in ("cache_implementation", "cache_config", "return_legacy_cache"):
                if getattr(self, arg_name) is not None:
                    logger.warning_once(
                        no_cache_warning.format(cache_arg=arg_name, cache_arg_value=getattr(self, arg_name)),
                        UserWarning,
                    )

        # 6.  check watermarking arguments
        if self.watermarking_config is not None:
            if not (
                isinstance(self.watermarking_config, WatermarkingConfig)
                or isinstance(self.watermarking_config, SynthIDTextWatermarkingConfig)
            ):
                warnings.warn(
                    "`watermarking_config` as a dict is deprecated. Please construct `watermarking_config` object with "
                    "`WatermarkingConfig` or `SynthIDTextWatermarkingConfig` class.",
                    FutureWarning,
                )
                self.watermarking_config = WatermarkingConfig.from_dict(self.watermarking_config)
            self.watermarking_config.validate()

        # 7. other incorrect combinations
        if self.return_dict_in_generate is not True:
            for extra_output_flag in self.extra_output_flags:
                if getattr(self, extra_output_flag) is True:
                    warnings.warn(
                        f"`return_dict_in_generate` is NOT set to `True`, but `{extra_output_flag}` is. When "
                        f"`return_dict_in_generate` is not `True`, `{extra_output_flag}` is ignored.",
                        UserWarning,
                    )

        # 8. check common issue: passing `generate` arguments inside the generation config
        generate_arguments = (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
        )
        for arg in generate_arguments:
            if hasattr(self, arg):
                raise ValueError(
                    f"Argument `{arg}` is not a valid argument of `GenerationConfig`. It should be passed to "
                    "`generate()` (or a pipeline) directly."
                )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        r"""
        Save a generation configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~GenerationConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be saved in `save_directory`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        # At save time, validate the instance -- if any warning/exception is thrown, we refuse to save the instance.
        # This strictness is enforced to prevent bad configurations from being saved and re-used.
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.validate()
            if len(caught_warnings) > 0:
                raise ValueError(str([w.message for w in caught_warnings]))
        except ValueError as exc:
            raise ValueError(
                "The generation config instance is invalid -- `.validate()` throws warnings and/or exceptions. "
                "Fix these issues to save the configuration.\n\nThrown during validation:\n" + str(exc)
            )

        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. "
                "Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "GenerationConfig":
        r"""
        Instantiate a [`GenerationConfig`] from a generation configuration file.

        Args:
            pretrained_model_name (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~GenerationConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be loaded from `pretrained_model_name`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        >>> from transformers import GenerationConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> generation_config = GenerationConfig.from_pretrained("openai-community/gpt2")

        >>> # E.g. config was saved using *save_pretrained('./test/saved_model/')*
        >>> generation_config.save_pretrained("./test/saved_model/")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")

        >>> # You can also specify configuration names to your generation configuration file
        >>> generation_config.save_pretrained("./test/saved_model/", config_file_name="my_configuration.json")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")

        >>> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
        >>> # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
        >>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        ...     "openai-community/gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
        ... )
        >>> generation_config.top_k
        1

        >>> unused_kwargs
        {'foo': False}
        ```"""
        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        config_path = os.path.join(pretrained_model_name, config_file_name)
        config_path = str(config_path)

        is_local = os.path.exists(config_path)
        if os.path.isfile(os.path.join(subfolder, config_path)):
            # Special case when config_path is a local file
            resolved_config_file = config_path
            is_local = True
        elif is_remote_url(config_path):
            configuration_file = config_path
            resolved_config_file = download_url(config_path)
        else:
            configuration_file = config_file_name
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        if kwargs.get("return_unused_kwargs") is True:
            config, unused_kwargs = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)  # Hash to detect whether the instance was modified
            return config, unused_kwargs
        else:
            config = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)  # Hash to detect whether the instance was modified
            return config

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # The line below allows model-specific config to be loaded as well through kwargs, with safety checks.
        # See https://github.com/huggingface/transformers/pull/21269
        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Fields to ignore at serialization time
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]

        # Transformers version when serializing this file
        output["transformers_version"] = __version__

        self.dict_torch_dtype_to_str(output)
        return output

    def to_json_string(self, use_diff: bool = True, ignore_metadata: bool = False) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.
            ignore_metadata (`bool`, *optional*, defaults to `False`):
                Whether to ignore the metadata fields present in the instance

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        def convert_dataclass_to_dict(obj):
            if isinstance(obj, dict):
                return {key: convert_dataclass_to_dict(value) for key, value in obj.items()}
            elif is_dataclass(obj):
                return obj.to_dict()
            else:
                return obj

        config_dict = convert_keys_to_string(config_dict)
        config_dict = convert_dataclass_to_dict(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`]. This function is useful to convert legacy
        [`PretrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PretrainedConfig`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        config_dict = model_config.to_dict()
        config_dict.pop("_from_model_config", None)

        # Removes all `None` from the model config dict -- this lets the generation config defaults to take hold
        config_dict = {key: value for key, value in config_dict.items() if value is not None}

        generation_config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        # Special case: some models have generation attributes set in the decoder. Use them if still unset in the
        # generation config (which in turn is defined from the outer attributes of model config).
        decoder_config = model_config.get_text_config(decoder=True)
        if decoder_config is not model_config:
            default_generation_config = GenerationConfig()
            decoder_config_dict = decoder_config.to_dict()
            for attr in generation_config.to_dict().keys():
                is_unset = getattr(generation_config, attr) == getattr(default_generation_config, attr)
                if attr in decoder_config_dict and is_unset:
                    setattr(generation_config, attr, decoder_config_dict[attr])

        # If any `output_...` flag is set to `True`, we ensure `return_dict_in_generate` is set to `True`.
        if generation_config.return_dict_in_generate is False:
            if any(
                getattr(generation_config, extra_output_flag, False)
                for extra_output_flag in generation_config.extra_output_flags
            ):
                generation_config.return_dict_in_generate = True

        # Hash to detect whether the instance was modified
        generation_config._original_object_hash = hash(generation_config)
        return generation_config

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Confirm that the updated instance is still valid
        self.validate()

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

class GenerationMixinKVDict(GenerationMixin):
    # def _prepare_generation_config(
    #     self, generation_config: Optional[GenerationConfig], **kwargs: Dict
    # ) -> Tuple[GenerationConfig, Dict]:
    #     """
    #     Prepares the base generation config, then applies any generation configuration options from kwargs. This
    #     function handles retrocompatibility with respect to configuration files.
    #     """
    #     # TODO joao: when we can detect `fullgraph=True` in `torch.compile` (https://github.com/pytorch/pytorch/pull/120400)
    #     # replace `is_torchdynamo_compiling` by the corresponding check. As it is, we are being too restrictive with
    #     # the parameterization in `fullgraph=False` so as to enable `fullgraph=True`.

    #     # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    #     using_model_generation_config = False
    #     if generation_config is None:
    #         # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
    #         # the following conditions must be met
    #         # 1) the generation config must have been created from the model config (`_from_model_config` field);
    #         # 2) the generation config must have seen no modification since its creation (the hash is the same);
    #         # 3) there are non-default generation parameters in the model config.
    #         # 4) the user must have set new generation parameters in the model config.
    #         # NOTE: `torch.compile` can't compile `hash`, this legacy support is disabled with compilation.
    #         if (
    #             not is_torchdynamo_compiling()
    #             and self.generation_config._from_model_config  # 1)
    #             and self.generation_config._original_object_hash == hash(self.generation_config)  # 2)
    #             and len(self.config._get_non_default_generation_parameters()) > 0  # 3)
    #         ):
    #             new_generation_config = GenerationConfig.from_model_config(self.config)
    #             if new_generation_config != self.generation_config:  # 4)
    #                 warnings.warn(
    #                     "You have modified the pretrained model configuration to control generation. This is a"
    #                     " deprecated strategy to control generation and will be removed in v5."
    #                     " Please use and modify the model generation configuration (see"
    #                     " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )",
    #                     UserWarning,
    #                 )
    #                 self.generation_config = new_generation_config

    #         generation_config = self.generation_config
    #         using_model_generation_config = True

    #     # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
    #     # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
    #     # exception will be raised in `_validate_model_kwargs`
    #     if not is_torchdynamo_compiling():
    #         generation_config = copy.deepcopy(generation_config)
    #         model_kwargs = generation_config.update(**kwargs)
    #         # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
    #         if not using_model_generation_config:
    #             if generation_config.bos_token_id is None:
    #                 generation_config.bos_token_id = self.generation_config.bos_token_id
    #             if generation_config.eos_token_id is None:
    #                 generation_config.eos_token_id = self.generation_config.eos_token_id
    #             if generation_config.pad_token_id is None:
    #                 generation_config.pad_token_id = self.generation_config.pad_token_id
    #             if generation_config.decoder_start_token_id is None:
    #                 generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id
    #     else:
    #         model_kwargs = kwargs

    #     return generation_config, model_kwargs

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s parameterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """

        cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        # Quick escape route 1: if the user specifies a cache, we only need to:
        # a) check for conflicting `generate` arguments
        # b) convert to the new cache format (if the user passes a legacy cache and model supports it)
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            if generation_config.cache_implementation is not None:
                raise ValueError(
                    f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                    "Cache object) is unsupported. Please use only one of the two."
                )
            if isinstance(user_defined_cache, tuple) and self._supports_default_dynamic_cache():
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(user_defined_cache)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(user_defined_cache)
                )
            return

        # Quick escape route 2: if the user specifies no cache is to be used. (conflicting arguments are handled in
        # `generation_config.validate()`)
        if generation_config.use_cache is False:
            return

        # Quick escape route 3: model that only supports legacy caches = nothing to prepare
        if not self._supports_default_dynamic_cache():
            if generation_config.cache_implementation is not None:
                warnings.warn(
                    "This model does not support `Cache` instances, it only supports the legacy cache format (tuple "
                    f"of tuples). `cache_implementation` (set to {generation_config.cache_implementation}) will be "
                    "ignored.",
                    UserWarning,
                )
            return

        # Otherwise we NEED to prepare a cache, based on `generation_config.cache_implementation`

        # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
        # which is only supported in dynamic caches atm
        if assistant_model is not None and generation_config.cache_implementation is not None:
            logger.warning_once(
                "An assistant model is provided, using a dynamic cache instead of a cache of type="
                f"'{generation_config.cache_implementation}'."
            )
            generation_config.cache_implementation = None

        if generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
                    max_cache_len=max_cache_length,
                    device=device,
                    model_kwargs=model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue and tag @zucchini-nlp."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not (is_optimum_quanto_available() or is_quanto_available()):
                    raise ImportError(
                        "You need to install optimum-quanto in order to use KV cache quantization with optimum-quanto backend. "
                        "Please install it via  with `pip install optimum-quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs[cache_name] = cache_class(cache_config)
            elif generation_config.cache_implementation == "offloaded":
                model_kwargs[cache_name] = OffloadedCache()
            elif generation_config.cache_implementation == "kvdict":
                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else KVDictCacheConfig()
                )
                model_kwargs[cache_name] = (KVDictCache(cache_config))

        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        else:
            model_kwargs[cache_name] = (
                DynamicCache()
                if not requires_cross_attention_cache
                else EncoderDecoderCache(DynamicCache(), DynamicCache())
            )

    # def _get_cache(
    #     self, cache_implementation: str, batch_size: int, max_cache_len: int, device: torch.device, model_kwargs
    # ) -> Cache:
    #     """
    #     Sets a cache for `generate`, that will persist across calls. A new cache will only be initialized a
    #     new `generate` call requires a larger cache or uses a different batch size.

    #     Returns the resulting cache object.
    #     """
    #     cache_cls: Cache = NEED_SETUP_CACHE_CLASSES_MAPPING[cache_implementation]
    #     requires_cross_attention_cache = (
    #         self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
    #     )

    #     if hasattr(self, "_cache"):
    #         cache_to_check = self._cache.self_attention_cache if requires_cross_attention_cache else self._cache

    #     if cache_implementation == "sliding_window":
    #         max_cache_len = min(self.config.sliding_window, max_cache_len)

    #     need_new_cache = (
    #         not hasattr(self, "_cache")
    #         or (not isinstance(cache_to_check, cache_cls))
    #         or cache_to_check.batch_size != batch_size
    #     )
    #     if cache_implementation != "mamba":
    #         need_new_cache = need_new_cache or cache_to_check.max_cache_len < max_cache_len

    #     if requires_cross_attention_cache and hasattr(self, "_cache"):
    #         need_new_cache = (
    #             need_new_cache
    #             or self._cache.cross_attention_cache.max_cache_len != model_kwargs["encoder_outputs"][0].shape[1]
    #         )

    #     if need_new_cache:
    #         if hasattr(self.config, "_pre_quantization_dtype"):
    #             cache_dtype = self.config._pre_quantization_dtype
    #         else:
    #             if not is_torchdynamo_compiling():
    #                 cache_dtype = self.dtype
    #             else:
    #                 # NOTE: self.dtype is not compatible with torch.compile, as it calls `self.parameters()`.
    #                 # Workaround: trust the lm_head, whose attribute name is somewhat consistent across generative
    #                 # models. May cause trobles with non-text modalities.
    #                 cache_dtype = self.get_output_embeddings().weight.dtype

    #         def get_layer_device_map(execution_device_map: Optional[dict] = None):
    #             if execution_device_map is None:
    #                 return None
    #             elif len(execution_device_map) == 1 and "" in execution_device_map:
    #                 return {idx: execution_device_map[""] for idx in range(self.config.num_hidden_layers)}
    #             layer_device_map = {}
    #             for layer in execution_device_map:
    #                 for idx in range(self.config.num_hidden_layers):
    #                     if f".{idx}." in f"{layer}.":
    #                         layer_device_map[idx] = execution_device_map[layer]
    #                         break
    #             for idx in range(self.config.num_hidden_layers):
    #                 if idx not in layer_device_map:
    #                     raise RuntimeError(f"layer {idx} has not been mapped to a device.")
    #             return layer_device_map

    #         execution_device_map = None
    #         # Taken from dispatch_model from accelerate.
    #         # This is needed here if we don't want to make changes in accelerate in order to save execution_device
    #         # For offloaded case, we need to get the execution device, not just the device where it is offloaded
    #         if hasattr(self, "hf_device_map"):
    #             main_device = [d for d in self.hf_device_map.values() if d not in ["cpu", "disk"]][0]
    #             execution_device_map = {
    #                 name: main_device if device in ["cpu", "disk"] else device
    #                 for name, device in self.hf_device_map.items()
    #             }
    #         layer_device_map = get_layer_device_map(execution_device_map)

    #         cache_kwargs = {
    #             "config": self.config.get_text_config(),
    #             "batch_size": batch_size,
    #             "max_cache_len": max_cache_len,
    #             "device": device,
    #             "dtype": cache_dtype,
    #             "layer_device_map": layer_device_map,
    #         }
    #         print(self.config.get_text_config())
    #         self._cache = cache_cls(**cache_kwargs)
    #         if requires_cross_attention_cache:
    #             encoder_kwargs = cache_kwargs.copy()
    #             encoder_kwargs["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
    #             self._cache = EncoderDecoderCache(self._cache, cache_cls(**encoder_kwargs))
    #     else:
    #         self._cache.reset()
    #     return self._cache    

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     generation_config: Optional[GenerationConfig] = None,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #     synced_gpus: Optional[bool] = None,
    #     assistant_model: Optional["PreTrainedModel"] = None,
    #     streamer: Optional["BaseStreamer"] = None,
    #     negative_prompt_ids: Optional[torch.Tensor] = None,
    #     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     r"""

    #     Generates sequences of token ids for models with a language modeling head.

    #     <Tip warning={true}>

    #     Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    #     model's default generation configuration. You can override any `generation_config` by passing the corresponding
    #     parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    #     For an overview of generation strategies and code examples, check out the [following
    #     guide](../generation_strategies).

    #     </Tip>

    #     Parameters:
    #         inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
    #             The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
    #             method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
    #             should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
    #             `input_ids`, `input_values`, `input_features`, or `pixel_values`.
    #         generation_config ([`~generation.GenerationConfig`], *optional*):
    #             The generation configuration to be used as base parametrization for the generation call. `**kwargs`
    #             passed to generate matching the attributes of `generation_config` will override them. If
    #             `generation_config` is not provided, the default will be used, which has the following loading
    #             priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
    #             configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
    #             default values, whose documentation should be checked to parameterize generation.
    #         logits_processor (`LogitsProcessorList`, *optional*):
    #             Custom logits processors that complement the default logits processors built from arguments and
    #             generation config. If a logit processor is passed that is already created with the arguments or a
    #             generation config an error is thrown. This feature is intended for advanced users.
    #         stopping_criteria (`StoppingCriteriaList`, *optional*):
    #             Custom stopping criteria that complements the default stopping criteria built from arguments and a
    #             generation config. If a stopping criteria is passed that is already created with the arguments or a
    #             generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
    #             sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
    #             intended for advanced users.
    #         prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
    #             If provided, this function constraints the beam search to allowed tokens only at each step. If not
    #             provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
    #             `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
    #             on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
    #             for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
    #             Retrieval](https://arxiv.org/abs/2010.00904).
    #         synced_gpus (`bool`, *optional*):
    #             Whether to continue running the while loop until max_length. Unless overridden, this flag will be set
    #             to `True` if using `FullyShardedDataParallel` or DeepSpeed ZeRO Stage 3 with multiple GPUs to avoid
    #             deadlocking if one GPU finishes generating before other GPUs. Otherwise, defaults to `False`.
    #         assistant_model (`PreTrainedModel`, *optional*):
    #             An assistant model that can be used to accelerate generation. The assistant model must have the exact
    #             same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistant model
    #             is much faster than running generation with the model you're calling generate from. As such, the
    #             assistant model should be much smaller.
    #         streamer (`BaseStreamer`, *optional*):
    #             Streamer object that will be used to stream the generated sequences. Generated tokens are passed
    #             through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
    #         negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             The negative prompt needed for some processors such as CFG. The batch size must match the input batch
    #             size. This is an experimental feature, subject to breaking API changes in future versions.
    #         negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Attention_mask for `negative_prompt_ids`.
    #         kwargs (`Dict[str, Any]`, *optional*):
    #             Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
    #             forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
    #             specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    #     Return:
    #         [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
    #         or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

    #             If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
    #             [`~utils.ModelOutput`] types are:

    #                 - [`~generation.GenerateDecoderOnlyOutput`],
    #                 - [`~generation.GenerateBeamDecoderOnlyOutput`]

    #             If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
    #             [`~utils.ModelOutput`] types are:

    #                 - [`~generation.GenerateEncoderDecoderOutput`],
    #                 - [`~generation.GenerateBeamEncoderDecoderOutput`]
    #     """

    #     # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    #     self._validate_model_class()
    #     tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
    #     assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

    #     generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    #     self._validate_model_kwargs(model_kwargs.copy())
    #     self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

    #     # 2. Set generation parameters if not already defined
    #     if synced_gpus is None:
    #         synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

    #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    #     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    #     accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    #     requires_attention_mask = "encoder_outputs" not in model_kwargs
    #     kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    #     # 3. Define model inputs
    #     inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
    #         inputs, generation_config.bos_token_id, model_kwargs
    #     )
    #     batch_size = inputs_tensor.shape[0]

    #     device = inputs_tensor.device
    #     self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

    #     # decoder-only models must use left-padding for batched generation.
    #     if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
    #         # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
    #         # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
    #         if (
    #             generation_config._pad_token_tensor is not None
    #             and batch_size > 1
    #             and len(inputs_tensor.shape) == 2
    #             and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
    #         ):
    #             logger.warning(
    #                 "A decoder-only architecture is being used, but right-padding was detected! For correct "
    #                 "generation results, please set `padding_side='left'` when initializing the tokenizer."
    #             )

    #     # 4. Define other model kwargs
    #     # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    #     # generating the first new token or not, and we only want to use the embeddings for the first new token)
    #     if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
    #         generation_config.use_cache = True

    #     if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
    #         model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
    #             inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
    #         )
    #     elif kwargs_has_attention_mask:
    #         # TODO (joao): generalize this check with other types of inputs
    #         if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
    #             raise ValueError("`attention_mask` passed to `generate` must be 2D.")

    #     if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
    #         # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
    #         model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
    #             inputs_tensor, model_kwargs, model_input_name, generation_config
    #         )

    #     # 5. Prepare `input_ids` which will be used for auto-regressive generation
    #     if self.config.is_encoder_decoder:
    #         input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
    #             batch_size=batch_size,
    #             model_input_name=model_input_name,
    #             model_kwargs=model_kwargs,
    #             decoder_start_token_id=generation_config._decoder_start_token_tensor,
    #             device=inputs_tensor.device,
    #         )
    #     else:
    #         input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    #     if generation_config.token_healing:
    #         input_ids = self.heal_tokens(input_ids, tokenizer)

    #     if streamer is not None:
    #         streamer.put(input_ids.cpu())

    #     # 6. Prepare `max_length` depending on other stopping criteria.
    #     input_ids_length = input_ids.shape[-1]
    #     has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    #     has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    #     generation_config = self._prepare_generated_length(
    #         generation_config=generation_config,
    #         has_default_max_length=has_default_max_length,
    #         has_default_min_length=has_default_min_length,
    #         model_input_name=model_input_name,
    #         inputs_tensor=inputs_tensor,
    #         input_ids_length=input_ids_length,
    #     )

    #     # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
    #     # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
    #     # dynamically overrides this value as it can need more than the last token logits
    #     if self._supports_num_logits_to_keep() and "num_logits_to_keep" not in model_kwargs:
    #         model_kwargs["num_logits_to_keep"] = 1

    #     self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    #     # 7. Prepare the cache.
    #     # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
    #     # - different models have a different cache name expected by the model (default = "past_key_values")
    #     # - `max_length`, prepared above, is used to determine the maximum cache length
    #     # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
    #     cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
    #     user_defined_cache = model_kwargs.get(cache_name)
    #     max_cache_length = generation_config.max_length
    #     if (
    #         inputs_tensor.shape[1] != input_ids_length
    #         and model_input_name == "inputs_embeds"
    #         and not self.config.is_encoder_decoder
    #     ):
    #         max_cache_length += inputs_tensor.shape[1]
    #     self._prepare_cache_for_generation(
    #         generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
    #     )

    #     # 8. determine generation mode
    #     generation_mode = generation_config.get_generation_mode(assistant_model)

    #     if streamer is not None and (generation_config.num_beams > 1):
    #         raise ValueError(
    #             "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
    #         )

    #     if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
    #         warnings.warn(
    #             "You are calling .generate() with the `input_ids` being on a device type different"
    #             f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
    #             f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
    #             " Please make sure that you have put `input_ids` to the"
    #             f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
    #             " running `.generate()`.",
    #             UserWarning,
    #         )

    #     # 9. prepare logits processors and stopping criteria
    #     prepared_logits_processor = self._get_logits_processor(
    #         generation_config=generation_config,
    #         input_ids_seq_length=input_ids_length,
    #         encoder_input_ids=inputs_tensor,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #         logits_processor=logits_processor,
    #         device=inputs_tensor.device,
    #         model_kwargs=model_kwargs,
    #         negative_prompt_ids=negative_prompt_ids,
    #         negative_prompt_attention_mask=negative_prompt_attention_mask,
    #     )
    #     prepared_stopping_criteria = self._get_stopping_criteria(
    #         generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
    #     )

    #     # Set model_kwargs `use_cache` so we can use it later in forward runs
    #     model_kwargs["use_cache"] = generation_config.use_cache

    #     # 10. go into different generation modes
    #     if generation_mode == GenerationMode.ASSISTED_GENERATION:
    #         if generation_config.num_return_sequences > 1:
    #             raise ValueError(
    #                 "num_return_sequences has to be 1 when doing assisted generate, "
    #                 f"but is {generation_config.num_return_sequences}."
    #             )
    #         if batch_size > 1:
    #             raise ValueError("assisted generate is only supported for batch_size = 1")
    #         if not model_kwargs["use_cache"]:
    #             raise ValueError("assisted generate requires `use_cache=True`")
    #         if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
    #             raise ValueError("assisted generate is not supported with Static cache classes`")
    #         if self._is_stateful:
    #             # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
    #             # which is not possible with stateful models (they can't reset to a previous subset of generated text)
    #             raise ValueError(
    #                 f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
    #             )

    #         # 11. Get the candidate generator, given the parameterization
    #         candidate_generator = self._get_candidate_generator(
    #             generation_config=generation_config,
    #             input_ids=input_ids,
    #             inputs_tensor=inputs_tensor,
    #             assistant_model=assistant_model,
    #             logits_processor=logits_processor,
    #             target_tokenizer=tokenizer,
    #             assistant_tokenizer=assistant_tokenizer,
    #             model_kwargs=model_kwargs,
    #         )

    #         # 12. run assisted generate
    #         result = self._assisted_decoding(
    #             input_ids,
    #             candidate_generator=candidate_generator,
    #             logits_processor=prepared_logits_processor,
    #             stopping_criteria=prepared_stopping_criteria,
    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             streamer=streamer,
    #             **model_kwargs,
    #         )
    #     elif generation_mode == GenerationMode.DOLA_GENERATION:
    #         if self._is_stateful:
    #             # DoLa decoding was not designed for stateful models, and would require some changes
    #             raise ValueError(
    #                 f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
    #             )
    #         result = self._dola_decoding(
    #             input_ids,
    #             dola_layers=generation_config.dola_layers,
    #             logits_processor=prepared_logits_processor,
    #             stopping_criteria=prepared_stopping_criteria,
    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             streamer=streamer,
    #             **model_kwargs,
    #         )

    #     elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
    #         if not model_kwargs["use_cache"]:
    #             raise ValueError("Contrastive search requires `use_cache=True`")
    #         if self._is_stateful:
    #             # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
    #             raise ValueError(
    #                 f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
    #             )

    #         result = self._contrastive_search(
    #             input_ids,
    #             logits_processor=prepared_logits_processor,
    #             stopping_criteria=prepared_stopping_criteria,
    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             streamer=streamer,
    #             **model_kwargs,
    #         )

    #     elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
    #         # 11. expand input_ids with `num_return_sequences` additional sequences per batch
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids=input_ids,
    #             expand_size=generation_config.num_return_sequences,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )

    #         # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
    #         result = self._sample(
    #             input_ids,
    #             logits_processor=prepared_logits_processor,
    #             stopping_criteria=prepared_stopping_criteria,
    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             streamer=streamer,
    #             **model_kwargs,
    #         )

    #     elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
    #         # 11. prepare beam search scorer
    #         beam_scorer = BeamSearchScorer(
    #             batch_size=batch_size,
    #             num_beams=generation_config.num_beams,
    #             device=inputs_tensor.device,
    #             length_penalty=generation_config.length_penalty,
    #             do_early_stopping=generation_config.early_stopping,
    #             num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #             max_length=generation_config.max_length,
    #         )

    #         # 12. interleave input_ids with `num_beams` additional sequences per batch
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids=input_ids,
    #             expand_size=generation_config.num_beams,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )

    #         # 13. run beam sample
    #         result = self._beam_search(
    #             input_ids,
    #             beam_scorer,
    #             logits_processor=prepared_logits_processor,
    #             stopping_criteria=prepared_stopping_criteria,
    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )

    #     elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
    #         # 11. prepare beam search scorer
    #         beam_scorer = BeamSearchScorer(
    #             batch_size=batch_size,
    #             num_beams=generation_config.num_beams,
    #             device=inputs_tensor.device,
    #             length_penalty=generation_config.length_penalty,
    #             do_early_stopping=generation_config.early_stopping,
    #             num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #             num_beam_groups=generation_config.num_beam_groups,
    #             max_length=generation_config.max_length,
    #         )
    #         # 12. interleave input_ids with `num_beams` additional sequences per batch
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids=input_ids,
    #             expand_size=generation_config.num_beams,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )
    #         # 13. run beam search
    #         result = self._group_beam_search(
    #             input_ids,
    #             beam_scorer,
    #             logits_processor=prepared_logits_processor,
    #             stopping_criteria=prepared_stopping_criteria,
    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )

    #     elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
    #         final_constraints = []
    #         if generation_config.constraints is not None:
    #             final_constraints = generation_config.constraints

    #         if generation_config.force_words_ids is not None:

    #             def typeerror():
    #                 raise ValueError(
    #                     "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
    #                     f"of positive integers, but is {generation_config.force_words_ids}."
    #                 )

    #             if (
    #                 not isinstance(generation_config.force_words_ids, list)
    #                 or len(generation_config.force_words_ids) == 0
    #             ):
    #                 typeerror()

    #             for word_ids in generation_config.force_words_ids:
    #                 if isinstance(word_ids[0], list):
    #                     if not isinstance(word_ids, list) or len(word_ids) == 0:
    #                         typeerror()
    #                     if any(not isinstance(token_ids, list) for token_ids in word_ids):
    #                         typeerror()
    #                     if any(
    #                         any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
    #                         for token_ids in word_ids
    #                     ):
    #                         typeerror()

    #                     constraint = DisjunctiveConstraint(word_ids)
    #                 else:
    #                     if not isinstance(word_ids, list) or len(word_ids) == 0:
    #                         typeerror()
    #                     if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
    #                         typeerror()

    #                     constraint = PhrasalConstraint(word_ids)
    #                 final_constraints.append(constraint)

    #         # 11. prepare beam search scorer
    #         constrained_beam_scorer = ConstrainedBeamSearchScorer(
    #             constraints=final_constraints,
    #             batch_size=batch_size,
    #             num_beams=generation_config.num_beams,
    #             device=inputs_tensor.device,
    #             length_penalty=generation_config.length_penalty,
    #             do_early_stopping=generation_config.early_stopping,
    #             num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #             max_length=generation_config.max_length,
    #         )
    #         # 12. interleave input_ids with `num_beams` additional sequences per batch
    #         input_ids, model_kwargs = self._expand_inputs_for_generation(
    #             input_ids=input_ids,
    #             expand_size=generation_config.num_beams,
    #             is_encoder_decoder=self.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )
    #         # 13. run beam search
    #         result = self._constrained_beam_search(
    #             input_ids,
    #             constrained_beam_scorer=constrained_beam_scorer,
    #             logits_processor=prepared_logits_processor,
    #             stopping_criteria=prepared_stopping_criteria,
    #             generation_config=generation_config,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )

    #     # Convert to legacy cache format if requested
    #     if (
    #         generation_config.return_legacy_cache is not False  # Should check for `True` after v4.47
    #         and not is_torchdynamo_compiling()
    #         and hasattr(result, "past_key_values")
    #         and hasattr(result.past_key_values, "to_legacy_cache")
    #         and result.past_key_values.to_legacy_cache is not None
    #     ):
    #         # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
    #         should_convert_cache = generation_config.return_legacy_cache
    #         is_user_defined_cache = user_defined_cache is not None
    #         is_default_cache_type = (
    #             type(result.past_key_values) == DynamicCache  # noqa E721
    #             or (
    #                 isinstance(result.past_key_values, EncoderDecoderCache)
    #                 and type(result.past_key_values.self_attention_cache) == DynamicCache  # noqa E721
    #                 and type(result.past_key_values.cross_attention_cache) == DynamicCache  # noqa E721
    #             )
    #         )
    #         if not is_user_defined_cache and is_default_cache_type:
    #             logger.warning_once(
    #                 "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
    #                 "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
    #                 "keep returning the legacy format, please set `return_legacy_cache=True`."
    #             )
    #             should_convert_cache = True
    #         if should_convert_cache:
    #             result.past_key_values = result.past_key_values.to_legacy_cache()
    #     return result