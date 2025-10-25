# model/dataset/util.py
import functools
import numpy as np

FIM_PREFIX_TOK = "<|fim_prefix|>"
FIM_MIDDLE_TOK = "<|fim_middle|>"
FIM_SUFFIX_TOK = "<|fim_suffix|>"
FIM_PAD_TOK    = "<|fim_pad|>"

@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    # 1) Try to resolve ids by name
    ids = {
        "prefix": tokenizer.convert_tokens_to_ids(FIM_PREFIX_TOK),
        "middle": tokenizer.convert_tokens_to_ids(FIM_MIDDLE_TOK),
        "suffix": tokenizer.convert_tokens_to_ids(FIM_SUFFIX_TOK),
        "pad":    tokenizer.convert_tokens_to_ids(FIM_PAD_TOK),
    }

    # 2) If any are missing, attempt to register them as *special* tokens (harmless if they already exist)
    missing = [tok for tok, tid in zip(
        [FIM_PREFIX_TOK, FIM_MIDDLE_TOK, FIM_SUFFIX_TOK, FIM_PAD_TOK],
        [ids["prefix"],  ids["middle"],  ids["suffix"],  ids["pad"]]
    ) if tid is None or tid == tokenizer.unk_token_id]

    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        # refresh ids after add
        ids = {
            "prefix": tokenizer.convert_tokens_to_ids(FIM_PREFIX_TOK),
            "middle": tokenizer.convert_tokens_to_ids(FIM_MIDDLE_TOK),
            "suffix": tokenizer.convert_tokens_to_ids(FIM_SUFFIX_TOK),
            "pad":    tokenizer.convert_tokens_to_ids(FIM_PAD_TOK),
        }

    return ids["suffix"], ids["prefix"], ids["middle"], ids["pad"]


## Adapted from https://github.com/bigcode-project/Megatron-LM/blob/6c4bf908df8fd86b4977f54bf5b8bd4b521003d1/megatron/data/gpt_dataset.py
def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """

    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:

            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng
