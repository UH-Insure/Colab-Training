from copy import deepcopy
from datasets import Dataset, load_dataset
from typing import List, Dict, Any

SKIPPED_FILES = set()

def _chunk_messages_by_tokens(messages, tokenizer, max_len: int, source_name: str | None = None):
    """
    Split a single conversation (list of {role, content}) into a list of
    shorter conversations, each of which tokenizes to <= max_len tokens.

    If a single message is itself > max_len tokens, we skip it and log
    its token length and (optionally) the source_name (e.g., filename).
    """
    chunks = []
    current = []

    def tokens_for(msgs):
        ids = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,   # we want the true length
        )
        return len(ids)

    def log_skip(n_tokens: int):
        msg = f"Skipping over-long single message with {n_tokens} tokens"
        if source_name is not None:
            msg += f" from {source_name}"
            SKIPPED_FILES.add(source_name)
        print(msg)

    for msg in messages:
        tentative = current + [msg]
        length = tokens_for(tentative)

        if length <= max_len:
            # Safe to add this message to the current chunk
            current = tentative
        else:
            # Current chunk is full -> flush it if non-empty
            if current:
                chunks.append(current)
                single_len = tokens_for([msg])
                if single_len <= max_len:
                    current = [msg]
                else:
                    log_skip(single_len)
                    current = []
            else:
                # No current chunk and this message alone is too long -> skip
                single_len = tokens_for([msg])
                if single_len <= max_len:
                    current = [msg]
                else:
                    log_skip(single_len)
                    current = []

    if current:
        chunks.append(current)

    return chunks


def explode_long_conversations(raw_ds: Dataset,
                               tokenizer,
                               max_len: int) -> Dataset:
    """
    Take the original `raw` Dataset with a `messages` column and return
    a new Dataset where long conversations have been split into multiple
    rows of <= max_len tokens.

    All other columns (filetype, filename, etc.) are copied through.
    Two extra columns are added:
      - orig_conv_idx: the original row index in `raw`
      - chunk_idx: which chunk number (0, 1, 2, ...) from that conversation
    """
    new_rows = []

    for i in range(len(raw_ds)):
        row = raw_ds[i]
        msgs = row["messages"]

        # normalize messages if you have this helper
        msgs_norm = _normalize_messages(msgs)

        # use whatever your filename column is called; adjust if needed
        source_name = row.get("filename", f"row_{i}")

        chunks = _chunk_messages_by_tokens(
            msgs_norm,
            tokenizer,
            max_len,
            source_name=str(source_name),
        )

        for j, chunk in enumerate(chunks):
            new_row = dict(row)
            new_row["messages"] = chunk
            new_row["orig_conv_idx"] = i
            new_row["chunk_idx"] = j
            new_rows.append(new_row)

    print(f"Expanded {len(raw_ds)} original rows into {len(new_rows)} chunks")

    # Optional: summary of which files had over-long messages
    if SKIPPED_FILES:
        print("\nFiles with at least one skipped over-long message:")
        for name in sorted(SKIPPED_FILES):
            print("  ", name)

    return Dataset.from_list(new_rows)

def _normalize_messages(msgs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize content fields to plain strings; keep only role & content."""
    out = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            # flatten text parts if given like [{"type":"text","text":"..."}]
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    parts.append(p)
            content = "".join(parts)
        elif not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out

def load_or_make_dataset(jsonl_path: str):
    if jsonl_path:
        if jsonl_path.endswith(".jsonl"):
            ds = load_dataset("json", data_files=jsonl_path, split="train")
        else:
            # assume JSON array
            ds = load_dataset("json", data_files=jsonl_path, split="train")
        # basic validation
        assert "messages" in ds.column_names, "Dataset must have a 'messages' column."
        return ds
    else:
        # Toy dataset (few-shot) for a quick smoke test
        toy = [
            {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a haiku about the moon."},
                {"role": "assistant", "content": "Silent silver orb\nDrifting high in velvet night\nDreams glow in cool light."}
            ]},
            {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 17 * 12?"},
                {"role": "assistant", "content": "17 * 12 = 204."}
            ]},
            {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Give me a short Python function that returns the square of a number."},
                {"role": "assistant", "content": "def square(x):\n    return x * x"}
            ]},
        ]
        return Dataset.from_list(toy)

