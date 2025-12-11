from dataclasses import dataclass
from time import sleep, time
import os
import re
import sys
import datetime
from typing import Callable, Optional
import pandas as pd
from cryptol import Qed, Counterexample

result_file = ""

MODEL_ALIASES = {
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-4B-Instruct": "Qwen3-4B-Instruct",
    "Qwen/Qwen2.5-Coder-7B": "Qwen2_5-Coder-7B",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "Qwen2_5-Coder-3B-Instruct",
}

@dataclass
class Config:
    SERVER_URL: str = "http://localhost:8080"          # Cryptol remote API
    MODEL_ID: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    TEMP_FILE: str = "/Users/josh/SecurityAnalytics/DataPreprocess/cryptol-files/generated.cry"       
    CRYPTOL_PATH: str = "/home/cryptol/files/generated.cry"  # path inside Cryptol server container
    EVALS_PATH: str = "/Users/josh/SecurityAnalytics/DataPreprocess/src/eval/.data/evals.jsonl"
    RESULTS_FILE_PATH: str | None = None
    SYSTEM_PROMPT: str = "Return exactly ONE fenced code block labeled `cryptol` and nothing else (no prose before/after)."

    FUNCTION_PROMPT_TEMPLATE: str = """\
### Instruction:
Write a Cryptol function that implements the tasks described below.

### Request:
Task: {task}
"""

    PROPERTY_PROMPT_TEMPLATE: str = """\
### Instruction:
Write a Cryptol property that tests the function described below.

### Request:
Task: {task}
"""

def get_cryptol_snippet_for_task(task_id: int) -> str:
    """
    Given the full eval log text and a task_id, return the Cryptol code snippet
    (inside ```cryptol fences) for that task.
    """
    global result_file
    # 1) Extract the block for this specific task
    task_pattern = re.compile(
        rf"^=== Task {task_id}\s*===\s*(.*?)(?=^=== Task \d+ ===|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    m_task = task_pattern.search(result_file)
    if not m_task:
        raise ValueError(f"Task {task_id} block not found")

    task_block = m_task.group(1)

    # 2) Extract the ```cryptol ... ``` snippet from that block
    cryptol_pattern = re.compile(
        r"```cryptol\s*(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    m_code = cryptol_pattern.search(task_block)
    if not m_code:
        raise ValueError(f"No ```cryptol``` block found for task {task_id}")

    return m_code.group(0).strip() 
   

# ------------------ Helpers -------------------------
def extract_code_block(text: str) -> str:
    """
    Get content inside ```cryptol ...``` or generic ```...```.
    If no fence is found, return the raw text.
    """
    m = re.search(r"```(?:cryptol)?\s*(.*?)```", text, flags=re.S | re.I)
    return (m.group(1) if m else text).strip()

def execute_test_code(source_code: str, tests: list[str], type: str = "function") -> tuple[bool, str]:
        import cryptol
        from cryptol import BV
        result_ = f"[GENERATE BEGIN]\n```cryptol\n{source_code}\n```\n[GENERATE END]\n\n"
        try:
            with open(config.TEMP_FILE, "w") as f:
                f.write(source_code)
                f.flush()
                os.fsync(f.fileno())
            #sleep(1)  # Give time for file to be written
            print(f"[INFO] Wrote generated Cryptol to {config.TEMP_FILE} (overwritten).")
        except Exception as e:
            result_ += f"[ERROR] Writing temp file failed"
            print(result_)
            return False, f"{result_}\n"
            
    # -------- Cryptol: load & test --------
        try:
            cry = cryptol.connect(url=config.SERVER_URL, reset_server=True)
            for attempt in range(3):
                try:
                    cry.load_file(config.CRYPTOL_PATH).result()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise  # re-raise after final attempt
                time.sleep(0.25)
        except Exception as e:
            result_ += f"[ERROR] Cryptol load failed"
            print(result_)
            # Try to close/reset and move on
            try:
                cry.reset_server()
            except Exception:
                pass
            return False, f"{result_}\n"

        # Namespace exposed to exec() tests (only what's needed)
        ns = {"cry": cry, "BV": BV}                
        # Run tests
        all_ok = True
        for i, test_src in enumerate(tests, 1):
            ok = False
            msg = ""
            try:
                if type == "function":
                    print(f"cryptol: {test_src['cryptol']}")
                    print(f"expected: {test_src['expected']}")
                    res = cry.eval_f(str(test_src["cryptol"])).result()
                    expected_res = cry.eval_f(str(test_src["expected"])).result()
                    assert res == expected_res
                    ok = True
                    msg = "OK"
                elif type == "property":
                    proof_future = cry.prove_f(test_src["cryptol"])  # equivalent is a Cryptol property
                    proof = proof_future.result()
                    if isinstance(proof, Qed):
                        ok = True
                        msg = "OK"
            except AssertionError as e:
                ok = False
                msg = f"AssertionError"
                
            except Exception as e:
                ok = False
                msg = f"General Error"

            all_ok &= ok
            status = "PASS" if ok else "FAIL"
            result_ += f"  [{status}] test {i}: {msg}\n"
        
        # Cleanup/reset between tasks
        try:
            cry.reset_server()
        except Exception:
            pass
        return all_ok, result_

def run_eval_suite(
        eval_df: pd.DataFrame, 
        config: Config, 
        execute: bool,
        generate_fn: Optional[Callable[[str], str]] = None, 
        provider: str = "nebius",
        local: bool = False,
    ) -> None:
    """
    Run the eval suite given by eval_df.
    Each row should have 'task', optional 'test_setup_code', and 'test_list'.
    """
    score = None
    attempts = 0
    successful_attempts = 0
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f"Starting eval suite at {start_time}, {len(eval_df)} tasks to process.")
    # ----------------- Inference Client -----------------
    client = None
    if generate_fn is None:
        from huggingface_hub import InferenceClient
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            print("ERROR: Set HF_TOKEN in your environment.", file=sys.stderr)
            sys.exit(1)
        client = InferenceClient(
            provider=provider,
            api_key=HF_TOKEN,
        )
    results = ""
    # -------------------- Main loop --------------------- #
    for idx, row in eval_df.iterrows():
        task           = row["task"]
        task_id        = row.get("task_id", f"row{idx}")
        tests          = row.get("test_list", []) or []
        setup_code     = row.get("test_setup_code", "") or ""
        type           = row.get("type", "function")

        # -------- Prepare messages (system + user) --------
        if type == "property":
            user_content = config.PROPERTY_PROMPT_TEMPLATE.format(task=task)
        else:
            user_content = config.FUNCTION_PROMPT_TEMPLATE.format(task=task)

        if setup_code != "":
            user_content += (
                f"\n### Additional setup code:\n```cryptol\n{setup_code}\n```"
            )

        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        # For logging, show both system + user parts
        prompt_log = (
            f"[SYSTEM]\n{config.SYSTEM_PROMPT}\n\n"
            f"[USER]\n{user_content}"
        )

        result_ = f"\n=== Task {task_id} ===\n"
        result_ += f"\n[PROMPT BEGIN]\n{prompt_log}\n[PROMPT END]\n\n"

        # -------- Inference --------
        try:
            if generate_fn is not None:
                content = generate_fn(
                    messages if not local else task_id
                )
            else:
                completion = client.chat.completions.create(
                    model=config.MODEL_ID,
                    messages=messages,          # <-- now using system + user
                    max_tokens=1024,
                    temperature=0.2,
                )
                # HF chat client returns .message.content
                content = completion.choices[0].message.content
        except Exception as e:
            result_ += f"[ERROR] Inference failed: {e}"
            print(result_)
            results += f"{result_}\n"
            continue

        # -------- Extract code & save to single temp file --------
        source_code = extract_code_block(content)
        if setup_code != "":
            source_code = f"{setup_code}\n\n{source_code}"

        if execute:
            attempts += 1
            all_ok, result_ = execute_test_code(source_code, tests, type=type)
            if all_ok:
                successful_attempts += 1
            result_ += f"[RESULT] Task {task_id}: {'ALL PASS' if all_ok else 'HAS FAILURES'}\n"
            print(result_)
            results += f"{result_}\n"
        else:
            result_ += f"[GENERATED BEGIN]\n```cryptol\n{source_code}\n```\n[GENERATED END]\n"
            print(result_)
            results += f"{result_}\n"
    end_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    if attempts > 0:
        score = successful_attempts / attempts * 100.0
        final_message = f"\n=== FINAL SCORE: {successful_attempts} / {attempts} = {score:.2f}% ===\n"
        print(f"\n=== FINAL SCORE: {successful_attempts} / {attempts} = {score:.2f}% ===\n")
    else:
        final_message = "\n=== FINAL SCORE: N/A (no executed tasks) ===\n"
        print("\n=== FINAL SCORE: N/A (no executed tasks) ===\n")
    results = f"{config.MODEL_ID} Eval Suite Results\nStarted at {start_time} Ended at {end_time}\nProcessed {len(eval_df)} tasks.\n{final_message}\n" + results
    print(f"\n{end_time}\nDone processing all evals.")
    if config.RESULTS_FILE_PATH is not None:
        with open(config.RESULTS_FILE_PATH, "w") as f:
            f.write(results)
        print(f"Wrote eval results to {config.RESULTS_FILE_PATH}.")

if __name__ == "__main__":
    '''
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model = "Qwen/Qwen2.5-Coder-7B"
    config = Config(
        MODEL_ID=model,
        EVALS_PATH="/Users/josh/SecurityAnalytics/Colab-Training/data/evals.jsonl",
        RESULTS_FILE_PATH=f"/Users/josh/SecurityAnalytics/Colab-Training/results/{MODEL_ALIASES[model]}_eval_{start_time}.txt",
    )
    '''
    file_name = sys.argv[1] if len(sys.argv) > 1 else ""
    config = Config(
        MODEL_ID="Qwen/Qwen2.5-Coder-7B",
        EVALS_PATH="data/evals.jsonl",
        RESULTS_FILE_PATH=f"{''.join(file_name.split('.')[:-1])}_results.txt" if file_name else None,
    )
    with open(file_name, "r") as f:
        result_file = f.read()
    eval_df = pd.read_json(config.EVALS_PATH, lines=True)
    run_eval_suite(
        eval_df,
        config,
        True,
        generate_fn=get_cryptol_snippet_for_task,
        local=True,
    )