"""LogicVista evaluation with vLLM and avg@N scoring."""

import argparse
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional

import pandas as pd
from datasets import load_dataset, Dataset
from Levenshtein import distance
from rich.logging import RichHandler
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def should_use_qwen_multimodal(model_type: str, checkpoint: str, processor: Optional[AutoProcessor] = None) -> bool:
    model_type = (model_type or "").strip().lower()
    if model_type == "qwen":
        return True
    if model_type == "none":
        return False
    if model_type not in {"", "auto"}:
        return "qwen" in model_type
    if checkpoint and "qwen" in checkpoint.lower():
        return True
    if processor is not None and "qwen" in processor.__class__.__name__.lower():
        return True
    return False


def prepare_qwen_image(messages):
    from qwen_vl_utils import process_vision_info
    image_obj, _ = process_vision_info(messages)
    return image_obj

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

def get_most_similar(prediction, choices):
    dists = [distance(str(prediction), str(c)) for c in choices]
    if not dists:
        return prediction
    return choices[int(dists.index(min(dists)))]

ANSWER_TAG_PATTERN = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)

def extract_answer(text: str) -> str:
    """Extract the answer span and normalize a few trailing wrappers."""
    if not isinstance(text, str):
        return ""
    m = re.search(ANSWER_TAG_PATTERN, text)
    if m and m.group(1).strip():
        s = m.group(1).strip()
    else:
        m2 = re.search(r"<\s*answer\s*>", text, flags=re.IGNORECASE)
        s = text[m2.end():].strip() if m2 else text.strip()
    box = re.search(r"\\boxed\s*\{(.*)\}\s*$", s, flags=re.DOTALL)
    if box:
        s = box.group(1).strip()
    if s and s[-1] in ('.', '。'):
        s = s[:-1].strip()
    s = s.replace("π", "\\pi")
    return s

def normalize_answer(ans: str) -> str:
    return (ans or "").lower().strip()

def qa_accuracy(pred: str, gt: str) -> int:
    """Return 1 for an exact single-letter match, otherwise 0."""
    return 1 if normalize_answer(pred) == normalize_answer(gt) else 0

_LETTER_RE = re.compile(r"\b([A-Za-z])\b")

def normalize_prediction_to_letter(resp_extracted: str) -> Optional[str]:
    """Normalize the extracted response into a single option letter."""
    if not isinstance(resp_extracted, str):
        return None
    s = resp_extracted.strip()
    m = re.findall(_LETTER_RE, s)
    if m:
        return m[-1].upper()
    m2 = re.search(r"[A-Za-z]", s)
    if m2:
        return m2.group(0).upper()
    return None

def get_acc_with_condition(res_pd, key, value):
    if key in res_pd.columns:
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x) if res_pd[key].dtype == "O" and res_pd[key].map(type).eq(list).any() else res_pd[key] == value]
    else:
        return 0.0, 0, 0.0
    if len(total_pd) == 0:
        return 0.0, 0, 0.0
    scores = total_pd['true_false'].astype(float)
    sum_score = float(scores.sum())
    avg_score = float(scores.mean())
    return sum_score, len(total_pd), avg_score

def compute_scores(results_by_pid: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for pid, item in results_by_pid.items():
        d = dict(item)
        flat[pid] = d
    results_df = pd.DataFrame(flat).T

    ids = list(results_by_pid.keys())
    total = len(ids)
    correct_sum = sum(float(results_by_pid[pid].get("true_false", 0.0)) for pid in ids)
    accuracy = (correct_sum / total) if total > 0 else 0.0

    target_keys = ['skill', 'broad_capability', 'specific_capability', 'imagesource']
    scores = {"average": {"accuracy": accuracy, "correct": correct_sum, "total": total}}
    for key in target_keys:
        if key not in results_df.columns:
            continue
        values = set()
        for v in results_df[key].dropna().tolist():
            if isinstance(v, list):
                values.update(v)
            else:
                values.add(v)
        bucket = {}
        for v in values:
            sum_score, t, acc = get_acc_with_condition(results_df, key, v)
            if t > 0:
                bucket[v] = {"accuracy": acc, "correct": sum_score, "total": t}
        scores[key] = dict(sorted(bucket.items(), key=lambda kv: float(kv[1]['accuracy']), reverse=True))
    return scores

INSTR_SUFFIX = "\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=None,
                        help="If not set: default to 1.0 when bon_n>1 (avg@N), else 0.0.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--bon_n", type=int, default=1, help="Number of candidates per input (avg@N).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--parallel-mode", type=str, default="dp", choices=["dp", "tp", "pp", "auto"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-backend", type=str, default="mp", choices=["mp", "ray"])
    parser.add_argument("--cache_dir", type=str, default=os.path.join(os.getcwd(), "data/LogicVista/"))
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument("--keep-multi-char-answers", action="store_true",
                        help="By default we DROP items whose answer is not a single letter (e.g., 'A, C'). Set this to keep them.")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--results_basename", type=str, default=None)
    parser.add_argument("--loglevel", type=str, default="INFO")
    parser.add_argument("--model_type", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.loglevel.upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False, show_path=False, omit_repeated_times=False)],
    )
    logger = logging.getLogger("LogicVistaEval-1JSON")

    if args.temperature is None:
        effective_temperature = 1.0 if args.bon_n > 1 else 0.0
    else:
        effective_temperature = float(args.temperature)

    logger.info(f"Loading dataset lscpku/LogicVista (split=test, cache_dir={args.cache_dir}) ...")
    data: Dataset = load_dataset("lscpku/LogicVista", split="test", cache_dir=args.cache_dir)

    total_before = len(data)

    def _is_single_letter(ans: Any) -> bool:
        if not isinstance(ans, str):
            return False
        return bool(re.fullmatch(r"[A-Za-z]", ans.strip()))

    if not args.keep_multi_char_answers:
        logger.info("Filtering out items whose answer is NOT a single letter (e.g., 'A, C') ...")
        data = data.filter(lambda x: _is_single_letter(x.get("answer")))
        logger.info(f"Answer-length filter: {total_before} -> {len(data)} items kept.")

    if args.max_num_problems > 0:
        data = data.select(range(min(args.max_num_problems, len(data))))
        logger.warning(f"Limiting number of problems to {len(data)}.")

    logger.info(f"Loading model: {args.checkpoint} | mode={args.parallel_mode} "
                f"tp={args.tensor_parallel_size} pp={args.pipeline_parallel_size} "
                f"| gpu_mem_util={args.gpu_memory_utilization}")
    tp = args.tensor_parallel_size
    pp = args.pipeline_parallel_size
    if args.parallel_mode == "dp":
        tp, pp = 1, 1
    elif args.parallel_mode == "tp":
        tp = max(tp, 2); pp = 1
    elif args.parallel_mode == "pp":
        pp = max(pp, 2); tp = 1
    else:
        if tp <= 1 and pp <= 1:
            tp, pp = 1, 1

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        distributed_executor_backend=args.distributed_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    use_qwen_multimodal = should_use_qwen_multimodal(args.model_type, args.checkpoint, processor)
    if use_qwen_multimodal:
        logger.info("Using Qwen-specific multimodal preprocessing.")

    inputs = []
    for item in tqdm(data, desc="prepare"):
        image_obj = item["image"]
        try:
            image_path_str = getattr(image_obj, "filename", None)
        except Exception:
            image_path_str = None

        user_text = (item.get("question") or "") + INSTR_SUFFIX
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_obj},
            {"type": "text", "text": user_text},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if use_qwen_multimodal:
            image_obj = prepare_qwen_image(messages)

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_obj},
            "_image_path_for_log": image_path_str,
        })

    sampling_params = SamplingParams(
        temperature=effective_temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=max(1, int(args.bon_n)),
    )
    logger.info(f"Generating with vLLM ... (avg@N n={args.bon_n}, temperature={effective_temperature}, top_p={args.top_p})")
    model_outputs = llm.generate(inputs, sampling_params=sampling_params)

    results_by_pid: Dict[str, Any] = {}
    for idx, (item, mo, packed) in enumerate(zip(data, model_outputs, inputs)):
        pid = str(item.get("id", idx))
        gt_letter = str(item.get("answer", "")).strip()

        responses: List[str] = [out.text for out in mo.outputs]
        extracted_list: List[str] = [extract_answer(r) for r in responses]
        pred_letters: List[Optional[str]] = [normalize_prediction_to_letter(x) for x in extracted_list]
        tf_list_int: List[int] = [qa_accuracy((p or ""), gt_letter) for p in pred_letters]
        n_cand = max(1, int(sampling_params.n))
        avg_correct = float(sum(tf_list_int)) / float(n_cand)

        resp = responses[0] if responses else ""
        extraction = extracted_list[0] if extracted_list else ""
        pred_letter_first = pred_letters[0] if pred_letters else None

        results_by_pid[pid] = {
            "pid": pid,
            "question": item.get("question"),
            "image_path": packed.get("_image_path_for_log", None),
            "answer": gt_letter,
            "reasoning": item.get("reasoning"),
            "skill": item.get("skill"),
            "broad_capability": item.get("broad_capability"),
            "specific_capability": item.get("specific_capability"),
            "imagesource": item.get("imagesource"),
            "sourcelink": item.get("sourcelink"),
            "liscenced": item.get("liscenced"),

            "responses": responses,
            "extractions_list": extracted_list,
            "pred_letters": pred_letters,
            "true_false_list": [bool(x) for x in tf_list_int],

            "response": resp,
            "extraction": extraction,
            "prediction": pred_letter_first,

            "true_false": avg_correct,
        }

    scores = compute_scores(results_by_pid)

    ts = time.strftime("%y%m%d%H%M%S", time.localtime())
    base = args.results_basename or f"LogicVista_{ts}"
    merged = {
        "config": {
            "checkpoint": args.checkpoint,
            "root": "lscpku/LogicVista",
            "split": "test",
            "cache_dir": args.cache_dir,
            "max_tokens": args.max_tokens,
            "temperature": effective_temperature,
            "top_p": args.top_p,
            "bon_n": int(args.bon_n),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "parallel_mode": args.parallel_mode,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "distributed_backend": args.distributed_backend,
            "num_samples": len(results_by_pid),
            "scoring": "avg@N (letter equality)",
            "dropped_multi_char_answers": not args.keep_multi_char_answers,
        },
        "scores": scores,
        "results": results_by_pid
    }
    out_path = os.path.join(args.out_dir, f"{base}_merged.json")
    save_json(merged, out_path)
    logging.getLogger("LogicVistaEval-1JSON").info(f"Saved ONE merged JSON to: {out_path}")

if __name__ == "__main__":
    main()
