"""MathVista evaluation with vLLM and avg@N scoring."""

from mathruler.grader import grade_answer
import argparse
import json
import logging
import os
import re
import time
from typing import Dict, Any, List, Optional
import pandas as pd
from datasets import load_dataset
from Levenshtein import distance
from rich.logging import RichHandler
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ds_collections = {
    "MathVista_testmini": {"root": "AI4Math/MathVista", "split": "testmini"},
    "MathVista_test": {"root": "AI4Math/MathVista", "split": "test"},
}


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

def strip_tex_wrappers(s: Optional[str]) -> Optional[str]:
    if s is None:
        return s
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = re.sub(r"\\boxed\s*\{(.*)\}\s*$", r"\1", s, flags=re.DOTALL)
    return s.strip()

def extract_answer_from_response(resp_text: str) -> str:
    if not isinstance(resp_text, str):
        return ""
    m = re.search(ANSWER_TAG_PATTERN, resp_text)
    if m and m.group(1).strip():
        return strip_tex_wrappers(m.group(1))
    m2 = re.search(r"<\s*answer\s*>", resp_text, flags=re.IGNORECASE)
    if m2:
        seg = resp_text[m2.end():].strip()
        return strip_tex_wrappers(seg)
    return strip_tex_wrappers(resp_text)

def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False):
    if question_type == 'multi_choice':
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""
        if ignore_empty_extractions and not extraction:
            return None
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()
        seq_chars = [chr(ord('A') + i) for i in range(len(choices))]
        if extraction in seq_chars:
            idx = seq_chars.index(extraction)
            norm = choices[idx]
        else:
            norm = get_most_similar(extraction, choices)
        if isinstance(choices, list) and choices:
            assert norm in choices, f"Normalized '{norm}' not in choices."
        return norm

    if answer_type == 'integer':
        try:
            return str(int(float(extraction)))
        except Exception:
            return None

    if answer_type == 'float':
        try:
            p = int(precision) if precision is not None else 0
            return str(round(float(extraction), p))
        except Exception:
            return None

    if answer_type == 'list':
        try:
            return str(extraction)
        except Exception:
            return None

    try:
        return str(extraction).strip()
    except Exception:
        return None

def safe_equal(prediction, answer):
    try:
        return grade_answer(answer, prediction)
    except Exception:
        return False

def get_acc_with_condition(res_pd, key, value):
    """
    Returns (sum_score, total_items, average_score) where 'score' is the per-item
    average correctness in [0,1]. This generalizes BoN boolean to avg@N float.
    """
    if key == 'skills':
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]
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
        if isinstance(d.get("metadata"), dict):
            d.update(d.pop("metadata"))
        flat[pid] = d
    results_df = pd.DataFrame(flat).T

    test_pids = list(results_by_pid.keys())
    total = len(test_pids)
    correct_sum = sum(float(results_by_pid[pid].get("true_false", 0.0)) for pid in test_pids)
    accuracy = (correct_sum / total) if total > 0 else 0.0

    target_keys = ['question_type', 'answer_type', 'language', 'source',
                   'category', 'task', 'context', 'grade', 'skills']

    scores = {"average": {"accuracy": accuracy, "correct": correct_sum, "total": total}}
    for key in target_keys:
        if key == 'skills':
            vals = set()
            if key in results_df.columns:
                for skills in results_df[key].dropna().tolist():
                    if isinstance(skills, list):
                        vals.update(skills)
            values = list(vals)
        else:
            values = results_df[key].dropna().unique().tolist() if key in results_df.columns else []

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
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None,
                        help="If not set: default to 1.0 when bon_n>1 (avg@N), else 0.0.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--bon_n", type=int, default=1, help="Number of candidates per input (avg@N).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--parallel-mode", type=str, default="dp", choices=["dp", "tp", "pp", "auto"],
                        help="dp=replicate model outside this process (no in-proc sharding), tp=tensor parallel, pp=pipeline parallel.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-backend", type=str, default="mp", choices=["mp", "ray"])
    parser.add_argument("--ds_name", type=str, default="MathVista_testmini", choices=list(ds_collections.keys()))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(os.getcwd(), "data/MathVista/"))
    parser.add_argument("--max_num_problems", type=int, default=-1)
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
    logger = logging.getLogger("MathVistaEval-1JSON")

    if args.temperature is None:
        effective_temperature = 1.0 if args.bon_n > 1 else 0.0
    else:
        effective_temperature = float(args.temperature)

    root = ds_collections[args.ds_name]["root"]
    split = ds_collections[args.ds_name]["split"]

    tp = args.tensor_parallel_size
    pp = args.pipeline_parallel_size
    if args.parallel_mode == "dp":
        tp, pp = 1, 1
    elif args.parallel_mode == "tp":
        tp = max(tp, 2)
        pp = 1
    elif args.parallel_mode == "pp":
        pp = max(pp, 2)
        tp = 1
    else:
        if tp <= 1 and pp <= 1:
            tp, pp = 1, 1

    logger.info(f"Loading dataset {args.ds_name} => {root}:{split} (cache_dir={args.cache_dir}) ...")
    data = load_dataset(root, split=split, cache_dir=args.cache_dir)
    if args.max_num_problems > 0:
        data = data.select(range(min(args.max_num_problems, len(data))))
        logger.warning(f"Limiting number of problems to {len(data)}.")

    logger.info(f"Loading model: {args.checkpoint} | mode={args.parallel_mode} tp={tp} pp={pp} "
                f"| gpu_mem_util={args.gpu_memory_utilization}")
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
        image_obj = item["decoded_image"]
        user_text = (item["query"] or "") + INSTR_SUFFIX
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
    for item, mo in zip(data, model_outputs):
        pid = str(item["pid"])
        ans = item["answer"]
        choices = item["choices"]
        qtype = item["question_type"]
        atype = item["answer_type"]
        prec = item["precision"]
        meta = item["metadata"]

        responses: List[str] = [out.text for out in mo.outputs]
        extractions_list: List[str] = [extract_answer_from_response(r) for r in responses]
        predictions_list: List[Optional[str]] = [
            normalize_extracted_answer(
                e,
                choices=choices or [],
                question_type=qtype,
                answer_type=atype,
                precision=prec,
                ignore_empty_extractions=False,
            )
            for e in extractions_list
        ]
        true_false_list: List[bool] = [safe_equal(p, str(ans).strip()) if p is not None else False
                                       for p in predictions_list]

        n_cand = max(1, int(args.bon_n))
        avg_correct = float(sum(1.0 if tf else 0.0 for tf in true_false_list)) / float(n_cand)

        first_index = 0
        resp = responses[first_index] if responses else ""
        extraction = extractions_list[first_index] if extractions_list else ""
        norm_pred = predictions_list[first_index] if predictions_list else None

        results_by_pid[pid] = {
            "pid": pid,
            "query": item["query"],
            "image": item["image"],
            "choices": choices,
            "answer": ans,
            "question_type": qtype,
            "answer_type": atype,
            "precision": prec,
            "metadata": meta,
            "responses": responses,
            "extractions_list": extractions_list,
            "predictions_list": predictions_list,
            "true_false_list": true_false_list,
            "response": resp,
            "extraction": extraction,
            "prediction": norm_pred,
            "true_false": avg_correct,
        }

    scores = compute_scores(results_by_pid)

    ts = time.strftime("%y%m%d%H%M%S", time.localtime())
    base = args.results_basename or f"{args.ds_name}_{ts}"
    merged = {
        "config": {
            "checkpoint": args.checkpoint,
            "ds_name": args.ds_name,
            "root": root,
            "split": split,
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
            "scoring": "avg@N",
        },
        "scores": scores,
        "results": results_by_pid
    }
    out_path = os.path.join(args.out_dir, f"{base}_merged.json")
    save_json(merged, out_path)
    logging.getLogger("MathVistaEval-1JSON").info(f"Saved ONE merged JSON to: {out_path}")

if __name__ == "__main__":
    main()
