"""MathVerse evaluation with vLLM and avg@N scoring."""

from mathruler.grader import grade_answer
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

ds_collections = {
    "MathVerse_testmini": {"root": "AI4Math/MathVerse", "name": "testmini"},
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

def extract_answer(text: str) -> str:
    """Prefer <answer>...</answer>; otherwise take text after <answer>; else use full text.
       Also unwrap \\boxed{...}; remove trailing '.'/'。'; replace π with \\pi."""
    if not isinstance(text, str):
        return ""
    m = re.search(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", text, flags=re.IGNORECASE | re.DOTALL)
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
    """Lowercase and trim."""
    return (ans or "").lower().strip()

def qa_accuracy(pred: str, gt: str) -> int:
    """Return 1 if correct, else 0."""
    try:
        return 1 if grade_answer(normalize_answer(pred), normalize_answer(gt)) else 0
    except Exception:
        return 0

def extract_answer_from_response(resp_text: str) -> str:
    return extract_answer(resp_text)

def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False):
    if question_type == 'multi_choice':
        s = str(extraction).strip() if isinstance(extraction, str) else str(extraction)
        if ignore_empty_extractions and not s:
            return None
        letter = None
        m1 = re.findall(r'\(([a-zA-Z])\)', s)
        if m1:
            letter = m1[0].upper()
        else:
            m2 = re.findall(r'\b([A-Za-z])\b', s)
            if m2:
                letter = m2[-1].upper()
        if letter and choices and 0 <= (ord(letter) - ord('A')) < len(choices):
            return choices[ord(letter) - ord('A')]
        return get_most_similar(s, choices or [])
    try:
        return str(extraction).strip()
    except Exception:
        return None

def safe_equal(prediction, answer):
    try:
        return bool(grade_answer(normalize_answer(prediction), normalize_answer(answer)))
    except Exception:
        return False

def get_acc_with_condition(res_pd, key, value):
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

    target_keys = ['question_type', 'language', 'source', 'category', 'task', 'context', 'grade', 'skills']
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

_CHOICE_TYPE_ALIASES = {"multi-choice", "multi_choice", "multichoice", "multi choice"}

def is_choice_question(item: Dict[str, Any], min_choices_count: int = 2) -> bool:
    qtype = (item.get("question_type") or "")
    qnorm = re.sub(r"[\s_\-]+", " ", str(qtype).strip().lower())
    if qnorm in _CHOICE_TYPE_ALIASES:
        return True
    choices = item.get("choices", None)
    if isinstance(choices, (list, tuple)) and len(choices) >= max(1, int(min_choices_count)):
        return True
    return False

INSTR_SUFFIX = "\nFirst output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Answer with the option's letter from the given choices directly."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=None,
                        help="If not set: default to 1.0 when bon_n>1 (avg@N), else 0.0.")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--bon_n", type=int, default=1, help="Number of candidates per input (avg@N).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--parallel-mode", type=str, default="dp", choices=["dp", "tp", "pp", "auto"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-backend", type=str, default="mp", choices=["mp", "ray"])
    parser.add_argument("--ds_name", type=str, default="MathVerse_testmini", choices=list(ds_collections.keys()))
    parser.add_argument("--cache_dir", type=str, default=os.path.join(os.getcwd(), "data/MathVerse/"))
    parser.add_argument("--max_num_problems", type=int, default=-1)
    parser.add_argument(
        "--only-choice-questions",
        "--only_choice_questions",
        dest="only_choice_questions",
        action="store_true",
        help="Keep only choice-based items for evaluation.",
    )
    parser.add_argument(
        "--min-choices-count",
        type=int,
        default=2,
        help="Minimum number of answer options required by the choice-question filter.",
    )
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
    logger = logging.getLogger("MathVerseEval-1JSON")

    if args.temperature is None:
        effective_temperature = 1.0 if args.bon_n > 1 else 0.0
    else:
        effective_temperature = float(args.temperature)

    cfg = ds_collections[args.ds_name]
    root = cfg["root"]
    name = cfg["name"]

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

    logger.info(f"Loading dataset {args.ds_name} => {root} (config={name}, cache_dir={args.cache_dir}) ...")
    try:
        data: Dataset = load_dataset(root, name=name, split="train", cache_dir=args.cache_dir)
    except Exception as e:
        logger.warning(f"load_dataset(..., split='train') failed: {e}. Falling back to first available split.")
        ds_dict = load_dataset(root, name=name, cache_dir=args.cache_dir)
        first_split = next(iter(ds_dict.keys()))
        data: Dataset = ds_dict[first_split]

    total_before = len(data)

    if args.only_choice_questions:
        logger.info(f"Filtering to choice-based items (min_choices_count={args.min_choices_count}) ...")
        data = data.filter(lambda x: is_choice_question(x, args.min_choices_count))
        logger.info(f"Choice-question filter: {total_before} -> {len(data)} items kept.")

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
        pid = str(item.get("pid", item.get("sample_index", item.get("problem_index", idx))))
        ans = item["answer"]
        choices = item.get("choices", None)
        qtype = "multi_choice" if is_choice_question(item, args.min_choices_count) else "open_ended"
        atype = item.get("answer_type", None)
        prec = item.get("precision", None)
        meta = item.get("metadata", {})

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

        tf_ints: List[int] = [qa_accuracy((p or ""), str(ans).strip()) for p in predictions_list]
        n_cand = max(1, int(sampling_params.n))
        avg_correct = float(sum(tf_ints)) / float(n_cand)

        first_index = 0
        resp = responses[first_index] if responses else ""
        extraction = extractions_list[first_index] if extractions_list else ""
        norm_pred = predictions_list[first_index] if predictions_list else None

        results_by_pid[pid] = {
            "pid": pid,
            "question": item.get("question"),
            "image_path": packed.get("_image_path_for_log", None),
            "answer": ans,
            "choices": choices,
            "question_type": qtype,
            "answer_type": atype,
            "precision": prec,
            "metadata": meta,

            "responses": responses,
            "extractions_list": extractions_list,
            "predictions_list": predictions_list,
            "true_false_list": [bool(x) for x in tf_ints],

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
            "name": name,
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
            "scoring": "avg@N with qa_accuracy",
            "only_choice_questions": bool(args.only_choice_questions),
            "min_choices_count": int(args.min_choices_count),
        },
        "scores": scores,
        "results": results_by_pid
    }
    out_path = os.path.join(args.out_dir, f"{base}_merged.json")
    save_json(merged, out_path)
    logging.getLogger("MathVerseEval-1JSON").info(f"Saved ONE merged JSON to: {out_path}")

if __name__ == "__main__":
    main()
