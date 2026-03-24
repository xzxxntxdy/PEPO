"""Geometry3K evaluation with vLLM and avg@N scoring."""

import argparse
from collections import Counter
import json
import os
import re
from typing import Any, Dict, List, Optional

from mathruler.grader import grade_answer
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def should_use_qwen_multimodal(model_type: str, model_path: str, processor: Optional[AutoProcessor] = None) -> bool:
    model_type = (model_type or "").strip().lower()
    if model_type == "qwen":
        return True
    if model_type == "none":
        return False
    if model_type not in {"", "auto"}:
        return "qwen" in model_type
    if model_path and "qwen" in model_path.lower():
        return True
    if processor is not None and "qwen" in processor.__class__.__name__.lower():
        return True
    return False


def prepare_qwen_image(messages):
    from qwen_vl_utils import process_vision_info
    image_obj, _ = process_vision_info(messages)
    return image_obj
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
    return 1 if grade_answer(normalize_answer(pred), normalize_answer(gt)) else 0


def majority_vote(preds: List[str]) -> str:
    """Return the majority-voted answer among normalized candidates.
       If tie, return the first occurrence."""
    if not preds:
        return ""
    norm = [normalize_answer(p) for p in preds]
    cnt = Counter(norm)
    best_norm, _ = max(cnt.items(), key=lambda kv: (kv[1], -norm.index(kv[0])))
    for p in preds:
        if normalize_answer(p) == best_norm:
            return p
    return preds[0]

def build_inputs_for_examples(examples: List[Dict[str, Any]],
                              processor: AutoProcessor,
                              use_qwen_multimodal: bool = False):
    """Return input list (for vLLM.generate) and metadata list."""
    inputs = []
    meta = []
    for ex in examples:
        image_path = ex["images"][0]
        user_prompt = (ex["messages"][0]["content"] or "").replace("<image>\n", "")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if use_qwen_multimodal:
            image_data = prepare_qwen_image(messages)
        else:
            image_data = Image.open(image_path).convert('RGB')
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_data}})
        meta.append((image_path, user_prompt, ex["solution"]))
    return inputs, meta


def batched_generate(all_examples: List[Dict[str, Any]],
                     llm: LLM,
                     processor: AutoProcessor,
                     sampling_params: SamplingParams,
                     use_qwen_multimodal: bool = False,
                     group_size: int = 0,
                     use_tqdm: bool = True):
    """Build inputs in groups and call vLLM.generate once per group."""
    if group_size is None or group_size <= 0:
        group_size = len(all_examples)

    results = []
    total_avg_score = 0.0

    for start in tqdm(range(0, len(all_examples), group_size), desc="groups"):
        chunk = all_examples[start:start + group_size]
        inputs, meta = build_inputs_for_examples(chunk, processor, use_qwen_multimodal=use_qwen_multimodal)

        outs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=use_tqdm)

        for (image_path, user_prompt, gt), req_out in zip(meta, outs):
            gens = [o.text for o in req_out.outputs]
            preds = [extract_answer(t) for t in gens]
            scores = [qa_accuracy(p, gt) for p in preds]
            avg_score = float(sum(scores)) / max(len(scores), 1)

            voted_pred = majority_vote(preds)

            total_avg_score += avg_score
            results.append({
                "image": image_path,
                "prompt": user_prompt,
                "ground_truth": gt,
                "model_output": gens,
                "predicted": voted_pred,
                "score": avg_score,
                "avg_score": avg_score,
                "all_predictions": preds,
                "all_scores": scores
            })

    return results, total_avg_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--model_path", type=str, required=True, help="vLLM model name or path")
    parser.add_argument("--processor_path", type=str, default=None, help="Processor name or path. Defaults to model_path.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--model-type", "--model_type", dest="model_type", type=str, default="auto")
    parser.add_argument("--bon_n", type=int, default=1, help="n candidates per input (used by avg@N).")
    parser.add_argument("--bon_temperature", type=float, default=1.0)
    parser.add_argument("--bon_top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--group_size", type=int, default=0, help="0 or negative ⇒ all-in-one; else per-group size.")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        all_examples = [json.loads(l) for l in f]
    if args.limit > 0:
        all_examples = all_examples[:args.limit]

    processor_path = args.processor_path or args.model_path
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    use_qwen_multimodal = should_use_qwen_multimodal(args.model_type, args.model_path, processor)
    if use_qwen_multimodal:
        print("Using Qwen-specific multimodal preprocessing.", flush=True)

    if args.bon_n > 1:
        sampling_params = SamplingParams(
            n=args.bon_n,
            temperature=args.bon_temperature,
            top_p=args.bon_top_p,
            max_tokens=args.max_new_tokens,
        )
    else:
        sampling_params = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    results, total_avg_score = batched_generate(
        all_examples,
        llm=llm,
        processor=processor,
        sampling_params=sampling_params,
        use_qwen_multimodal=use_qwen_multimodal,
        group_size=args.group_size,
        use_tqdm=True,
    )

    accuracy = (total_avg_score / max(len(results), 1)) * 100.0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": round(accuracy, 2),
            "total": len(results),
            "correct_like": round(float(total_avg_score), 4),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Accuracy (avg@N): {accuracy:.2f}% on {len(results)} samples")
    print(f"≈ Correct-like (sum of per-sample avg scores): {total_avg_score:.4f}")
    print(f"📁 Results saved to {args.output}")


if __name__ == "__main__":
    main()
