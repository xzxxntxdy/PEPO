"""Reward plugins used by PEPO training."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from datetime import datetime
from typing import List

from mathruler.grader import grade_answer
from swift.plugin.orm import ORM, orms

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _maybe_log_format_results(completions: List[str], matches: List[bool]) -> None:
    if os.getenv("DEBUG_MODE") != "true":
        return

    log_path = os.getenv("LOG_PATH")
    if not log_path:
        return

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8") as handle:
        handle.write(f"------------- {current_time} Format reward -------------\n")
        for content, match in zip(completions, matches):
            handle.write(f"Content: {content}\n")
            handle.write(f"Has correct format: {bool(match)}\n")


class Format(ORM):

    def __call__(self, completions, solution, **kwargs):
        matches = []
        for content in completions:
            text = "" if content is None else str(content)
            think_matches = THINK_PATTERN.findall(text)
            answer_matches = ANSWER_PATTERN.findall(text)
            matches.append(len(think_matches) == 1 and len(answer_matches) == 1)

        _maybe_log_format_results(completions, matches)
        return [1.0 if match else 0.0 for match in matches]


class QA_Accuracy(ORM):

    @staticmethod
    def normalize_answer(ans: str) -> str:
        if ans is None:
            return ""

        text = unicodedata.normalize("NFKC", str(ans))
        text = text.casefold()
        text = re.sub(r"\s+", " ", text).strip()
        if text.endswith(".") or text.endswith("\u3002"):
            text = text[:-1].strip()
        return text.replace("\u03c0", "\\pi")

    def __call__(self, completions, solution, **kwargs):
        results = []

        for pred, gt in zip(completions, solution):
            try:
                match = ANSWER_PATTERN.search("" if pred is None else str(pred))
                if not match:
                    results.append(0.0)
                    continue

                pred_answer = self.normalize_answer(match.group(1).strip())
                ground_truth = self.normalize_answer(gt)
                results.append(1.0 if grade_answer(ground_truth, pred_answer) else 0.0)
            except Exception as exc:
                logger.exception("[QA_Accuracy] Error while computing accuracy: %s", exc)
                results.append(0.0)

        return results


orms["external_format"] = Format
orms["external_qa_acc"] = QA_Accuracy
