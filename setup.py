from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
MS_SWIFT_PATH = (ROOT / "third_party" / "ms-swift").resolve()
MS_SWIFT_REQUIREMENT = f"ms_swift @ {MS_SWIFT_PATH.as_uri()}"
PEPO_REQUIREMENTS = [
    "mathruler",
    "python-Levenshtein",
    "qwen_vl_utils",
    "rich",
    "vllm",
]


setup(
    name="pepo-vlm",
    version="0.1.0",
    description="PEPO training and evaluation code for multimodal reasoning.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    author="PEPO Authors",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[MS_SWIFT_REQUIREMENT, *PEPO_REQUIREMENTS],
    entry_points={
        "console_scripts": [
            "pepo-resolve-image-paths=pepo.data.resolve_image_paths:main",
            "pepo-train=pepo.train.rlhf:main",
            "pepo-eval-geometry3k=pepo.evaluation.evaluate_geometry3k:main",
            "pepo-eval-logicvista=pepo.evaluation.evaluate_logicvista:main",
            "pepo-eval-mathverse=pepo.evaluation.evaluate_mathverse:main",
            "pepo-eval-mathvista=pepo.evaluation.evaluate_mathvista:main",
        ],
    },
)
