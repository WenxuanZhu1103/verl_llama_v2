import re
import logging
from contextlib import contextmanager

try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


@contextmanager
def suppress_math_verify_logs():
    """上下文管理器，用于临时抑制math_verify的日志输出"""
    loggers_to_suppress = [
        'math_verify',
        'math_verify.grader',
        'math_verify.utils'
    ]
    
    original_levels = {}
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.CRITICAL)
    
    try:
        yield
    finally:
        for logger_name, original_level in original_levels.items():
            logging.getLogger(logger_name).setLevel(original_level)


def extract_boxed_answer(text: str) -> str:
    patterns = [
        r'\\boxed\{([^}]*)\}',             # Most specific: exact \boxed{content}
        r'\\boxed\s*\{\s*([^}]*?)\s*\}',   # With whitespace handling
        r'boxed\s*\{\s*([^}]*?)\s*\}',     # Without leading backslash
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip()
            if answer:
                return answer
    return None  


def compute_score(model_output: str, ground_truth: str, timeout_score=0) -> tuple[bool, float]:
    with suppress_math_verify_logs():
        if not ground_truth:
            print("Warning: ground_truth is None. Returning timeout_score.")
            return True, timeout_score

        # ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        extracted_output = extract_boxed_answer(model_output)
        if extracted_output is None or extracted_output == "":
            # print("Unabled to extract answer from model output: ", model_output)
            return False, timeout_score

        if ground_truth.strip() == extracted_output.strip():
            return True, 1.0
        else:
            return True, 0.0