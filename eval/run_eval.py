"""Entry point required by the brief: uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

The logic lives in src/aides_rag/evaluation/run_eval.py (importable and tested).
"""

from aides_rag.evaluation.run_eval import main

if __name__ == "__main__":
    main()
