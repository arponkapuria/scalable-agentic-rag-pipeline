"""
Single entrypoint for every eval run, per EVALUATION_DESIGN.md's "each is
its own CLI command, resumable on its own":

  python -m eval.cli ingest
  python -m eval.cli verify-gold
  python -m eval.cli retrieve
  python -m eval.cli generate [--closed-book]
  python -m eval.cli judge-check
  python -m eval.cli judge (--retrieval | --generation)
  python -m eval.cli report

Run from the repo root (needs services/, pipelines/, libs/, models/, eval/
all importable — same PYTHONPATH assumption as every other standalone
script in scripts/).
"""
import argparse
import asyncio
import logging


def main():
    parser = argparse.ArgumentParser(prog="python -m eval.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest")
    sub.add_parser("verify-gold")
    sub.add_parser("retrieve")

    gen_parser = sub.add_parser("generate")
    gen_parser.add_argument("--closed-book", action="store_true")

    judge_check_parser = sub.add_parser("judge-check")
    judge_check_parser.add_argument("--judge-backend", choices=["gemma", "cohere"], default="gemma")

    judge_parser = sub.add_parser("judge")
    judge_parser.add_argument("--retrieval", action="store_true")
    judge_parser.add_argument("--generation", action="store_true")
    judge_parser.add_argument("--judge-backend", choices=["gemma", "cohere"], default="gemma")

    sub.add_parser("report")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "ingest":
        from eval import ingest_corpus
        asyncio.run(ingest_corpus.run())
    elif args.command == "verify-gold":
        from eval import verify_gold
        asyncio.run(verify_gold.main())
    elif args.command == "retrieve":
        from eval import run_retrieval
        asyncio.run(run_retrieval.run())
    elif args.command == "generate":
        from eval import run_generation
        asyncio.run(run_generation.run(args.closed_book))
    elif args.command == "judge-check":
        from eval import judge_check
        asyncio.run(judge_check.main(args.judge_backend))
    elif args.command == "judge":
        if args.retrieval == args.generation:
            judge_parser.error("pass exactly one of --retrieval / --generation")
        from eval import run_judge
        asyncio.run(run_judge.run("retrieval" if args.retrieval else "generation", args.judge_backend))
    elif args.command == "report":
        from eval import report
        report.main()


if __name__ == "__main__":
    main()