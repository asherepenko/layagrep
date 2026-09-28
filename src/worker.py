#!/usr/bin/env python
"""layagrep worker: keeps a laya-mlx agent resident and answers JSONL requests.

Protocol (one JSON object per line, both directions):
  request:  {"id": <int>, "state": <str|any-json>, "questions": {qid: {"type": ..., "instructions": ..., "criteria": ...}}}
  response: {"id": <int>, "ok": true, "answers": {qid: number|object}, "tokens": <int>}
            {"id": <int>, "ok": false, "error": "<Type: message>"}

Answer extraction: noul -> P(true) float, choice -> {label: p}, score -> expected score.
The worker prints {"ready": true, ...} once the model is loaded, then serves stdin.
"""
import argparse
import json
import sys
import time


def log(message):
    sys.stderr.write("[layagrep-worker] %s\n" % message)
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(prog="layagrep-worker")
    parser.add_argument("--model", default="aac6fef/laya-mlx")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None, choices=[None, "gpu", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import laya_mlx as laya

    started = time.time()
    agent = laya.load(
        args.model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )
    log("model %s ready in %.2fs" % (args.model, time.time() - started))
    sys.stdout.write(
        json.dumps({"ready": True, "model": args.model, "dtype": args.dtype}) + "\n"
    )
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            request = json.loads(line)
            rid = request.get("id")
            result = agent.predict(request["state"], request["questions"])
            answers = {}
            for qid, answer in result["answers"].items():
                kind = answer.get("type")
                if kind == "noul":
                    answers[qid] = answer["noul"]
                elif kind == "choice":
                    answers[qid] = answer["probabilities"]
                elif kind == "score":
                    answers[qid] = answer["score"]
                else:
                    raise ValueError("unsupported answer type %r" % kind)
            out = {
                "id": rid,
                "ok": True,
                "answers": answers,
                "tokens": result.get("usage", {}).get("input_tokens", 0),
            }
        except KeyboardInterrupt:
            raise
        except Exception as error:  # surfaced to the Rust side per request
            out = {"id": rid, "ok": False, "error": "%s: %s" % (type(error).__name__, error)}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
