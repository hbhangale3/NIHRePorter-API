from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="reporter_outreach")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional safety limit for pagination")

    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_yaml = config_path.read_text(encoding="utf-8")

    results, summary, keyword_expansions = run_pipeline(config_yaml, max_pages=args.max_pages)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if keyword_expansions:
        (out_dir / "keyword_expansions.json").write_text(json.dumps(keyword_expansions, indent=2), encoding="utf-8")

    # CSV generation via local import to keep CSV schema consistent
    from .models import PIOutreachRow
    from .csv_export import rows_to_csv_bytes

    rows = [PIOutreachRow.model_validate(r) for r in results]
    (out_dir / "outreach.csv").write_bytes(rows_to_csv_bytes(rows))


if __name__ == "__main__":
    main()
