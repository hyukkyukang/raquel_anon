# get_schema_stats.py
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import hkkang_utils.misc as misc_utils

misc_utils.load_dotenv()

import logging
from typing import *
from pathlib import Path

import hydra
from omegaconf import DictConfig

import pandas as pd
import matplotlib.pyplot as plt

from src.aligned_db.utils import get_schema_stats
from src.utils.logging import get_logger

logger = get_logger(__name__, __file__)

SAMPLE_NUMS = [1, 2, 4, 8, 16, 20, 32, 64, 100]


@hydra.main(
    version_base=None, config_path="/home/user/RAQUEL/config", config_name="config"
)
def main(cfg: DictConfig) -> None:
    """
    For each sample size in SAMPLE_NUMS
      • read the aligned schema file
      • compute schema statistics
    Then save a CSV-like log and a comparison figure.
    """
    project_path: str = cfg.project_path
    assert project_path, "`project_path` must be set in the Hydra config."

    rows: List[Dict[str, Any]] = []

    for n in SAMPLE_NUMS:
        schema_file = (
            Path(project_path) / f"aligned_dbs/sample_{n}/combined_db/schema.sql"
        )
        if not schema_file.exists():
            logger.warning(f"[skip] schema not found for sample_{n}: {schema_file}")
            continue

        logger.info(f"Reading {schema_file} …")
        schema = schema_file.read_text(encoding="utf-8")
        stats = get_schema_stats(schema)
        stats["sample_num"] = n
        rows.append(stats)

    if not rows:
        logger.error("No schemas processed — nothing to plot.")
        return

    # -------------------------------------------------
    # aggregate → DataFrame (good for printing & plotting)
    # -------------------------------------------------
    df_stats = pd.DataFrame(rows).sort_values("sample_num").reset_index(drop=True)

    # pretty-print to log
    logger.info("Schema statistics:\n" + df_stats.to_markdown(index=False))

    # -------------------------------------------------
    # draw *one* line chart (no sub-plots)
    # -------------------------------------------------
    plt.figure(figsize=(7, 4))

    for metric in ["num_tables", "num_columns", "avg_columns_per_table"]:
        plt.plot(df_stats["sample_num"], df_stats[metric], marker="o", label=metric)

    plt.xlabel("sample_num (# QA pairs)")
    plt.ylabel("value")
    plt.title("Schema statistics vs. sample size")
    plt.legend()
    plt.tight_layout()

    out_png = Path(project_path) / "schema_stats.png"
    plt.savefig(out_png, dpi=150)
    logger.info(f"Figure saved to {out_png}")

    # optional: also save raw stats as JSON or CSV
    out_json = Path(project_path) / "schema_stats.json"
    out_json.write_text(df_stats.to_json(orient="records", indent=2))
    logger.info(f"Stats JSON saved to {out_json}")

    # -------------------------------------------------
    # draw a grouped bar chart (single axes, no sub-plots)
    # -------------------------------------------------
    plt.figure(figsize=(8, 4.5))

    metrics = [
        "num_tables",
        "num_columns",
        "avg_columns_per_table",
        "max_columns_per_table",
        "min_columns_per_table",
    ]
    x = df_stats["sample_num"].to_numpy()
    x_indices = range(len(x))
    bar_width = 0.15  # width of each bar
    offsets = [(i - 2) * bar_width for i in range(len(metrics))]  # centered

    for off, metric in zip(offsets, metrics):
        plt.bar(
            [xi + off for xi in x_indices],  # shifted positions
            df_stats[metric],
            width=bar_width,
            label=metric.replace("_", " "),  # nicer legend labels
        )

    plt.xticks(x_indices, x)  # tick labels are the sample sizes
    plt.xlabel("sample_num (# QA pairs)")
    plt.ylabel("value")
    plt.title("Schema statistics vs. sample size")
    plt.legend()
    plt.tight_layout()

    out_png = Path(project_path) / "schema_stats_grouped.png"
    plt.savefig(out_png, dpi=150)
    logger.info(f"Figure saved to {out_png}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
    logging.info("Done!")
