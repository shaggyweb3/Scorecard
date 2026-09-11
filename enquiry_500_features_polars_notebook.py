# %% [markdown]
# # ENQ Enquiry Feature Engineering Notebook - Polars Only
#
# This notebook-style Python file is meant for VS Code interactive notebooks.
# Run each `# %%` cell one by one.
#
# Goal:
# - Read an ENQ enquiry file.
# - Validate the required schema.
# - Map enquiry purpose codes to descriptions, secured/unsecured status, and product groups.
# - Exclude future enquiries where `enq_date > input_date`.
# - Diagnose duplicates before deduplication.
# - Generate an MRN-input_date level candidate feature table.
# - Keep up to 500 business-prioritized non-constant candidate variables.
# - Export the final feature table, feature dictionary, diagnostics, and feature-quality summary.
#
# Important:
# These are candidate variables, not final model variables. Statistical significance requires target/outcome data.

# %%
# Cell 1 - Imports
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)

print("Polars version:", pl.__version__)

# %% [markdown]
# ## Cell 2 - User Inputs
#
# Change only this cell for your real file.
#
# Supported input formats:
# - `csv`
# - `parquet`
# - `excel`
#
# Output grain will be:
# `custid + mrn + input_date`

# %%
# Cell 2 - Parameters
INPUT_FILE = r"C:\path\to\your\enquiry_file.csv"
INPUT_FORMAT = "csv"  # csv, parquet, or excel
OUTPUT_DIR = Path("outputs/enquiry_500_feature_notebook_run")

# Dedup strategy:
# none     = no duplicate removal
# exact    = remove fully identical source rows
# probable = remove probable duplicated enquiries using ECN/fallback keys
DEDUP_STRATEGY = "exact"

# Max number of candidate feature columns to keep in the final modelling table.
# The code first creates a larger controlled feature pool, removes all-null/constant
# variables, then keeps the first 500 in business-priority order.
MAX_FEATURES = 500

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Output directory:", OUTPUT_DIR.resolve())

# %% [markdown]
# ## Cell 3 - Read ENQ File
#
# The required input columns are:
# `custid, mrn, input_date, run_date, short_name, ecn, enq_date, enq_purpose, enq_amt`

# %%
# Cell 3 - File reader
def read_enq_file(input_file: str | Path, input_format: str) -> pl.DataFrame:
    path = Path(input_file)
    if input_format == "csv":
        return pl.read_csv(path, infer_schema_length=10000)
    if input_format == "parquet":
        return pl.read_parquet(path)
    if input_format == "excel":
        return pl.read_excel(path)
    raise ValueError("INPUT_FORMAT must be csv, parquet, or excel")


raw_enq_df = read_enq_file(INPUT_FILE, INPUT_FORMAT)

print("Raw ENQ shape:", raw_enq_df.shape)
raw_enq_df.head(5)

# %% [markdown]
# ## Cell 4 - Schema Validation
#
# Repeated MRNs are expected because the file is enquiry-event level.
# One MRN/application observation can have many enquiry events.
# Do not remove rows just because MRN repeats.

# %%
# Cell 4 - Required schema checks
REQUIRED_COLUMNS = [
    "custid",
    "mrn",
    "input_date",
    "run_date",
    "short_name",
    "ecn",
    "enq_date",
    "enq_purpose",
    "enq_amt",
]


def standardize_col(name: str) -> str:
    """Make column names consistent across files."""
    return "_".join(name.strip().lower().replace("-", "_").split())


raw_enq_df = raw_enq_df.rename({c: standardize_col(c) for c in raw_enq_df.columns})
missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_enq_df.columns]

print("Columns:", raw_enq_df.columns)
print("Dtypes:", raw_enq_df.schema)
print("Missing required columns:", missing_cols)

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# %% [markdown]
# ## Cell 5 - Purpose Glossary And Business Segments
#
# Unknown purpose codes are not mapped to `Other`.
# Unknowns are assigned:
# - `purpose_description = UNKNOWN_CODE`
# - `security_group = Unknown`
# - `broad_product_group = unknown`

# %%
# Cell 5 - Purpose mapping configuration
PURPOSE_GLOSSARY: dict[int, tuple[str, str, str]] = {
    0: ("Other", "Unsecured", "other"),
    1: ("Auto Loan Personal", "Secured", "vehicle"),
    2: ("Housing Loan", "Secured", "housing"),
    3: ("Property Loan", "Secured", "housing"),
    4: ("Loan Against Shares/Securities", "Secured", "asset_backed"),
    5: ("Personal Loan", "Unsecured", "personal"),
    6: ("Consumer Loan", "Unsecured", "personal"),
    7: ("Gold Loan", "Secured", "asset_backed"),
    8: ("Education Loan", "Unsecured", "education"),
    9: ("Loan to Professional", "Unsecured", "professional"),
    10: ("Credit Card", "Unsecured", "card"),
    11: ("Leasing", "Secured", "vehicle"),
    12: ("Overdraft", "Unsecured", "overdraft"),
    13: ("Two-wheeler Loan", "Secured", "vehicle"),
    14: ("Non-Funded Credit Facility", "Secured", "business"),
    15: ("Loan Against Bank Deposits", "Secured", "asset_backed"),
    16: ("Fleet Card", "Unsecured", "card"),
    17: ("Commercial Vehicle Loan", "Secured", "vehicle"),
    21: ("Seller Financing", "Secured", "business"),
    23: ("GECL Loan Secured", "Secured", "government_scheme"),
    24: ("GECL Loan Unsecured", "Unsecured", "government_scheme"),
    25: ("Working Capital Loan", "Unsecured", "business"),
    26: ("Commercial Equipment Loan", "Secured", "business"),
    27: ("Agriculture Loan", "Secured", "agriculture"),
    28: ("Overdraft against FD", "Secured", "asset_backed"),
    29: ("Overdraft against Shares/Securities", "Secured", "asset_backed"),
    30: ("Overdraft against Collateral", "Secured", "asset_backed"),
    31: ("Secured Credit Card", "Secured", "card"),
    32: ("Used Car Loan", "Secured", "vehicle"),
    33: ("Construction Equipment Loan", "Secured", "business"),
    34: ("Tractor Loan", "Secured", "agriculture"),
    35: ("Corporate Credit Card", "Unsecured", "card"),
    36: ("Kisan Credit Card", "Unsecured", "agriculture"),
    37: ("Loan on Credit Card", "Unsecured", "card"),
    38: ("PM Jan Dhan Yojana Overdraft", "Unsecured", "government_scheme"),
    39: ("Mudra Loans", "Unsecured", "business"),
    44: ("PMAY CLSS", "Secured", "government_scheme"),
    45: ("P2P Personal Loan", "Unsecured", "personal"),
    46: ("P2P Auto Loan", "Secured", "vehicle"),
    47: ("P2P Education Loan", "Unsecured", "education"),
    50: ("Business Loan Secured", "Secured", "business"),
    51: ("Business Loan General", "Secured", "business"),
    52: ("Business Loan Priority Sector Small Business", "Secured", "business"),
    53: ("Business Loan Priority Sector Agriculture", "Secured", "agriculture"),
    54: ("Business Loan Priority Sector Others", "Secured", "business"),
    55: ("Business Non-Funded Credit Facility General", "Secured", "business"),
    56: ("Business Non-Funded Credit Facility Small Business", "Secured", "business"),
    57: ("Business Non-Funded Credit Facility Agriculture", "Secured", "agriculture"),
    58: ("Business Non-Funded Credit Facility Others", "Secured", "business"),
    59: ("Business Loan Against Bank Deposits", "Secured", "asset_backed"),
    61: ("Business Loan Unsecured", "Unsecured", "business"),
    69: ("Short Term Personal Loan", "Unsecured", "personal"),
    70: ("Priority Sector Gold Loan", "Secured", "asset_backed"),
    71: ("Temporary Overdraft", "Unsecured", "overdraft"),
}

SBI_ALIASES = {"SBI", "STATE BANK OF INDIA"}
NOT_DISCLOSED_VALUES = {"NOT DISCLOSED", "NOTDISCLOSED", "UNDISCLOSED"}

WINDOWS: list[int | None] = [7, 15, 30, 60, 90, 180, 365, 730, None]
SEGMENTS = [
    "all",
    "secured",
    "unsecured",
    "personal",
    "professional",
    "business",
    "card",
    "overdraft",
    "education",
    "vehicle",
    "housing",
    "asset_backed",
    "sbi",
    "non_sbi_or_not_disclosed_lender",
    "unknown",
]
PRIORITY_PURPOSE_CODES = [5, 6, 8, 9, 10, 12, 25, 37, 39, 50, 51, 61, 69, 71]
AMOUNT_BANDS = [
    ("missing", None, None),
    ("non_positive", None, 0),
    ("gt_0_to_50k", 0, 50_000),
    ("gt_50k_to_100k", 50_000, 100_000),
    ("gt_100k_to_200k", 100_000, 200_000),
    ("gt_200k_to_500k", 200_000, 500_000),
    ("gt_500k_to_1m", 500_000, 1_000_000),
    ("gt_1m_to_2_5m", 1_000_000, 2_500_000),
    ("gt_2_5m_to_5m", 2_500_000, 5_000_000),
    ("gt_5m", 5_000_000, None),
]

# Expert feature-control plan:
# We do not create every metric for every segment/window because that produces
# many repetitive variables. Instead, use broader coverage for core segments and
# lighter coverage for narrower business/product/lender segments.
CORE_SEGMENTS = ["all", "secured", "unsecured"]
PRODUCT_SEGMENTS = ["personal", "professional", "business", "card", "overdraft", "education", "vehicle", "housing", "asset_backed"]
LENDER_SEGMENTS = ["sbi", "non_sbi_or_not_disclosed_lender"]
QUALITY_SEGMENTS = ["unknown"]
PRODUCT_WINDOWS: list[int | None] = [30, 90, 180, 365, 730, None]
LENDER_WINDOWS: list[int | None] = [30, 90, 180, 365, 730, None]
AMOUNT_BAND_WINDOWS: list[int | None] = [30, 90, 180, 365, 730, None]
PURPOSE_CODE_WINDOWS: list[int | None] = [90, 365, 730, None]
RATIO_WINDOWS: list[int | None] = [30, 90, 180, 365, 730, None]
CONCENTRATION_WINDOWS: list[int | None] = [90, 180, 365, 730, None]

purpose_map_df = pl.DataFrame(
    [
        {
            "purpose_code": code,
            "purpose_description": values[0],
            "security_group": values[1],
            "broad_product_group": values[2],
        }
        for code, values in PURPOSE_GLOSSARY.items()
    ]
)

purpose_map_df.head(10)

# %% [markdown]
# ## Cell 6 - Prepare Event-Level Data
#
# This cell:
# - preserves identifiers as strings
# - parses dates
# - calculates days before observation
# - flags future enquiries
# - maps purpose codes
# - standardizes lender and ECN
# - creates exact/probable duplicate flags

# %%
# Cell 6 - Event preparation functions
def parse_date_expr(column: str) -> pl.Expr:
    raw = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.coalesce(
        [
            raw.str.to_date("%Y-%m-%d", strict=False),
            raw.str.to_date("%d-%m-%Y", strict=False),
            raw.str.to_date("%d/%m/%Y", strict=False),
            raw.str.to_date("%Y/%m/%d", strict=False),
        ]
    )


def prepare_events(df: pl.DataFrame) -> pl.DataFrame:
    prepared = df.with_columns(
        [
            pl.col("custid").cast(pl.Utf8, strict=False),
            pl.col("mrn").cast(pl.Utf8, strict=False),
            pl.col("short_name").cast(pl.Utf8, strict=False),
            pl.col("ecn").cast(pl.Utf8, strict=False),
            pl.col("enq_purpose").cast(pl.Int64, strict=False).alias("purpose_code"),
            pl.col("enq_amt").cast(pl.Float64, strict=False),
            parse_date_expr("input_date").alias("input_date_parsed"),
            parse_date_expr("run_date").alias("run_date_parsed"),
            parse_date_expr("enq_date").alias("enq_date_parsed"),
        ]
    )

    prepared = prepared.join(purpose_map_df, on="purpose_code", how="left")
    prepared = prepared.with_columns(
        [
            pl.col("purpose_description").fill_null("UNKNOWN_CODE"),
            pl.col("security_group").fill_null("Unknown"),
            pl.col("broad_product_group").fill_null("unknown"),
            pl.col("short_name").str.strip_chars().str.to_uppercase().str.replace_all(r"\s+", " ").alias("lender_std"),
            pl.col("ecn").str.strip_chars().str.to_uppercase().str.replace_all(r"\s+", " ").alias("ecn_std"),
        ]
    )

    prepared = prepared.with_columns(
        [
            pl.col("lender_std").is_null().alias("missing_lender_flag"),
            pl.col("ecn_std").is_null().alias("missing_ecn_flag"),
            pl.col("lender_std").is_in(list(NOT_DISCLOSED_VALUES)).fill_null(False).alias("not_disclosed_lender_flag"),
            pl.col("ecn_std").is_in(list(NOT_DISCLOSED_VALUES)).fill_null(False).alias("not_disclosed_ecn_flag"),
            pl.col("lender_std").is_in(list(SBI_ALIASES)).fill_null(False).alias("is_sbi_lender"),
            (pl.col("purpose_description") == "UNKNOWN_CODE").alias("unknown_purpose_flag"),
            pl.col("enq_amt").is_null().alias("missing_amount_flag"),
            (pl.col("enq_amt") == 0).fill_null(False).alias("zero_amount_flag"),
            (pl.col("enq_amt") < 0).fill_null(False).alias("negative_amount_flag"),
        ]
    )

    prepared = prepared.with_columns(
        [
            pl.when(pl.col("not_disclosed_ecn_flag") | pl.col("missing_ecn_flag"))
            .then(None)
            .otherwise(pl.col("ecn_std"))
            .alias("valid_ecn"),
            (pl.col("input_date_parsed") - pl.col("enq_date_parsed")).dt.total_days().alias("days_before_observation"),
            (pl.col("run_date_parsed") - pl.col("enq_date_parsed")).dt.total_days().alias("days_between_enquiry_and_run"),
            (pl.col("run_date_parsed") - pl.col("input_date_parsed")).dt.total_days().alias("days_between_input_and_run"),
        ]
    )

    prepared = prepared.with_columns(
        [
            (pl.col("days_before_observation") < 0).fill_null(False).alias("enq_after_input_date_flag"),
            (pl.col("days_between_enquiry_and_run") < 0).fill_null(False).alias("enq_after_run_date_flag"),
            (pl.col("days_between_input_and_run") < 0).fill_null(False).alias("input_date_after_run_date_flag"),
            pl.col("enq_date_parsed").is_null().alias("enq_date_missing_flag"),
            pl.col("input_date_parsed").is_null().alias("input_date_missing_flag"),
            pl.col("run_date_parsed").is_null().alias("run_date_missing_flag"),
            (pl.col("days_before_observation") == 0).fill_null(False).alias("same_day_enquiry_flag"),
        ]
    )

    prepared = prepared.with_columns(
        [
            (pl.col("security_group") == "Secured").alias("is_secured"),
            (pl.col("security_group") == "Unsecured").alias("is_unsecured"),
            (pl.col("broad_product_group") == "personal").alias("is_personal"),
            (pl.col("broad_product_group") == "professional").alias("is_professional"),
            (pl.col("broad_product_group") == "business").alias("is_business"),
            (pl.col("broad_product_group") == "card").alias("is_card"),
            (pl.col("broad_product_group") == "overdraft").alias("is_overdraft"),
            (pl.col("broad_product_group") == "education").alias("is_education"),
            (pl.col("broad_product_group") == "vehicle").alias("is_vehicle"),
            (pl.col("broad_product_group") == "housing").alias("is_housing"),
            (pl.col("broad_product_group") == "asset_backed").alias("is_asset_backed"),
            (pl.col("broad_product_group") == "unknown").alias("is_unknown"),
        ]
    )

    source_cols = REQUIRED_COLUMNS
    prepared = prepared.with_columns(
        [
            pl.struct(source_cols).is_duplicated().alias("exact_duplicate_flag"),
            (pl.col("valid_ecn").is_not_null() & pl.struct(["custid", "valid_ecn"]).is_duplicated()).alias("prob_dup_valid_ecn_flag"),
            pl.struct(["custid", "enq_date_parsed", "purpose_code", "enq_amt", "lender_std"]).is_duplicated().alias("prob_dup_lender_key_flag"),
            pl.struct(["custid", "enq_date_parsed", "purpose_code", "enq_amt"]).is_duplicated().alias("prob_dup_no_lender_key_flag"),
        ]
    )
    prepared = prepared.with_columns(
        (
            pl.col("prob_dup_valid_ecn_flag")
            | pl.col("prob_dup_lender_key_flag")
            | pl.col("prob_dup_no_lender_key_flag")
        ).alias("probable_duplicate_flag")
    )
    return prepared


events_df = prepare_events(raw_enq_df)

print("Prepared event-level shape:", events_df.shape)
events_df.head(5)

# %% [markdown]
# ## Cell 7 - Diagnostics Before Deduplication
#
# Review this before deciding whether `DEDUP_STRATEGY = exact` or `probable` is appropriate.

# %%
# Cell 7 - Relationship, leakage, purpose, and duplicate diagnostics
diagnostics_df = pl.DataFrame(
    [
        {"metric": "raw_rows", "value": events_df.height},
        {"metric": "distinct_custid", "value": events_df.select(pl.col("custid").n_unique()).item()},
        {"metric": "distinct_mrn", "value": events_df.select(pl.col("mrn").n_unique()).item()},
        {
            "metric": "mrn_with_multiple_custid",
            "value": events_df.group_by("mrn").agg(pl.col("custid").n_unique().alias("n")).filter(pl.col("n") > 1).height,
        },
        {
            "metric": "mrn_with_multiple_input_date",
            "value": events_df.group_by("mrn").agg(pl.col("input_date_parsed").n_unique().alias("n")).filter(pl.col("n") > 1).height,
        },
        {"metric": "future_enquiry_rows_excluded", "value": int(events_df.select(pl.sum("enq_after_input_date_flag")).item() or 0)},
        {"metric": "unknown_purpose_rows", "value": int(events_df.select(pl.sum("unknown_purpose_flag")).item() or 0)},
        {"metric": "exact_duplicate_member_rows", "value": int(events_df.select(pl.sum("exact_duplicate_flag")).item() or 0)},
        {"metric": "probable_duplicate_member_rows", "value": int(events_df.select(pl.sum("probable_duplicate_flag")).item() or 0)},
    ]
)

purpose_validation_df = (
    events_df.group_by(["purpose_code", "purpose_description", "security_group", "broad_product_group"])
    .agg(
        [
            pl.len().alias("rows"),
            pl.col("mrn").n_unique().alias("distinct_mrns"),
            pl.col("enq_amt").sum().alias("total_amount"),
            pl.col("enq_amt").is_null().mean().alias("missing_amount_rate"),
        ]
    )
    .sort("purpose_code")
)

display(diagnostics_df)
display(purpose_validation_df)

# %% [markdown]
# Pause here and check:
# - future enquiry count
# - duplicate count
# - unknown purpose count
# - MRN/customer/date conflicts

# %% [markdown]
# ## Cell 8 - Point-In-Time Filtering And Deduplication
#
# Only rows where `enq_date <= input_date` are eligible for feature generation.
# Rows after `input_date` are leakage exceptions and are not used in features.

# %%
# Cell 8 - Eligible event population
KEY_COLS = ["custid", "mrn", "input_date_parsed"]


def apply_dedup(df: pl.DataFrame, strategy: str) -> pl.DataFrame:
    if strategy == "none":
        return df
    if strategy == "exact":
        return df.unique(subset=REQUIRED_COLUMNS, keep="first", maintain_order=True)
    if strategy == "probable":
        with_key = df.with_columns(
            pl.when(pl.col("valid_ecn").is_not_null())
            .then(pl.concat_str([pl.lit("ecn"), pl.col("custid"), pl.col("valid_ecn")], separator="|"))
            .otherwise(
                pl.concat_str(
                    [
                        pl.lit("fallback"),
                        pl.col("custid"),
                        pl.col("enq_date_parsed").cast(pl.Utf8),
                        pl.col("purpose_code").cast(pl.Utf8),
                        pl.col("enq_amt").cast(pl.Utf8),
                        pl.col("lender_std").fill_null("MISSING"),
                    ],
                    separator="|",
                )
            )
            .alias("_dedup_key")
        )
        return with_key.unique(subset=KEY_COLS + ["_dedup_key"], keep="first", maintain_order=True).drop("_dedup_key")
    raise ValueError("DEDUP_STRATEGY must be none, exact, or probable")


eligible_events_df = events_df.filter(
    pl.col("enq_date_parsed").is_not_null()
    & pl.col("input_date_parsed").is_not_null()
    & (pl.col("days_before_observation") >= 0)
)
eligible_events_df = apply_dedup(eligible_events_df, DEDUP_STRATEGY)

observation_base_df = events_df.select(["custid", "mrn", "input_date", "input_date_parsed", "run_date_parsed"]).unique(
    subset=KEY_COLS,
    keep="first",
    maintain_order=True,
)

print("Observation base rows:", observation_base_df.height)
print("Eligible event rows after point-in-time and dedup:", eligible_events_df.height)
eligible_events_df.head(5)

# %% [markdown]
# ## Cell 9 - Feature Helper Functions
#
# Feature names follow:
# `enq_[metric]_[segment]_[window]`

# %%
# Cell 9 - Reusable feature expressions
feature_dictionary_rows: list[dict[str, Any]] = []


def win_suffix(window: int | None) -> str:
    return "all" if window is None else f"{window}d"


def window_condition(window: int | None, extra_condition: pl.Expr | None = None) -> pl.Expr:
    cond = pl.col("days_before_observation").is_not_null() & (pl.col("days_before_observation") >= 0)
    if window is not None:
        cond = cond & (pl.col("days_before_observation") <= window)
    if extra_condition is not None:
        cond = cond & extra_condition
    return cond


def segment_condition(segment: str) -> pl.Expr | None:
    if segment == "all":
        return None
    if segment == "secured":
        return pl.col("is_secured")
    if segment == "unsecured":
        return pl.col("is_unsecured")
    if segment == "sbi":
        return pl.col("is_sbi_lender")
    if segment == "non_sbi_or_not_disclosed_lender":
        return ~pl.col("is_sbi_lender") & pl.col("lender_std").is_not_null()
    return pl.col(f"is_{segment}")


def safe_ratio(num: str, den: str, alias: str) -> pl.Expr:
    return pl.when(pl.col(den).is_null() | (pl.col(den) == 0)).then(None).otherwise(pl.col(num) / pl.col(den)).alias(alias)


def expected_values_for_feature(name: str, family: str) -> str:
    """Describe the usual value range/type for feature documentation."""
    if name.startswith("enq_flag_"):
        return "0 or 1"
    if name.startswith("enq_trend_"):
        return "signed numeric; positive means recent activity is higher"
    if name.startswith("enq_shift_"):
        return "signed decimal, usually -1 to 1; positive means recent share is higher"
    if "_share_" in name:
        return "decimal share, usually 0 to 1; null if denominator is zero"
    if name.startswith("enq_velocity_"):
        return "decimal ratio >= 0; null if denominator is zero"
    if "days_since" in name or family == "gap":
        return "days >= 0; null if no eligible enquiry exists"
    if "_cnt_" in name or "_nunique_" in name or name.endswith("_count"):
        return "integer count >= 0"
    if "_amt_" in name or "_amount" in name:
        return "amount in source currency; null if no non-null amount exists"
    if name.startswith("enq_largest_"):
        return "decimal share, usually 0 to 1"
    if name.startswith("enq_interaction_"):
        return "numeric interaction value; validate distribution before modelling"
    return "numeric candidate value; inspect distribution before modelling"


def register_feature(name: str, family: str, formula: str, window: int | None, segment: str) -> None:
    feature_dictionary_rows.append(
        {
            "feature_name": name,
            "feature_family": family,
            "formula": formula,
            "lookback_window": win_suffix(window),
            "segment": segment,
            "expected_values": expected_values_for_feature(name, family),
            "source_columns": "custid,mrn,input_date,enq_date,enq_purpose,enq_amt,short_name,ecn",
            "missing_treatment": "Missing amounts remain null; count nulls separately; future enquiries excluded.",
            "leakage_control": "Uses only enq_date <= input_date.",
            "candidate_status": "candidate_pending_screening",
        }
    )

# %% [markdown]
# ## Cell 10 - Build Curated Feature Pool
#
# This cell avoids a blind cross-product of every metric with every segment/window.
# Instead it creates a tiered, business-prioritized feature pool:
# - Core segments get richer metrics across all windows.
# - Product/lender segments get selected high-value metrics and windows.
# - Purpose-code variables are limited to priority codes.
# - Amount-band variables are generated only for selected windows.

# %%
# Cell 10 - Curated aggregation expressions for counts, amounts, recency, segments, and purpose codes
agg_exprs: list[pl.Expr] = []

def add_count_amount_recency_features(
    segment: str,
    window: int | None,
    metrics: list[str],
    family: str,
) -> None:
    """Add selected metric expressions for one segment/window.

    This is the main control point that prevents feature explosion.
    """
    w = win_suffix(window)
    cond = window_condition(window, segment_condition(segment))
    metric_exprs = {
        "cnt": pl.when(cond).then(1).otherwise(0).sum().alias(f"enq_cnt_{segment}_{w}"),
        "amt_sum": pl.col("enq_amt").filter(cond).sum().alias(f"enq_amt_sum_{segment}_{w}"),
        "amt_mean": pl.col("enq_amt").filter(cond).mean().alias(f"enq_amt_mean_{segment}_{w}"),
        "amt_max": pl.col("enq_amt").filter(cond).max().alias(f"enq_amt_max_{segment}_{w}"),
        "days_since_last": pl.col("days_before_observation").filter(cond).min().alias(f"enq_days_since_last_{segment}_{w}"),
    }
    for metric in metrics:
        agg_exprs.append(metric_exprs[metric])
        feature_name = metric_exprs[metric].meta.output_name()
        register_feature(feature_name, family, f"{metric} for {segment} in {w}", window, segment)


# 1. Core all/secured/unsecured features receive richer metrics across all windows.
core_metrics = ["cnt", "amt_sum", "amt_mean", "amt_max", "days_since_last"]
for window in WINDOWS:
    for segment in CORE_SEGMENTS:
        add_count_amount_recency_features(segment, window, core_metrics, "core_segment")

    w = win_suffix(window)
    all_cond = window_condition(window)
    agg_exprs.extend(
        [
            pl.col("enq_date_parsed").filter(all_cond).n_unique().alias(f"enq_active_days_all_{w}"),
            pl.col("lender_std").filter(all_cond & ~pl.col("not_disclosed_lender_flag")).n_unique().alias(f"enq_lender_nunique_all_{w}"),
            pl.col("purpose_code").filter(all_cond).n_unique().alias(f"enq_purpose_nunique_all_{w}"),
            pl.col("broad_product_group").filter(all_cond).n_unique().alias(f"enq_product_group_nunique_all_{w}"),
            pl.when(all_cond & pl.col("missing_amount_flag")).then(1).otherwise(0).sum().alias(f"enq_amt_missing_cnt_all_{w}"),
            pl.when(all_cond & pl.col("zero_amount_flag")).then(1).otherwise(0).sum().alias(f"enq_amt_zero_cnt_all_{w}"),
            pl.when(all_cond & pl.col("negative_amount_flag")).then(1).otherwise(0).sum().alias(f"enq_amt_negative_cnt_all_{w}"),
            pl.when(all_cond & pl.col("same_day_enquiry_flag")).then(1).otherwise(0).sum().alias(f"enq_cnt_same_day_all_{w}"),
        ]
    )
    for feature_name in [
        f"enq_active_days_all_{w}",
        f"enq_lender_nunique_all_{w}",
        f"enq_purpose_nunique_all_{w}",
        f"enq_product_group_nunique_all_{w}",
        f"enq_amt_missing_cnt_all_{w}",
        f"enq_amt_zero_cnt_all_{w}",
        f"enq_amt_negative_cnt_all_{w}",
        f"enq_cnt_same_day_all_{w}",
    ]:
        register_feature(feature_name, "overall_extra", feature_name, window, "all")

# 2. Product segments get compact but useful count, amount, and recency metrics.
product_metrics = ["cnt", "amt_sum", "amt_max", "days_since_last"]
for window in PRODUCT_WINDOWS:
    for segment in PRODUCT_SEGMENTS:
        add_count_amount_recency_features(segment, window, product_metrics, "product_segment")

# 3. Lender segments are useful but can be unstable, so keep them compact.
lender_metrics = ["cnt", "amt_sum", "days_since_last"]
for window in LENDER_WINDOWS:
    for segment in LENDER_SEGMENTS:
        add_count_amount_recency_features(segment, window, lender_metrics, "lender_segment")

# 4. Unknown-code segment is monitoring-first; count and amount are enough.
for window in [90, 365, 730, None]:
    add_count_amount_recency_features("unknown", window, ["cnt", "amt_sum"], "unknown_purpose_monitoring")

# 5. Amount-band counts for analytical bands and selected windows only.
for window in AMOUNT_BAND_WINDOWS:
    w = win_suffix(window)
    all_cond = window_condition(window)
    for band_name, lower, upper in AMOUNT_BANDS:
        if band_name == "missing":
            band_expr = pl.col("enq_amt").is_null()
        elif band_name == "non_positive":
            band_expr = pl.col("enq_amt").is_not_null() & (pl.col("enq_amt") <= 0)
        elif upper is None:
            band_expr = pl.col("enq_amt") > lower
        else:
            band_expr = (pl.col("enq_amt") > lower) & (pl.col("enq_amt") <= upper)
        fname = f"enq_cnt_amt_band_{band_name}_{w}"
        agg_exprs.append(pl.when(all_cond & band_expr).then(1).otherwise(0).sum().alias(fname))
        register_feature(fname, "amount_band", f"count of enquiries in amount band {band_name}", window, band_name)

# 6. Purpose-code features are limited to priority codes and selected windows.
for window in PURPOSE_CODE_WINDOWS:
    w = win_suffix(window)
    for code in PRIORITY_PURPOSE_CODES:
        cond = window_condition(window, pl.col("purpose_code") == code)
        for metric_name, expr in {
            "cnt": pl.when(cond).then(1).otherwise(0).sum(),
            "amt_sum": pl.col("enq_amt").filter(cond).sum(),
            "days_since_last": pl.col("days_before_observation").filter(cond).min(),
        }.items():
            fname = f"enq_{metric_name}_code_{code}_{w}"
            agg_exprs.append(expr.alias(fname))
            register_feature(fname, "purpose_code", f"{metric_name} for purpose code {code}", window, f"code_{code}")

print("Aggregation expressions created:", len(agg_exprs))

# %% [markdown]
# ## Cell 11 - Aggregate Features To MRN-Input_Date Level

# %%
# Cell 11 - Main aggregation
feature_pool_df = eligible_events_df.group_by(KEY_COLS).agg(agg_exprs)
feature_pool_df = observation_base_df.join(feature_pool_df, on=KEY_COLS, how="left")

# Count-like features are zero when an MRN has no eligible enquiry in that segment/window.
# Amount and recency features remain null if no enquiry exists.
count_like_cols = [
    c
    for c in feature_pool_df.columns
    if c.startswith("enq_cnt_") or c.startswith("enq_active_") or "_nunique_" in c
]
feature_pool_df = feature_pool_df.with_columns([pl.col(c).fill_null(0) for c in count_like_cols])

print("Feature pool shape:", feature_pool_df.shape)
feature_pool_df.head(5)

# %% [markdown]
# ## Cell 12 - Recency Gap Features
#
# These variables use consecutive enquiry dates within each MRN-input_date observation.

# %%
# Cell 12 - Enquiry gap features
gap_base_df = (
    eligible_events_df.sort(KEY_COLS + ["enq_date_parsed"])
    .with_columns((pl.col("enq_date_parsed").diff().dt.total_days()).over(KEY_COLS).alias("gap_days"))
)

gap_features_df = gap_base_df.group_by(KEY_COLS).agg(
    [
        pl.col("gap_days").drop_nulls().mean().alias("enq_gap_mean_days_all"),
        pl.col("gap_days").drop_nulls().median().alias("enq_gap_median_days_all"),
        pl.col("gap_days").drop_nulls().min().alias("enq_gap_min_days_all"),
        pl.col("gap_days").drop_nulls().max().alias("enq_gap_max_days_all"),
        pl.col("gap_days").drop_nulls().std().alias("enq_gap_std_days_all"),
    ]
)

for fname in ["enq_gap_mean_days_all", "enq_gap_median_days_all", "enq_gap_min_days_all", "enq_gap_max_days_all", "enq_gap_std_days_all"]:
    register_feature(fname, "gap", "gap between consecutive enquiry dates", None, "all")

feature_pool_df = feature_pool_df.join(gap_features_df, on=KEY_COLS, how="left")
print("After gap features:", feature_pool_df.shape)

# %% [markdown]
# ## Cell 13 - Same-Day And Multi-Lender Shopping Features

# %%
# Cell 13 - Same-day multi-lender and burst indicators
same_day_df = (
    eligible_events_df.group_by(KEY_COLS + ["enq_date_parsed"])
    .agg(
        [
            pl.len().alias("daily_enq_count"),
            pl.col("lender_std").filter(~pl.col("not_disclosed_lender_flag")).n_unique().alias("daily_distinct_lenders"),
            pl.col("purpose_code").n_unique().alias("daily_distinct_purposes"),
        ]
    )
)

shopping_features_df = same_day_df.group_by(KEY_COLS).agg(
    [
        pl.col("daily_enq_count").max().alias("enq_max_same_day_count"),
        (pl.col("daily_enq_count") >= 2).sum().alias("enq_days_with_2plus_same_day"),
        (pl.col("daily_enq_count") >= 3).sum().alias("enq_days_with_3plus_same_day"),
        pl.col("daily_distinct_lenders").max().alias("enq_max_same_day_distinct_lenders"),
        (pl.col("daily_distinct_lenders") >= 2).sum().alias("enq_days_with_multi_lender"),
    ]
)

for fname in shopping_features_df.columns:
    if fname not in KEY_COLS:
        register_feature(fname, "same_day_shopping", "same-day or multi-lender enquiry activity", None, "all")

feature_pool_df = feature_pool_df.join(shopping_features_df, on=KEY_COLS, how="left")
feature_pool_df = feature_pool_df.with_columns(
    [
        pl.col("enq_max_same_day_count").fill_null(0),
        pl.col("enq_days_with_2plus_same_day").fill_null(0),
        pl.col("enq_days_with_3plus_same_day").fill_null(0),
        pl.col("enq_max_same_day_distinct_lenders").fill_null(0),
        pl.col("enq_days_with_multi_lender").fill_null(0),
    ]
)

print("After shopping features:", feature_pool_df.shape)

# %% [markdown]
# ## Cell 14 - Concentration Features
#
# The notebook keeps largest-share concentration because it is simple to explain:
# a value close to 1 means one purpose/lender dominates the enquiry history.
# HHI and entropy are intentionally excluded from the production candidate pool.

# %%
# Cell 14 - Largest-share concentration features
def largest_share_by_column(events: pl.DataFrame, group_col: str, prefix: str, window: int | None) -> pl.DataFrame:
    w = win_suffix(window)
    subset = events.filter(window_condition(window))
    total = subset.group_by(KEY_COLS).len().rename({"len": "_total"})
    by_group = subset.group_by(KEY_COLS + [group_col]).len().rename({"len": "_n"})
    return (
        by_group.join(total, on=KEY_COLS, how="left")
        .with_columns((pl.col("_n") / pl.col("_total")).alias("_share"))
        .group_by(KEY_COLS)
        .agg(
            [
                pl.col("_share").max().alias(f"enq_largest_{prefix}_share_{w}"),
            ]
        )
    )


concentration_df = observation_base_df.select(KEY_COLS)
for window in CONCENTRATION_WINDOWS:
    for group_col, prefix in [("purpose_code", "purpose"), ("lender_std", "lender")]:
        temp = largest_share_by_column(eligible_events_df, group_col, prefix, window)
        concentration_df = concentration_df.join(temp, on=KEY_COLS, how="left")
        for fname in temp.columns:
            if fname not in KEY_COLS:
                register_feature(fname, "concentration", f"largest category share by {prefix}", window, prefix)

feature_pool_df = feature_pool_df.join(concentration_df, on=KEY_COLS, how="left")
print("After concentration features:", feature_pool_df.shape)

# %% [markdown]
# ## Cell 15 - Ratios, Shares, Velocity, And Interactions

# %%
# Cell 15 - Post-aggregation ratios and interpretable interactions
#
# Important Polars rule:
# A column created inside one with_columns() call cannot be referenced by another
# expression in that same with_columns() call. Therefore this cell creates
# ratios/shares first, then creates interactions in a second with_columns() call.
ratio_exprs: list[pl.Expr] = []

for window in WINDOWS:
    w = win_suffix(window)
    ratio_exprs.extend(
        [
            (pl.col(f"enq_cnt_all_{w}") == 0).cast(pl.Int8).alias(f"enq_flag_no_history_{w}"),
            (pl.col(f"enq_cnt_all_{w}") > 0).cast(pl.Int8).alias(f"enq_flag_any_history_{w}"),
            (pl.col(f"enq_cnt_all_{w}") >= 2).cast(pl.Int8).alias(f"enq_flag_multiple_enquiries_{w}"),
            safe_ratio(f"enq_cnt_unsecured_{w}", f"enq_cnt_all_{w}", f"enq_share_cnt_unsecured_{w}"),
            safe_ratio(f"enq_cnt_secured_{w}", f"enq_cnt_all_{w}", f"enq_share_cnt_secured_{w}"),
            safe_ratio(f"enq_amt_sum_unsecured_{w}", f"enq_amt_sum_all_{w}", f"enq_share_amt_unsecured_{w}"),
            safe_ratio(f"enq_amt_sum_secured_{w}", f"enq_amt_sum_all_{w}", f"enq_share_amt_secured_{w}"),
            safe_ratio(f"enq_amt_missing_cnt_all_{w}", f"enq_cnt_all_{w}", f"enq_amt_missing_rate_{w}"),
        ]
    )
    if f"enq_cnt_sbi_{w}" in feature_pool_df.columns:
        ratio_exprs.append(safe_ratio(f"enq_cnt_sbi_{w}", f"enq_cnt_all_{w}", f"enq_share_cnt_sbi_{w}"))
        register_feature(f"enq_share_cnt_sbi_{w}", "ratio_share_flag", "SBI count divided by total count", window, "sbi")
    if f"enq_cnt_non_sbi_or_not_disclosed_lender_{w}" in feature_pool_df.columns:
        ratio_exprs.append(safe_ratio(f"enq_cnt_non_sbi_or_not_disclosed_lender_{w}", f"enq_cnt_all_{w}", f"enq_share_cnt_non_sbi_or_not_disclosed_lender_{w}"))
        register_feature(f"enq_share_cnt_non_sbi_or_not_disclosed_lender_{w}", "ratio_share_flag", "non-SBI or not-disclosed lender count divided by total count", window, "non_sbi_or_not_disclosed_lender")
    for segment in PRODUCT_SEGMENTS:
        if f"enq_cnt_{segment}_{w}" in feature_pool_df.columns:
            ratio_exprs.append(safe_ratio(f"enq_cnt_{segment}_{w}", f"enq_cnt_all_{w}", f"enq_share_cnt_{segment}_{w}"))
            register_feature(f"enq_share_cnt_{segment}_{w}", "ratio_share_flag", "product segment count divided by total count", window, segment)
    for fname, segment in [
        (f"enq_flag_no_history_{w}", "all"),
        (f"enq_flag_any_history_{w}", "all"),
        (f"enq_flag_multiple_enquiries_{w}", "all"),
        (f"enq_share_cnt_unsecured_{w}", "unsecured"),
        (f"enq_share_cnt_secured_{w}", "secured"),
        (f"enq_share_amt_unsecured_{w}", "unsecured"),
        (f"enq_share_amt_secured_{w}", "secured"),
        (f"enq_amt_missing_rate_{w}", "all"),
    ]:
        register_feature(fname, "ratio_share_flag", "post-aggregation ratio/share/flag", window, segment)

for short, long in [(7, 30), (15, 60), (30, 90), (90, 180), (180, 365), (365, 730)]:
    ratio_exprs.extend(
        [
            safe_ratio(f"enq_cnt_all_{short}d", f"enq_cnt_all_{long}d", f"enq_velocity_cnt_{short}d_to_{long}d"),
            safe_ratio(f"enq_amt_sum_all_{short}d", f"enq_amt_sum_all_{long}d", f"enq_velocity_amt_{short}d_to_{long}d"),
            safe_ratio(f"enq_cnt_unsecured_{short}d", f"enq_cnt_unsecured_{long}d", f"enq_velocity_unsecured_cnt_{short}d_to_{long}d"),
            safe_ratio(f"enq_cnt_secured_{short}d", f"enq_cnt_secured_{long}d", f"enq_velocity_secured_cnt_{short}d_to_{long}d"),
        ]
    )
    register_feature(f"enq_velocity_cnt_{short}d_to_{long}d", "velocity", "short window count divided by long window count", short, "all")
    register_feature(f"enq_velocity_amt_{short}d_to_{long}d", "velocity", "short window amount divided by long window amount", short, "all")
    register_feature(f"enq_velocity_unsecured_cnt_{short}d_to_{long}d", "velocity", "short unsecured count divided by long unsecured count", short, "unsecured")
    register_feature(f"enq_velocity_secured_cnt_{short}d_to_{long}d", "velocity", "short secured count divided by long secured count", short, "secured")

important_velocity_specs = [
    ("personal", "cnt", 30, 180),
    ("professional", "cnt", 30, 180),
    ("business", "cnt", 30, 180),
    ("card", "cnt", 30, 180),
    ("sbi", "cnt", 30, 180),
    ("non_sbi_or_not_disclosed_lender", "cnt", 30, 180),
    ("secured", "amt_sum", 30, 180),
    ("unsecured", "amt_sum", 30, 180),
    ("personal", "amt_sum", 30, 180),
    ("professional", "amt_sum", 30, 180),
    ("business", "amt_sum", 30, 180),
]

for segment, metric, short, long in important_velocity_specs:
    short_col = f"enq_{metric}_{segment}_{short}d"
    long_col = f"enq_{metric}_{segment}_{long}d"
    metric_label = "amt" if metric == "amt_sum" else metric
    feature_name = f"enq_velocity_{segment}_{metric_label}_{short}d_to_{long}d"
    if short_col in feature_pool_df.columns and long_col in feature_pool_df.columns:
        ratio_exprs.append(safe_ratio(short_col, long_col, feature_name))
        register_feature(feature_name, "velocity", f"important {segment} {metric_label} acceleration versus longer window", short, segment)

# First create ratios, shares, flags, and velocity variables.
feature_pool_df = feature_pool_df.with_columns(ratio_exprs)

interaction_exprs: list[pl.Expr] = []
interaction_exprs.extend(
    [
        (pl.col("enq_cnt_all_30d") - (pl.col("enq_cnt_all_60d") - pl.col("enq_cnt_all_30d"))).alias("enq_trend_cnt_0_30_vs_31_60"),
        (pl.col("enq_cnt_all_90d") - (pl.col("enq_cnt_all_180d") - pl.col("enq_cnt_all_90d"))).alias("enq_trend_cnt_0_90_vs_91_180"),
        (pl.col("enq_share_cnt_unsecured_30d") - pl.col("enq_share_cnt_unsecured_180d")).alias("enq_shift_unsecured_share_30d_vs_180d"),
        (pl.col("enq_share_cnt_sbi_30d") - pl.col("enq_share_cnt_sbi_180d")).alias("enq_shift_sbi_share_30d_vs_180d"),
        (pl.col("enq_share_cnt_non_sbi_or_not_disclosed_lender_30d") - pl.col("enq_share_cnt_non_sbi_or_not_disclosed_lender_180d")).alias("enq_shift_non_sbi_or_not_disclosed_lender_share_30d_vs_180d"),
        (pl.col("enq_cnt_all_30d") * pl.col("enq_amt_max_all_30d")).alias("enq_interaction_recent_cnt_x_max_amt_30d"),
        (pl.col("enq_cnt_professional_365d") * pl.col("enq_cnt_business_365d")).alias("enq_interaction_professional_x_business_365d"),
        (pl.col("enq_cnt_personal_180d") * pl.col("enq_cnt_card_180d")).alias("enq_interaction_personal_x_card_180d"),
        (pl.col("enq_lender_nunique_all_180d") * pl.col("enq_cnt_all_30d")).alias("enq_interaction_lender_diversity_x_recent_cnt"),
    ]
)

for segment in ["secured", "unsecured", "personal", "professional", "business", "card"]:
    short_col = f"enq_cnt_{segment}_30d"
    long_col = f"enq_cnt_{segment}_180d"
    feature_name = f"enq_trend_cnt_{segment}_0_30_vs_31_180_avg"
    if short_col in feature_pool_df.columns and long_col in feature_pool_df.columns:
        interaction_exprs.append((pl.col(short_col) - ((pl.col(long_col) - pl.col(short_col)) / 5)).alias(feature_name))
        register_feature(feature_name, "trend_or_interaction", f"recent 30d {segment} count minus average 30d count from days 31-180", None, segment)

for segment in ["secured", "personal", "professional", "business", "card", "overdraft", "education"]:
    short_col = f"enq_share_cnt_{segment}_30d"
    long_col = f"enq_share_cnt_{segment}_180d"
    feature_name = f"enq_shift_{segment}_share_30d_vs_180d"
    if short_col in feature_pool_df.columns and long_col in feature_pool_df.columns:
        interaction_exprs.append((pl.col(short_col) - pl.col(long_col)).alias(feature_name))
        register_feature(feature_name, "trend_or_interaction", f"30d {segment} count share minus 180d {segment} count share", None, segment)

for fname, segment in [
    ("enq_trend_cnt_0_30_vs_31_60", "all"),
    ("enq_trend_cnt_0_90_vs_91_180", "all"),
    ("enq_shift_unsecured_share_30d_vs_180d", "unsecured"),
    ("enq_shift_sbi_share_30d_vs_180d", "sbi"),
    ("enq_shift_non_sbi_or_not_disclosed_lender_share_30d_vs_180d", "non_sbi_or_not_disclosed_lender"),
    ("enq_interaction_recent_cnt_x_max_amt_30d", "all"),
    ("enq_interaction_professional_x_business_365d", "professional_business"),
    ("enq_interaction_personal_x_card_180d", "personal_card"),
    ("enq_interaction_lender_diversity_x_recent_cnt", "all"),
]:
    register_feature(fname, "trend_or_interaction", "interpretable trend or interaction variable", None, segment)

feature_pool_df = feature_pool_df.with_columns(interaction_exprs)
print("After ratios/interactions:", feature_pool_df.shape)

# %% [markdown]
# ## Cell 16 - Feature Quality Summary And Final 500 Selection
#
# The code removes all-null and constant features from the final selected output.
# It does not claim these are statistically significant.

# %%
# Cell 16 - Quality checks and select up to 500 features
ID_COLS = ["custid", "mrn", "input_date", "input_date_parsed", "run_date_parsed"]
candidate_cols = [c for c in feature_pool_df.columns if c not in ID_COLS]

quality_rows = []
for col in candidate_cols:
    s = feature_pool_df.get_column(col)
    null_count = s.null_count()
    n_unique = s.n_unique()
    non_null_count = feature_pool_df.height - null_count
    quality_rows.append(
        {
            "feature_name": col,
            "null_count": null_count,
            "missing_rate": null_count / feature_pool_df.height if feature_pool_df.height else None,
            "non_null_count": non_null_count,
            "n_unique": n_unique,
            "is_all_null": non_null_count == 0,
            "is_constant_or_single_value": n_unique <= 1,
            "recommended_action": "drop_all_null"
            if non_null_count == 0
            else "drop_constant"
            if n_unique <= 1
            else "retain_candidate",
        }
    )

feature_quality_df = pl.DataFrame(quality_rows)

usable_feature_names = (
    feature_quality_df.filter(~pl.col("is_all_null") & ~pl.col("is_constant_or_single_value"))
    .select("feature_name")
    .to_series()
    .to_list()
)

selected_feature_names = usable_feature_names[:MAX_FEATURES]
final_features_df = feature_pool_df.select(ID_COLS + selected_feature_names)

feature_dictionary_df = pl.DataFrame(feature_dictionary_rows).unique(subset=["feature_name"], keep="first")
feature_dictionary_df = feature_dictionary_df.filter(pl.col("feature_name").is_in(selected_feature_names))

print("Total generated feature pool columns:", len(candidate_cols))
print("Usable non-constant feature columns:", len(usable_feature_names))
print("Final selected feature columns:", len(selected_feature_names))
print("Final feature table shape:", final_features_df.shape)

if len(selected_feature_names) < MAX_FEATURES:
    print(
        "Note: fewer than MAX_FEATURES were retained because the data did not "
        "contain enough non-constant/non-null candidate variables. This is expected "
        "when only a small number of purpose codes or segments are present."
    )

feature_quality_df.head(20)

# %% [markdown]
# ## Cell 17 - Reconciliations
#
# This confirms:
# - one row per MRN-input_date key
# - no future enquiry is used
# - wider-window counts are never below narrower-window counts
# - no infinite values exist in selected features

# %%
# Cell 17 - Reconciliation checks
reconciliation_rows = []

expected_rows = observation_base_df.height
actual_rows = final_features_df.height
duplicate_key_rows = final_features_df.group_by(KEY_COLS).len().filter(pl.col("len") > 1).height

reconciliation_rows.append(
    {
        "check_name": "one_row_per_custid_mrn_input_date",
        "pass_fail": "PASS" if actual_rows == expected_rows and duplicate_key_rows == 0 else "FAIL",
        "exception_count": duplicate_key_rows,
        "expected": expected_rows,
        "actual": actual_rows,
    }
)

future_used_count = eligible_events_df.filter(pl.col("days_before_observation") < 0).height
reconciliation_rows.append(
    {
        "check_name": "no_future_enquiry_used",
        "pass_fail": "PASS" if future_used_count == 0 else "FAIL",
        "exception_count": future_used_count,
        "expected": 0,
        "actual": future_used_count,
    }
)

window_count_exceptions = 0
ordered_windows = ["7d", "15d", "30d", "60d", "90d", "180d", "365d", "730d", "all"]
for left, right in zip(ordered_windows, ordered_windows[1:]):
    left_col = f"enq_cnt_all_{left}"
    right_col = f"enq_cnt_all_{right}"
    if left_col in feature_pool_df.columns and right_col in feature_pool_df.columns:
        window_count_exceptions += feature_pool_df.filter(pl.col(left_col) > pl.col(right_col)).height

reconciliation_rows.append(
    {
        "check_name": "window_counts_non_decreasing",
        "pass_fail": "PASS" if window_count_exceptions == 0 else "FAIL",
        "exception_count": window_count_exceptions,
        "expected": 0,
        "actual": window_count_exceptions,
    }
)

infinite_count = 0
for col in selected_feature_names:
    if final_features_df.schema[col].is_numeric():
        infinite_count += int(final_features_df.select(pl.col(col).is_infinite().fill_null(False).sum()).item() or 0)

reconciliation_rows.append(
    {
        "check_name": "no_infinite_values",
        "pass_fail": "PASS" if infinite_count == 0 else "FAIL",
        "exception_count": infinite_count,
        "expected": 0,
        "actual": infinite_count,
    }
)

reconciliation_df = pl.DataFrame(reconciliation_rows)
reconciliation_df

# %% [markdown]
# ## Cell 18 - Inspect Final Features And Dictionary

# %%
# Cell 18 - Final output preview
display(final_features_df.head(5))
display(feature_dictionary_df.head(30))

# %% [markdown]
# ## Cell 19 - Export Outputs
#
# This notebook exports CSV and Parquet only. It does not create JSON files.

# %%
# Cell 19 - Save all outputs
(OUTPUT_DIR / "diagnostics").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "exceptions").mkdir(parents=True, exist_ok=True)

final_features_df.write_parquet(OUTPUT_DIR / "enquiry_500_candidate_features.parquet")
final_features_df.write_csv(OUTPUT_DIR / "enquiry_500_candidate_features.csv")
feature_dictionary_df.write_csv(OUTPUT_DIR / "enquiry_500_feature_dictionary.csv")
feature_quality_df.write_csv(OUTPUT_DIR / "enquiry_feature_quality_summary.csv")
reconciliation_df.write_csv(OUTPUT_DIR / "diagnostics" / "reconciliation_report.csv")
diagnostics_df.write_csv(OUTPUT_DIR / "diagnostics" / "input_relationship_quality_summary.csv")
purpose_validation_df.write_csv(OUTPUT_DIR / "diagnostics" / "purpose_mapping_validation.csv")

events_df.filter(pl.col("enq_after_input_date_flag")).write_parquet(OUTPUT_DIR / "exceptions" / "future_enquiries_excluded.parquet")
events_df.filter(pl.col("unknown_purpose_flag")).write_parquet(OUTPUT_DIR / "exceptions" / "unknown_purpose_codes.parquet")
events_df.filter(pl.col("exact_duplicate_flag")).write_parquet(OUTPUT_DIR / "exceptions" / "exact_duplicate_members.parquet")
events_df.filter(pl.col("probable_duplicate_flag")).write_parquet(OUTPUT_DIR / "exceptions" / "probable_duplicate_members.parquet")

print("Saved feature table:", OUTPUT_DIR / "enquiry_500_candidate_features.csv")
print("Saved feature dictionary:", OUTPUT_DIR / "enquiry_500_feature_dictionary.csv")
print("Saved feature quality summary:", OUTPUT_DIR / "enquiry_feature_quality_summary.csv")
print("Saved reconciliation:", OUTPUT_DIR / "diagnostics" / "reconciliation_report.csv")

# %% [markdown]
# ## What To Review After Running
#
# Please review these outputs before modelling:
#
# 1. `diagnostics_df`
# 2. `purpose_validation_df`
# 3. `reconciliation_df`
# 4. `feature_quality_df`
# 5. `final_features_df.shape`
# 6. `feature_dictionary_df`
#
# Remember: without target/outcome data, these are only candidate variables, not confirmed significant scorecard variables.
