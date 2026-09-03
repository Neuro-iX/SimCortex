#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

EVAL_ROOT=""
SAMPLE_ROOT=""
PRED_ROOT=""

DEVICE="auto"
EVAL_SET="sample40"

EXPECTED_CASES=560
EXPECTED_CASES_PER_DATASET=40
EXPECTED_N_DATASETS=14
EXPECTED_SURFACES_PER_CASE=4

EXPECTED_DATASETS=(
    cnp
    ds000115
    ds000144
    ds001486
    ds001748
    ds002424
    ds002862
    ds002886
    ds003499
    ds003568
    ds003763
    ds005234
    ds006067
    hcp_oasis
)

OVERWRITE=0
STRICT=0
DRY_RUN=0


usage() {
    cat <<'EOF'
Usage:
  evaluation/run_evaluation.sh \
    --eval-root PATH \
    --sample-root PATH \
    --pred-root PATH \
    [options]

Required:
  --eval-root PATH
      Evaluation output root.

  --sample-root PATH
      Root containing the evaluation FreeSurfer dataset folders.

  --pred-root PATH
      Root containing SimCortex predictions.

Options:
  --device DEVICE
      Device passed to metric evaluation.
      Default: auto

  --eval-set NAME
      Metadata label written to metric/collision outputs.
      Default: sample40

  --expected-cases N
      Expected total number of cases.
      Default: 560

  --expected-cases-per-dataset N
      Expected cases per dataset.
      Default: 40

  --expected-n-datasets N
      Expected number of datasets.
      Default: 14

  --expected-datasets NAME [NAME ...]
      Expected dataset keys.
      Default: the historical 14-dataset sample40 cohort.

  --overwrite
      Allow stages that protect outputs to replace existing results.

  --strict
      Enable strict validation where supported.

  --dry-run
      Print commands without executing them.

  -h, --help
      Show this help.
EOF
}


die() {
    echo "[ERROR] $*" >&2
    exit 2
}


run_cmd() {
    printf '\n[RUN]'
    printf ' %q' "$@"
    printf '\n'

    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "$@"
    fi
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval-root)
            [[ $# -ge 2 ]] || die "--eval-root requires a value"
            EVAL_ROOT="$2"
            shift 2
            ;;
        --sample-root)
            [[ $# -ge 2 ]] || die "--sample-root requires a value"
            SAMPLE_ROOT="$2"
            shift 2
            ;;
        --pred-root)
            [[ $# -ge 2 ]] || die "--pred-root requires a value"
            PRED_ROOT="$2"
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || die "--device requires a value"
            DEVICE="$2"
            shift 2
            ;;
        --eval-set)
            [[ $# -ge 2 ]] || die "--eval-set requires a value"
            EVAL_SET="$2"
            shift 2
            ;;
        --expected-cases)
            [[ $# -ge 2 ]] || die "--expected-cases requires a value"
            EXPECTED_CASES="$2"
            shift 2
            ;;
        --expected-cases-per-dataset)
            [[ $# -ge 2 ]] || die "--expected-cases-per-dataset requires a value"
            EXPECTED_CASES_PER_DATASET="$2"
            shift 2
            ;;
        --expected-n-datasets)
            [[ $# -ge 2 ]] || die "--expected-n-datasets requires a value"
            EXPECTED_N_DATASETS="$2"
            shift 2
            ;;
        --expected-datasets)
            shift
            EXPECTED_DATASETS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXPECTED_DATASETS+=("$1")
                shift
            done
            [[ "${#EXPECTED_DATASETS[@]}" -gt 0 ]]                 || die "--expected-datasets requires at least one dataset name"
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done


[[ -n "${EVAL_ROOT}" ]] || die "--eval-root is required"
[[ -n "${SAMPLE_ROOT}" ]] || die "--sample-root is required"
[[ -n "${PRED_ROOT}" ]] || die "--pred-root is required"

[[ "${EXPECTED_CASES}" =~ ^[1-9][0-9]*$ ]] \
    || die "--expected-cases must be a positive integer"

[[ "${EXPECTED_CASES_PER_DATASET}" =~ ^[1-9][0-9]*$ ]] \
    || die "--expected-cases-per-dataset must be a positive integer"

[[ "${EXPECTED_N_DATASETS}" =~ ^[1-9][0-9]*$ ]] \
    || die "--expected-n-datasets must be a positive integer"

[[ "${#EXPECTED_DATASETS[@]}" -eq "${EXPECTED_N_DATASETS}" ]] \
    || die "--expected-n-datasets (${EXPECTED_N_DATASETS}) must match the number of --expected-datasets entries (${#EXPECTED_DATASETS[@]})"


echo "=== SimCortex evaluation ==="
echo "repo_root=${REPO_ROOT}"
echo "eval_root=${EVAL_ROOT}"
echo "sample_root=${SAMPLE_ROOT}"
echo "pred_root=${PRED_ROOT}"
echo "device=${DEVICE}"
echo "eval_set=${EVAL_SET}"
echo "expected_cases=${EXPECTED_CASES}"
echo "expected_cases_per_dataset=${EXPECTED_CASES_PER_DATASET}"
echo "expected_n_datasets=${EXPECTED_N_DATASETS}"
echo "expected_datasets=${EXPECTED_DATASETS[*]}"
echo "overwrite=${OVERWRITE}"
echo "strict=${STRICT}"
echo "dry_run=${DRY_RUN}"


# ---------------------------------------------------------------------------
# 1. Canonical evaluation cohort
# ---------------------------------------------------------------------------

cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/build_case_manifest.py"
    --eval-root "${EVAL_ROOT}"
    --sample-root "${SAMPLE_ROOT}"
    --expected-datasets "${EXPECTED_DATASETS[@]}"
    --expected-cases-per-dataset "${EXPECTED_CASES_PER_DATASET}"
    --expected-total-cases "${EXPECTED_CASES}"
)

if [[ "${OVERWRITE}" -eq 1 ]]; then
    cmd+=(--overwrite)
fi

if [[ "${STRICT}" -eq 1 ]]; then
    cmd+=(--strict)
fi

run_cmd "${cmd[@]}"


# ---------------------------------------------------------------------------
# 2. FreeSurfer GT: tkRAS -> scannerRAS
# ---------------------------------------------------------------------------

run_cmd \
    "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/build_gt_manifest.py" \
    --eval-root "${EVAL_ROOT}" \
    --expected-cases "${EXPECTED_CASES}" \
    --expected-cases-per-dataset "${EXPECTED_CASES_PER_DATASET}" \
    --expected-surfaces-per-case "${EXPECTED_SURFACES_PER_CASE}"


# ---------------------------------------------------------------------------
# 3. SimCortex prediction manifest
# ---------------------------------------------------------------------------

cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/build_pred_manifest.py"
    --eval-root "${EVAL_ROOT}"
    --pred-root "${PRED_ROOT}"
    --expected-cases "${EXPECTED_CASES}"
    --expected-cases-per-dataset "${EXPECTED_CASES_PER_DATASET}"
    --expected-surfaces-per-case "${EXPECTED_SURFACES_PER_CASE}"
)

if [[ "${OVERWRITE}" -eq 1 ]]; then
    cmd+=(--overwrite)
fi

if [[ "${STRICT}" -eq 1 ]]; then
    cmd+=(--strict)
fi

run_cmd "${cmd[@]}"


# ---------------------------------------------------------------------------
# 4. Prediction coordinate / geometry audit
# ---------------------------------------------------------------------------

run_cmd \
    "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/audit_predictions.py" \
    --eval-root "${EVAL_ROOT}" \
    --expected-n-datasets "${EXPECTED_N_DATASETS}" \
    --expected-cases-per-dataset "${EXPECTED_CASES_PER_DATASET}" \
    --expected-surfaces-per-case "${EXPECTED_SURFACES_PER_CASE}"


# ---------------------------------------------------------------------------
# 5. Reconstruction metrics
# ---------------------------------------------------------------------------

cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/evaluate_metrics.py"
    --eval-root "${EVAL_ROOT}"
    --device "${DEVICE}"
    --eval-set "${EVAL_SET}"
)

if [[ "${OVERWRITE}" -eq 1 ]]; then
    cmd+=(--overwrite)
fi

if [[ "${STRICT}" -eq 1 ]]; then
    cmd+=(--strict)
fi

run_cmd "${cmd[@]}"


# ---------------------------------------------------------------------------
# 6. Exact FCL collision evaluation
# ---------------------------------------------------------------------------

cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/evaluate_collisions.py"
    --eval-root "${EVAL_ROOT}"
    --eval-set "${EVAL_SET}"
)

if [[ "${OVERWRITE}" -eq 1 ]]; then
    cmd+=(--overwrite)
fi

if [[ "${STRICT}" -eq 1 ]]; then
    cmd+=(--strict)
fi

run_cmd "${cmd[@]}"


# ---------------------------------------------------------------------------
# 7. Final aggregation
# ---------------------------------------------------------------------------

cmd=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/summarize_results.py"
    --eval-root "${EVAL_ROOT}"
    --expected-cases "${EXPECTED_CASES}"
    --expected-cases-per-dataset "${EXPECTED_CASES_PER_DATASET}"
)

if [[ "${OVERWRITE}" -eq 1 ]]; then
    cmd+=(--overwrite)
fi

if [[ "${STRICT}" -eq 1 ]]; then
    cmd+=(--strict)
fi

run_cmd "${cmd[@]}"


echo
echo "=== SimCortex evaluation complete ==="
echo "Results:"
echo "  ${EVAL_ROOT}/metrics"
echo "  ${EVAL_ROOT}/collisions"
echo "  ${EVAL_ROOT}/summary"
