#!/usr/bin/env bash
set -euo pipefail

# --- Inputs (override via CLI or env) ----------------------------------------
BASE_CONF="${1:-conf/speed_benchmark.yaml}"     # base config to start from
SIM_BIN="${SIM_BIN:-./examples/run_and_tumble/run_and_tumble}"
OUT_DIR="${OUT_DIR:-results/speed_benchmark}"
TMP_DIR="${TMP_DIR:-tmp/speed_bench}"
RUNS_PER_CASE=5                                  # pogobatch -r
BACKEND="sequential"                             # required by spec

# --- Robot sizes to sweep ----------------------------------------------------
ROBOT_COUNTS=(1 5 25 50 100 250 500 1000)

# --- Conditions --------------------------------------------------------------
# name                  GUI?  DATA/VIDEO PERIOD (1 = every 1s; -1 = disabled)
COND_NAMES=("noGUI_noExport" "noGUI_export1s" "GUI_noExport" "GUI_export1s")
COND_GUI=(0 0 1 1)
COND_PERIOD=( -1 1 -1 1 )

# --- Prep --------------------------------------------------------------------
mkdir -p "$TMP_DIR" "$OUT_DIR"
CSV="$OUT_DIR/speed_results.csv"
echo "condition,robots,real,user,sys" > "$CSV"

have_yq=0
if command -v yq >/dev/null 2>&1; then
  have_yq=1
fi

have_pyyaml=0
if python3 - <<'PY' >/dev/null 2>&1
try:
    import yaml  # type: ignore
except Exception:
    raise SystemExit(1)
PY
then
  have_pyyaml=1
fi

modify_yaml() {
  # Args: input_yaml output_yaml nrobots period
  local in="$1" out="$2" nrobots="$3" period="$4"

  if [ "$have_yq" -eq 1 ]; then
    yq eval \
      ".save_data_period = ${period} |
       .save_video_period = ${period} |
       .object.robots.nb = ${nrobots}" \
      "$in" > "$out"
    return
  fi

  if [ "$have_pyyaml" -eq 1 ]; then
    DATA_PERIOD="$period" VIDEO_PERIOD="$period" NROBOTS="$nrobots" \
    python3 - "$in" "$out" <<'PY'
import os, sys, yaml
inp, outp = sys.argv[1], sys.argv[2]
data = yaml.safe_load(open(inp))
def set_path(d, keys, val):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = val
set_path(data, ["save_data_period"], int(os.environ["DATA_PERIOD"]))
set_path(data, ["save_video_period"], int(os.environ["VIDEO_PERIOD"]))
set_path(data, ["object","robots","nb"], int(os.environ["NROBOTS"]))
with open(outp, "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
    return
  fi

  echo "ERROR: Need either 'yq' or Python with PyYAML installed to edit YAML." >&2
  exit 1
}

run_case() {
  local conf="$1" gui="$2"

  local timefile
  timefile="$(mktemp "$TMP_DIR/time.XXXX.txt")"

  if [ "$gui" -eq 1 ]; then
    # GUI enabled -> add --gui (pogobatch will omit '-g')
    { /usr/bin/time -p pogobatch --backend "$BACKEND" -c "$conf" -S "$SIM_BIN" \
        -r "$RUNS_PER_CASE" -t tmp -o "$OUT_DIR" --gui; } 2> "$timefile"
  else
    { /usr/bin/time -p pogobatch --backend "$BACKEND" -c "$conf" -S "$SIM_BIN" \
        -r "$RUNS_PER_CASE" -t tmp -o "$OUT_DIR"; } 2> "$timefile"
  fi

  awk '
    BEGIN{real="";user="";sys=""}
    /^real/ {real=$2}
    /^user/ {user=$2}
    /^sys/  {sys=$2}
    END{printf "%s,%s,%s\n", real, user, sys}
  ' "$timefile"
  rm -f "$timefile"
}

# --- Sweep -------------------------------------------------------------------
for idx in "${!COND_NAMES[@]}"; do
  cname="${COND_NAMES[$idx]}"
  gui="${COND_GUI[$idx]}"
  period="${COND_PERIOD[$idx]}"

  echo "===== Condition: $cname (GUI=$gui, period=$period) ====="

  for n in "${ROBOT_COUNTS[@]}"; do
    tmp_conf="$TMP_DIR/${cname}_n${n}.yaml"
    modify_yaml "$BASE_CONF" "$tmp_conf" "$n" "$period"

    echo "→ Robots=$n | Config=$tmp_conf"
    # Optional: ensure we don't append to stale feather from a previous unrelated run
    # (pogobatch typically manages this, but we keep it explicit)
    # rm -f "$OUT_DIR/result.feather" || true

    # Execute and capture times
    IFS=',' read -r real user sys < <(run_case "$tmp_conf" "$gui")
    echo "$cname,$n,$real,$user,$sys" | tee -a "$CSV"
  done
done

echo
echo "Benchmark complete. Results saved to: $CSV"
