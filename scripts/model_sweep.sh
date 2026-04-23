#!/usr/bin/env bash
# scripts/model_sweep.sh
#
# Run a zero-shot eval for every config listed below. Per model:
#   1. Pull the Ollama model if not already present
#   2. Run the eval (writes results/runs/<ts>_<model>/)
#   3. Refresh results/LEADERBOARD.md
#   4. Commit results + leaderboard with a descriptive message
#   5. Push to origin (unless DRY_RUN=1)
#   6. Evict the model from Ollama memory before the next one loads
#
# Usage:
#   scripts/model_sweep.sh                  # sweep every config in the default list
#   CONFIGS="configs/a.yaml configs/b.yaml" scripts/model_sweep.sh
#   DRY_RUN=1 scripts/model_sweep.sh        # run evals but don't push
#   SKIP_PULL=1 scripts/model_sweep.sh      # assume models already pulled
#
# Safe to Ctrl+C mid-sweep. Each completed model is its own commit, so
# partial progress is preserved.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

: "${CONFIGS:=}"
: "${DRY_RUN:=0}"
: "${SKIP_PULL:=0}"

# Default sweep — small models first, larger last so you see early numbers.
if [[ -z "$CONFIGS" ]]; then
    CONFIGS=$(cat <<'EOF'
configs/v0_zeroshot_phi4_mini.yaml
configs/v0_zeroshot_phi35.yaml
configs/v0_zeroshot_qwen25_7b.yaml
configs/v0_zeroshot_llama31_8b.yaml
configs/v0_zeroshot_qwen3_8b.yaml
configs/v0_zeroshot_deepseek_r1_7b.yaml
configs/v0_zeroshot_gemma3_12b.yaml
EOF
)
fi

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

# --- preflight ------------------------------------------------------------

if ! command -v ollama >/dev/null 2>&1; then
    err "ollama not on PATH. Install it and retry."
    exit 1
fi

if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null; then
    err "ollama server is not responding at localhost:11434. Start it with: ollama serve &"
    exit 1
fi

if [[ ! -f data/parsed/eval.jsonl ]]; then
    err "data/parsed/eval.jsonl missing. Run: make parse"
    exit 1
fi

# --- helpers --------------------------------------------------------------

model_from_config() {
    python - <<PY
import yaml, sys
print(yaml.safe_load(open("$1"))["llm"]["model"])
PY
}

pull_model() {
    local model="$1"
    if [[ "$SKIP_PULL" == "1" ]]; then return 0; fi
    if ollama list | awk 'NR>1 {print $1}' | grep -qx "$model"; then
        log "model already pulled: $model"
    else
        log "pulling $model (this can be several GB)"
        ollama pull "$model"
    fi
}

evict_model() {
    local model="$1"
    # best-effort — free VRAM before loading the next model
    ollama stop "$model" >/dev/null 2>&1 || true
}

refresh_leaderboard() {
    python -m ribo_agent.eval.compare --markdown > results/LEADERBOARD.md
    python -m ribo_agent.eval.compare --readme >/dev/null
}

commit_and_push() {
    local model="$1"
    local accuracy="$2"
    git add results/runs results/LEADERBOARD.md README.md
    if git diff --cached --quiet; then
        warn "nothing to commit for $model"
        return 0
    fi
    git commit -q -m "eval: ${model} zero-shot, accuracy=${accuracy}"
    if [[ "$DRY_RUN" == "1" ]]; then
        warn "DRY_RUN=1, not pushing"
    else
        git push origin HEAD >/dev/null 2>&1 || warn "push failed (will retry on next model)"
    fi
}

# --- run ------------------------------------------------------------------

mkdir -p results
START_TS=$(date +%s)

# strip empty lines / comments from $CONFIGS
CONFIG_LIST=$(echo "$CONFIGS" | sed '/^$/d' | sed '/^[[:space:]]*#/d')

N=$(echo "$CONFIG_LIST" | wc -l | tr -d ' ')
i=0
for cfg in $CONFIG_LIST; do
    i=$((i + 1))
    log "[$i/$N] $cfg"

    if [[ ! -f "$cfg" ]]; then
        warn "config missing: $cfg — skipping"
        continue
    fi

    model=$(model_from_config "$cfg")
    log "model: $model"

    # 1. ensure the model is pulled
    if ! pull_model "$model"; then
        warn "pull failed for $model — skipping"
        continue
    fi

    # 2. run eval
    set +e
    python -m ribo_agent.eval.runner --config "$cfg"
    status=$?
    set -e
    if [[ $status -ne 0 ]]; then
        warn "eval failed for $model (exit $status) — skipping push"
        evict_model "$model"
        continue
    fi

    # 3. find the run we just wrote (newest under results/runs/)
    run_dir=$(ls -1td results/runs/*/ 2>/dev/null | head -1)
    accuracy=$(python - <<PY
import json, sys
p = "${run_dir}metrics.json"
try:
    print(f"{json.load(open(p))['accuracy']:.4f}")
except Exception:
    print("n/a")
PY
)
    log "accuracy=$accuracy  (results at $run_dir)"

    # 4. leaderboard + commit + push
    refresh_leaderboard
    commit_and_push "$model" "$accuracy"

    # 5. free memory for the next one
    evict_model "$model"
done

ELAPSED=$(( $(date +%s) - START_TS ))
log "sweep complete in ${ELAPSED}s"
log "leaderboard:"
cat results/LEADERBOARD.md
