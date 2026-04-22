#!/usr/bin/env bash
# improve-loop.sh - Iterative improvement loop for the reproduction pipeline
#
# This script:
# 1. Runs the pipeline against a target paper
# 2. Evaluates the result against PaperBench rubric
# 3. Analyzes failures
# 4. Suggests improvements (using Claude)
# 5. Applies improvements to the generated code
# 6. Re-evaluates and measures if score improved
# 7. Logs everything
#
# Usage:
#   cd ~/dev/autores/reproduce
#   export ANTHROPIC_API_KEY=...
#   bash scripts/improve-loop.sh [--max-iterations 5] [--target-score 60]

set -euo pipefail

# Configuration
AUTORES_ROOT="${AUTORES_ROOT:-$HOME/dev/autores}"
REPRODUCE_DIR="$AUTORES_ROOT/reproduce"
RESULTS_DIR="$AUTORES_ROOT/results"
PAPERBENCH_DIR="$AUTORES_ROOT/paperbench-data/project/paperbench"
LOGS_DIR="$AUTORES_ROOT/docs/improvement-logs"

# Default target: Stochastic Interpolants
PAPER_ID="${PAPER_ID:-stochastic-interpolants}"
ARXIV_URL="${ARXIV_URL:-https://arxiv.org/abs/2310.03725}"
RUBRIC_PATH="$PAPERBENCH_DIR/data/papers/$PAPER_ID/rubric.json"

# Parameters
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
TARGET_SCORE="${TARGET_SCORE:-60}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-4-20250514}"
IMPROVE_MODEL="${IMPROVE_MODEL:-claude-sonnet-4-20250514}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        --target-score) TARGET_SCORE="$2"; shift 2 ;;
        --paper-id) PAPER_ID="$2"; shift 2 ;;
        --arxiv-url) ARXIV_URL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Setup
mkdir -p "$LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$RESULTS_DIR/$PAPER_ID/improvement-$TIMESTAMP"
LOG_FILE="$LOGS_DIR/$PAPER_ID-$TIMESTAMP.log"

log() {
    echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

log "======================================"
log "IMPROVEMENT LOOP START"
log "======================================"
log "Paper: $PAPER_ID"
log "arXiv: $ARXIV_URL"
log "Max iterations: $MAX_ITERATIONS"
log "Target score: $TARGET_SCORE%"
log "Run directory: $RUN_DIR"
log ""

# Step 0: Initial pipeline run (if no previous results exist)
ITER_DIR="$RUN_DIR/iter-0"
mkdir -p "$ITER_DIR"

if [[ -d "$RESULTS_DIR/$PAPER_ID/code/generated" ]]; then
    log "Using existing pipeline output from previous run"
    cp -r "$RESULTS_DIR/$PAPER_ID/code/generated" "$ITER_DIR/code"
else
    log "Running initial pipeline..."
    cd "$REPRODUCE_DIR"
    uv run reproduce "$ARXIV_URL" \
        --output-dir "$ITER_DIR/pipeline-output" \
        --no-gpu --verbose 2>&1 | tee -a "$LOG_FILE"
    cp -r "$ITER_DIR/pipeline-output/code/generated" "$ITER_DIR/code"
fi

# Step 1: Initial evaluation
log ""
log "=== ITERATION 0: Initial Evaluation ==="
cd "$REPRODUCE_DIR"
uv run python scripts/evaluate_paperbench.py \
    --submission "$ITER_DIR/code" \
    --rubric "$RUBRIC_PATH" \
    --mode code-dev \
    --output "$ITER_DIR/evaluation.json" \
    --model "$JUDGE_MODEL" 2>&1 | tee -a "$LOG_FILE"

CURRENT_SCORE=$(python3 -c "import json; print(json.load(open('$ITER_DIR/evaluation.json'))['hierarchical_score'])")
log "Initial score: $CURRENT_SCORE%"

# Check if already at target
if (( $(echo "$CURRENT_SCORE >= $TARGET_SCORE" | bc -l) )); then
    log "Already at target score! Done."
    exit 0
fi

# Improvement iterations
for ((iter=1; iter<=MAX_ITERATIONS; iter++)); do
    log ""
    log "======================================"
    log "ITERATION $iter / $MAX_ITERATIONS"
    log "======================================"
    log "Current score: $CURRENT_SCORE%"

    PREV_DIR="$RUN_DIR/iter-$((iter-1))"
    ITER_DIR="$RUN_DIR/iter-$iter"
    mkdir -p "$ITER_DIR"

    # Step 3: Analyze failures and suggest improvements
    log "Analyzing failures and generating improvements..."

    uv run python scripts/improve_code.py \
        --submission "$PREV_DIR/code" \
        --evaluation "$PREV_DIR/evaluation.json" \
        --rubric "$RUBRIC_PATH" \
        --paper-dir "$PAPERBENCH_DIR/data/papers/$PAPER_ID" \
        --output-dir "$ITER_DIR/code" \
        --model "$IMPROVE_MODEL" 2>&1 | tee -a "$LOG_FILE"

    # Step 4: Re-evaluate
    log "Re-evaluating improved code..."
    uv run python scripts/evaluate_paperbench.py \
        --submission "$ITER_DIR/code" \
        --rubric "$RUBRIC_PATH" \
        --mode code-dev \
        --output "$ITER_DIR/evaluation.json" \
        --model "$JUDGE_MODEL" 2>&1 | tee -a "$LOG_FILE"

    NEW_SCORE=$(python3 -c "import json; print(json.load(open('$ITER_DIR/evaluation.json'))['hierarchical_score'])")
    DELTA=$(python3 -c "print(round($NEW_SCORE - $CURRENT_SCORE, 2))")

    log ""
    log "Score: $CURRENT_SCORE% -> $NEW_SCORE% (delta: ${DELTA}%)"

    # Update score
    CURRENT_SCORE="$NEW_SCORE"

    # Log to score-log
    echo "" >> "$AUTORES_ROOT/docs/score-log.md"
    echo "### Iteration $iter ($TIMESTAMP)" >> "$AUTORES_ROOT/docs/score-log.md"
    echo "- Score: $NEW_SCORE% (delta: ${DELTA}%)" >> "$AUTORES_ROOT/docs/score-log.md"

    # Check target
    if (( $(echo "$CURRENT_SCORE >= $TARGET_SCORE" | bc -l) )); then
        log "TARGET REACHED! Score: $CURRENT_SCORE% >= $TARGET_SCORE%"
        break
    fi

    # Check if improvement stalled
    if (( $(echo "$DELTA <= 0" | bc -l) )); then
        log "WARNING: No improvement in this iteration (delta=$DELTA)"
    fi
done

# Final summary
log ""
log "======================================"
log "IMPROVEMENT LOOP COMPLETE"
log "======================================"
log "Final score: $CURRENT_SCORE%"
log "Target: $TARGET_SCORE%"
log "Iterations run: $iter"
log "Log: $LOG_FILE"
log "Results: $RUN_DIR"
