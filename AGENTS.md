# Repository Guidelines

## Project Structure & Module Organization
- Core PPO code lives in `src/`: `env.py` (Gym/Gymnasium wrappers, reward shaping, frame stacking), `model.py` (PPO CNN+actor/critic heads), `process.py` (evaluation loop for asynchronous training).
- Entry points: `train.py` launches multiprocess training; `test.py` loads a saved checkpoint and renders/records; shell helpers in `scripts/` wrap common runs.
- Artifacts: saved weights in `trained_models*/`, rollouts in `output/`, TensorBoard logs in `tensorboard/`, demo GIFs in `demo/`.

## Setup & Dependency Management
- Python 3.13+; sync dependencies with `uv sync` (reads `pyproject.toml`/`uv.lock`). For pip, fall back to `pip install -r requirements.txt`.
- FFmpeg is required for video export; GPU (CUDA) support is optional but recommended.

## Build, Test, and Development Commands
- `uv run train.py --world 1 --stage 1 --lr 1e-4 --action_type jump --num_processes 1` — start training (mirrors `scripts/train.sh` defaults).
- `uv run test.py --world 1 --stage 1 --action_type simple` — evaluate a checkpoint from `trained_models/` and render/record gameplay.
- `python train.py ...` / `python test.py ...` work if the environment is already activated; keep `OMP_NUM_THREADS=1` (set in scripts) to avoid CPU contention.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indents, snake_case for functions/variables, CamelCase for classes, and import standard→third-party→local groups.
- Prefer type-safe, explicit arguments (e.g., `action_type`, `num_processes`), and keep module-level constants uppercase.
- Use concise inline comments only where logic is non-obvious (e.g., reward shaping rules); avoid cluttering training loops.

## Testing Guidelines
- There is no formal unit test suite; validate changes by running `uv run test.py` on representative worlds/stages and confirming the rendered video or on-screen play.
- For training changes, run a short training session (reduced `num_global_steps`) and ensure loss decreases and TensorBoard logs write to `tensorboard/`.

## Commit & Pull Request Guidelines
- Commit messages: imperative, present tense, focused scope (e.g., `Refine reward shaping for 7-4`, `Add Gymnasium adapter`). Group related changes together.
- Pull requests should describe the motivation, main code paths touched (`src/env.py`, `train.py`, etc.), and any hyperparameter changes. Link issues if relevant and attach brief run notes (commands used, checkpoints produced, unexpected behaviors).

## Security & Configuration Tips
- Avoid committing large checkpoints or generated videos; keep `trained_models*/` and `output/` artifacts local or use `.gitignore`.
- When running in containers or headless systems, disable rendering in `src/process.py` and ensure `ffmpeg` is available for recording-only runs.
