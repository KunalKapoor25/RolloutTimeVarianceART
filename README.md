# Rollout Time Variance Experiment

This experiment demonstrates variance in rollout times during training. Tested on an A100 GPU.

## Setup

1. **Install dependencies**
   ```bash
   uv pip install openpipe-art[backend]
   ```

2. **Configure reward model key**
   Set an appropriate key for the reward model (very cheap although likely not central to the issue).

3. **Verify logging directory**
   Ensure the path to the logging directory is correct.

4. **Run the experiment**
   ```bash
   uv run python example.py
   ```

## Expected Results

After a few batches, you should observe significant variance in rollout times between different batches in training_times.log.

**Example observed variance:**
- Batch 1: ~85 seconds average rollout time
- Batch 2: ~1050 seconds average rollout time