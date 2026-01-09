# AgentGym VL dataset utilities

This folder contains small utilities for preparing/validating AgentGym vision-language conversation datasets.

## `split_vl_conversations.py`

Splits `processed_conversations_vl_*.json` (a JSON list of multi-round conversations) into per-round prefix samples
and decodes `screenshot_base64` images to files, producing a JSONL with image-path references.

## `clean_vl_conversations_actions.py`

Validates and (optionally) repairs the assistant `Action:` lines in `processed_conversations_vl_*.json`.

Typical usage (safest: drop bad targets from loss):

```bash
python AgentGym2/AgentGym/agentenv/scripts/clean_vl_conversations_actions.py \
  --input AgentGym2/AgentGym/data/processed_conversations_vl_validation.json \
  --output AgentGym2/AgentGym/data/processed_conversations_vl_validation.cleaned.json \
  --mode mark_loss_false
```

