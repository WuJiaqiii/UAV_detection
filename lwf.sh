export LWF_WORK="$PWD/experiment/lwf_supplement_20261108_s42"

python lwf_supplement.py prepare \
  --repo "$PWD" \
  --source-experiment "$PWD/experiment/all_known_training_20261107_113635" \
  --work "$LWF_WORK" \
  --seed 42

python lwf_supplement.py check --work "$LWF_WORK"

python -u lwf_supplement.py run --work "$LWF_WORK"