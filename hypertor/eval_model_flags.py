"""Hyper-Tor flags."""

from absl import flags

# `FLAGS.samples_cnn = FLAGS.eval_batch_size` in `evaluate.py`
flags.DEFINE_integer("eval_batch_size", 200, "Number of samples when evaluating.")
flags.DEFINE_integer(
    "num_eval_batches", 16, "Number of batches to evaluate for the same task."
)
flags.DEFINE_integer(
    "num_task_evals", 512, 'Number of different "tasks" to evaluate.' 
)
flags.DEFINE_string("eval_datasets", "", "List of datasets to evaluate.")
