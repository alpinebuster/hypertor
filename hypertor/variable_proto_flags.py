"""Variable prototype flags."""

from absl import flags

# Prototype pushing
flags.DEFINE_integer("push_freq", 500, "Frequency of pushing prototypes.")
flags.DEFINE_integer("push_start", 5000, "Epoch to start pushing prototypes.")

# Prototype parameters
flags.DEFINE_integer("prototypes", 1, "Number of prototypes per class.")
flags.DEFINE_list("prototype_size", ["1", "1"], "Prototype size as [height, width].")
flags.DEFINE_string("prototype_activation_func", "log", "Activation function for prototype layers.")

# Loss coefficients
flags.DEFINE_float("coef_cross_entropy", 1.0, "Coefficient for cross-entropy loss.")
flags.DEFINE_float("coef_cluster_cost", 0.8, "Coefficient for cluster loss.")
flags.DEFINE_float("coef_separation_cost", -0.08, "Coefficient for separation loss.")
flags.DEFINE_float("coef_reg_protos", 1e-3, "Coefficient for prototype regularization.")
flags.DEFINE_float("coef_reg_last", 1e-4, "Coefficient for last layer regularization.")

flags.DEFINE_bool(
    "use_variable_proto",
    True,
    "If set, enable support for variable protos.",
)
flags.DEFINE_bool(
    "use_prior_proto",
    True,
    "If set, a prototype–class prior is added to the hyper-generated weights of the last layer.",
)
flags.DEFINE_bool(
    "use_l_mask",
    True,
    "If True, apply a mask to the regularization term.",
)
flags.DEFINE_bool(
    "use_class_specific",
    True,
    "If True, enable class-specific processing.",
)
