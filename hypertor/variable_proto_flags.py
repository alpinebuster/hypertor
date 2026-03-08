"""Variable prototype flags."""

from absl import flags

# Prototype pushing
flags.DEFINE_integer("push_freq", 1000, "Frequency of pushing prototypes.")
flags.DEFINE_integer("push_start", 10000, "Steps to start pushing prototypes.")
flags.DEFINE_integer("push_last_steps", 2000, "Steps to train last layer only.")
flags.DEFINE_integer(
    "push_warmup_steps",
    3000, 
    "Number of steps for the warmup of prototype pushing (0 disables warmup).",
)

# NOTE: [3e-4, 1e-3]
flags.DEFINE_float("lr_last_layer", 1e-3, "Learning rate for last layer.")
# NOTE: [3e-4, 1e-3]
flags.DEFINE_float("lr_proto_layer", 3e-4, "Learning rate for prototype layer.")
# NOTE: [1e-4, 3e-5]
flags.DEFINE_float("lr_conv_layer", 5e-4, "Learning rate for convolution layer.")
flags.DEFINE_integer("joint_lr_step_size", 100, "Step size of joint learning rate in joint training.")
flags.DEFINE_float("joint_lr_gamma", 5e-1, "Allow the generator to move to a certain extent in joint training.")

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
    "If True and `use_variable_proto` is enabled, a prototype class prior is added to the hyper-generated weights of the last layer.",
)
flags.DEFINE_bool(
    "use_l_mask",
    True,
    "If True and `use_variable_proto` is enabled, apply a mask to the regularization term.",
)
flags.DEFINE_bool(
    "use_class_specific",
    True,
    "If True and `use_variable_proto` is enabled, enable class-specific processing.",
)
