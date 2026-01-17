"""Datatypes, classes and functions used across `ca_supp`."""

import functools
import dataclasses
import os
import time

from typing import cast, TYPE_CHECKING, Callable, Optional, \
    TypedDict, Union
from typing_extensions import Annotated

from absl import flags, logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from hypertor.core import util
from hypertor import common_flags
import hypertor.core.layerwise as layerwise
import hypertor.variable_proto.push as push
from hypertor.variable_proto.util import list_of_distances

if TYPE_CHECKING:
    from hypertor.core import common_ht
    from hypertor.variable_proto import common_vp
    from hypertor.core import common

FLAGS = flags.FLAGS
field = dataclasses.field


@dataclasses.dataclass
class TrainConfig:
    """Training configuration."""

    train_steps: int = 10000
    steps_between_saves: int = 1000
    small_summaries_every: int = 100
    large_summaries_every: int = 500


@dataclasses.dataclass
class TrainState:
    """Model state."""
    model: Union["layerwise.LayerwiseModel", DDP]
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    global_step: int = 0

    # Variable prototype parameters
    vp_state: Optional["common_vp.VariableProtoState"] = None

    checkpoint_dir: str = "logs"
    checkpoint_suffix: str = "model"

    summary_writer: Optional[util.MultiFileWriter] = None
    small_summaries: dict[str, torch.Tensor] = field(default_factory=dict)
    large_summaries: dict[str, torch.Tensor] = field(default_factory=dict)

    def save(self):
        if FLAGS.use_ddp and dist.get_rank() != 0:
            return

        logging.info(f"Saving checkpoint at step {self.global_step}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.vp_state is not None:
            logging.info(f"With step_updated: {self.vp_state.step_updated}")
            pt_file_name = f"{self.checkpoint_suffix}-{self.global_step}-{self.vp_state.step_updated}.pt"
        else:
            pt_file_name = f"{self.checkpoint_suffix}-{self.global_step}.pt"
        path = os.path.join(
            self.checkpoint_dir,
            pt_file_name,
        )
        unwrapped_model = util.unwrap_model(self.model)
        unwrapped_model = cast("layerwise.LayerwiseModel", unwrapped_model)
        vp_data = {}
        if self.vp_state is not None:
            vp_data = {
                # Variable prototype status
                "step_updated": self.vp_state.step_updated,
                "prototype_shape": self.vp_state.prototype_shape,
                "prototype_class_identity": self.vp_state.prototype_class_identity.cpu(),
                "prototype_ids_per_class": self.vp_state.prototype_ids_per_class,
            }
        torch.save({
            # Machines without a GPU can still use `torch.load` to load it
            # No reliance on CUDA version
            "model": {
                k: v.detach().cpu()
                for k, v in unwrapped_model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            **vp_data,
        }, path)

    def load_latest(self) -> tuple[bool, str]:
        ckpt_path = util.latest_checkpoint(self.checkpoint_dir)
        if not ckpt_path:
            return False, str(ckpt_path)

        data = torch.load(ckpt_path, map_location="cpu")

        assert not isinstance(self.model, DDP)
        self.model.load_state_dict(data["model"], strict=False)
        self.optimizer.load_state_dict(data["optimizer"])
        self.global_step = int(data.get("global_step", 0))

        # Restore the states of variable prototype
        if self.vp_state is not None:
            self.vp_state.step_updated = int(data.get("step_updated", 0))
            self.vp_state.prototype_shape = tuple(data.get(
                "prototype_shape",
                (0, 0, 0, 0),
            ))
            self.vp_state.prototype_class_identity = data.get(
                "prototype_class_identity",
                torch.zeros(0, 0),
            )
            self.vp_state.prototype_ids_per_class = data.get(
                "prototype_ids_per_class",
                {},
            )

        return True, str(ckpt_path)


@dataclasses.dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    learning_rate: float = 1e-3
    learning_momentum: float = .9
    lr_decay_steps: int = 10000
    lr_decay_rate: float = 1.0


@dataclasses.dataclass
class ModelOutputs:
    """Model outputs."""

    predictions: torch.Tensor
    accuracy: torch.Tensor
    labels: Optional[torch.Tensor] = None
    test_accuracy: Optional[torch.Tensor] = None


def _make_warmup_loss(
    loss_heads: list[torch.Tensor],
    loss_prediction: torch.Tensor,
    global_step: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Uses head losses to build aggregate loss cycling through them.

    Assume there are:
       1) Multiple intermediate heads (loss_heads)
       2) One final head (loss_prediction)
       3) A global training step (global_step)

    The goal is:
       1) Early training phase: Mainly rely on shallow/early heads.
       2) Mid-training phase: Transition supervision linearly between adjacent heads.
       3) Late training phase: Only the final head remains active.

    At any moment:
       1) At most, only two heads are "active."
       2) Weights are defined by a triangular function that changes linearly.
          ▲ 1
          |
          |      /\\
          |     /  \\
          |____/____\\______ step
               center

    ---

    e.g.:
       1) Given:
          - warmup_steps = 9000
          - num_heads = 3

       2) Then:
          - steps_per_stage = warmup_steps / num_heads = 3000

       3) This means:
          - Head 0: centered at step = 0
          - Head 1: centered at step = 3000
          - Head 2: centered at step = 6000

       4) Finally, the last head starts to take over completely from step = 9000.
    """
    num_heads = len(loss_heads)
    assert FLAGS.warmup_steps > 0
    # The warmup time length corresponding to each head
    steps_per_stage = FLAGS.warmup_steps / num_heads
    loss = torch.tensor(0., device=loss_prediction.device)
    weights = []

    _torch_ref = loss_heads[0]
    global_step_t = torch.tensor(
        global_step,
        device=_torch_ref.device,
        dtype=_torch_ref.dtype,
    )
    steps_per_stage_t = torch.tensor(
        steps_per_stage,
        device=_torch_ref.device,
        dtype=_torch_ref.dtype,
    )

    # The following code ends up returning just the true model head loss
    # after `global_step` reaches `warmup_steps`.

    for stage, head_loss in enumerate(loss_heads):
        target_steps = stage * steps_per_stage_t
        norm_step_dist = torch.abs(global_step_t-target_steps) / steps_per_stage_t
        # This weight starts at 0 and peaks reaching 1 at `target_steps`. It then
        # decays linearly to 0 and stays 0.
        # weight = max(0, 1-norm_step_dist)
        weight = torch.clamp(1.-norm_step_dist, min=0.)
        weights.append(weight)
        loss += weight * head_loss

    """Weight for the actual objective linearly grows after the final layer head
    peaks and then stays equal to 1.

    1) When `global_step < target_steps-steps_per_stage`:
       - norm_step_dist<0 → relu → 0

    2) When `target_steps-steps_per_stage <= global_step <= target_steps`:
       - weight increases linearly from 0 → 1

    3) When `global_step >= target_steps`:
       - weight=1
    """
    target_steps = num_heads * steps_per_stage_t
    norm_step_dist = 1. + (global_step_t-target_steps)/steps_per_stage_t
    norm_step_dist = torch.relu(norm_step_dist)
    weight = torch.clamp(norm_step_dist, max=1.)
    weights.append(weight)
    loss += weight * loss_prediction

    return loss, weights

def _make_loss(
    labels: torch.Tensor, # [B,]
    predictions: torch.Tensor, # [B, num_classes]
    heads: list[torch.Tensor], # list[(B, num_classes)]
    global_step: int,
    label_smoothing: float = 0.,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Makes a full loss including head 'warmup' losses."""
    losses: list[torch.Tensor] = []

    for head in heads + [predictions]:
        head_loss = F.cross_entropy(
            head,
            labels,
            label_smoothing=label_smoothing,
        )
        losses.append(head_loss)

    if len(losses) == 1:
        return losses[0], losses, [torch.tensor(1.0, device=losses[0].device)]

    loss, warmup_weights = _make_warmup_loss(losses[:-1], losses[-1], global_step)
    return loss, losses, warmup_weights

def train(
    train_config: TrainConfig,
    model_config: "common_ht.LayerwiseModelConfig",
    variable_proto_config: "common_vp.ProtoTrainConfig",
    optimizer_config: "common.OptimizerConfig",
    state: TrainState,
    batch_provider: tuple[
        Callable[[], "common_ht.DatasetSamples"],
        Callable[[], "common_ht.DatasetSamples"]
    ],
) -> tuple[
    bool,
    int,
    Optional[tuple[int, int, int, int]],
    Optional[torch.Tensor],
    Optional[dict[int, list[int]]],
]:
    """Train loop.

    Args:
       train_config: Contains train_steps, steps_between_saves, small/large summary freqs
       state: TrainState (holds model/optimizer/step/etc.)
       batch_provider: Callable that returns a dict containing the inputs needed.
          Example return value for layerwise model:
            {
              "support_images": Tensor(Bs, C, H, W),
              "support_labels": Tensor(Bs,),
              "query_images": Tensor(Bq, C, H, W),
              "query_labels": Tensor(Bq,),
                ...
            }
    """
    assert state.summary_writer is not None, "Must run `common.init_training(state)` before start training!"

    local_rank = None
    global_rank = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if FLAGS.use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    state.model = state.model.to(device)
    if not FLAGS.use_ddp or global_rank == 0:
        for name, param in state.model.named_parameters():
            logging.info(f"{name}, device: {param.device}, grad: {param.grad is None}")
    state.model.dataset.to(device)
    state.optimizer, state.scheduler = util.make_optimizer(optimizer_config, state.model)

    # DDP
    if FLAGS.use_ddp:
        # torch.autograd.set_detect_anomaly(True)
        """
        When `find_unused_parameters=True`, DDP will, after each forward-backward pass:

        * Check all the model parameters to determine which ones did not receive gradients.
        * Only synchronize the parameters that have gradients, skipping those that were not used.

        This checking operation adds extra overhead for every step. Although the cost is small,
        in large models with multiple GPUs, it can accumulate and result in noticeable performance loss.
        """
        state.model = DDP(
            state.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True, # Speed up
        )
        state.optimizer, state.scheduler = util.make_optimizer(optimizer_config, state.model)

    start_step = state.global_step
    step_updated = 0
    if model_config.use_variable_proto:
        assert state.vp_state is not None
        step_updated = state.vp_state.step_updated
    update_prototype_vectors = False
    p_shape = None
    p_class_identity = None
    p_ids_per_class = None

    starting_time = time.time()
    while state.global_step <= train_config.train_steps-step_updated:
        if state.vp_state is not None:
            writer_name = f"train-{step_updated}"
        else:
            writer_name = "train"
        add_scalar = functools.partial(
            state.summary_writer.add_scalar,
            global_step=state.global_step,
            writer_name=writer_name,
        )
        add_scalar_ddp = functools.partial(
            util.add_scalar_ddp,
            add_scalar_fn=add_scalar,
        )
        add_scalars_ddp = functools.partial(
            util.add_scalars_ddp,
            add_scalar_fn=add_scalar,
        )

        train_batch = batch_provider[0]()
        test_batch = batch_provider[1]()

        train_batch.to(device)
        test_batch.to(device)

        unwrapped_model = util.unwrap_model(state.model)
        unwrapped_model = cast("layerwise.LayerwiseModel", unwrapped_model)
        state.model.train()
        total_loss = torch.tensor(0., device=device)
        if common_flags.PRETRAIN_SHARED_FEATURE.value:
            _ = state.model(
                train_batch.transformer_images,
                train_batch.transformer_labels,
                mask=train_batch.transformer_masks,
                mask_random_samples=True,
                enable_fe_dropout=True,
                only_shared_feature=True,
            )
            shared_head_loss, shared_head_acc = unwrapped_model.shared_head_outputs.values()
            assert shared_head_loss is not None and shared_head_acc is not None
            total_loss = shared_head_loss

            state.small_summaries["loss/shared_head_loss"] = shared_head_loss
            state.small_summaries["accuracy/shared_head_accuracy"] = shared_head_acc
        else:
            predictions = state.model(
                train_batch.transformer_images,
                train_batch.transformer_labels,
                train_batch.cnn_images,
                mask=train_batch.transformer_masks,
                mask_random_samples=True,
                enable_fe_dropout=True,
                deterministic_inference=False,
            )

            heads = []
            if model_config.train_heads:
                outputs = unwrapped_model.layer_outputs.values()
                # layer_outputs[layer.name] => (inputs, head)
                heads = [output[1] for output in outputs if output[1] is not None]

            labels = train_batch.cnn_labels.long()
            # Train accuracy
            pred_labels = torch.argmax(predictions, dim=-1) # (B,)

            def _acc(pred):
                accuracy = (pred == train_batch.cnn_labels).float().sum()
                return accuracy / train_batch.cnn_labels.numel()

            accuracy = _acc(pred_labels)
            head_preds = [torch.argmax(head, dim=-1) for head in heads]
            head_accs = [_acc(pred) for pred in head_preds]
            state.small_summaries["accuracy/accuracy"] = accuracy

            total_loss, _, warmup_weights = _make_loss(
                labels,
                predictions,
                heads,
                state.global_step,
            )
            add_scalar_ddp("loss/ce", total_loss)

            reg_losses = unwrapped_model.regularization_loss()
            if reg_losses:
                add_scalar_ddp("loss/regularization", reg_losses)
                total_loss += reg_losses

            shared_head_loss, shared_head_acc = unwrapped_model.shared_head_outputs.values()
            if shared_head_loss is not None and model_config.shared_head_weight > 0.:
                weighted_head_loss = shared_head_loss * model_config.shared_head_weight
                total_loss += weighted_head_loss

                add_scalar_ddp("loss/shared_head_loss", shared_head_loss)
                add_scalar_ddp("loss/weighted_shared_head_loss", weighted_head_loss)

            for head_id, acc in enumerate(head_accs):
                add_scalar_ddp(f"accuracy/head-{head_id+1}", acc)
            for head_id, warmup_weight in enumerate(warmup_weights[:-1]):
                add_scalar_ddp(f"warmup_weights/head-{head_id+1}", warmup_weight)
            if heads:
                add_scalar_ddp("warmup_weights/main", warmup_weights[-1])

            if shared_head_acc is not None and model_config.shared_head_weight > 0.:
                add_scalar_ddp("accuracy/shared_head_accuracy", shared_head_acc[-1])

            # Variable prototype
            if model_config.use_variable_proto:
                p_layer = unwrapped_model.layers[-2]
                assert isinstance(p_layer, layerwise.VariableProtoLayer)
                last_layer = unwrapped_model.layers[-1]
                assert isinstance(last_layer, layerwise.ProtoLogitsLayer)
                assert last_layer.last_linear_weight is not None
                min_distances = p_layer.min_distances
                assert min_distances is not None
                p_vectors = p_layer.used_prototype_vectors
                assert p_vectors is not None
                p_class_identity = p_layer.prototype_class_identity
                assert p_class_identity is not None
                coefs = variable_proto_config

                if model_config.use_class_specific and (
                    # NOTE: WARMUP first before using variable prototype!!!
                    state.global_step+step_updated > max(
                        5000,
                        variable_proto_config.push_start/2,
                        variable_proto_config.warm_epochs,
                    )
                ):
                    # The maximum difference per pixel is 1 (feature map normalized)
                    # There are a total of C×H×W positions
                    max_dist = (
                        p_layer.prototype_shape[1] * \
                        p_layer.prototype_shape[2] * \
                        p_layer.prototype_shape[3]
                    )

                    # Calculate cluster cost
                    # (batch_size, num_prototypes)
                    p_of_correct_class = torch.t(
                        # prototype_class_identity -> (num_prototypes, num_classes)
                        # pred_labels -> (batch_size,)
                        #       ↓
                        # (num_prototypes, batch_size)
                        p_layer.prototype_class_identity[:, train_batch.cnn_labels.cpu()]
                    ).cuda()
                    # The larger the score in the similarity style, the better,
                    # so the distance must be reversed
                    inverted_distances, _ = torch.max(
                        (max_dist-min_distances) * p_of_correct_class,
                        dim=1,
                    )
                    cluster_cost = torch.mean(max_dist - inverted_distances)
                    add_scalar_ddp("loss/cluster_cost", cluster_cost)

                    # Calculate separation cost
                    p_of_wrong_class = 1 - p_of_correct_class
                    inverted_dists_to_nontarget_p, _ = torch.max(
                        (max_dist-min_distances) * p_of_wrong_class,
                        dim=1,
                    )
                    separation_cost = torch.mean(
                        max_dist - inverted_dists_to_nontarget_p
                    )
                    add_scalar_ddp("loss/separation_cost", separation_cost)

                    # Calculate avg cluster cost
                    avg_separation_cost = torch.sum(
                        min_distances * p_of_wrong_class, dim=1
                    ) / torch.sum(p_of_wrong_class, dim=1)
                    avg_separation_cost = torch.mean(avg_separation_cost)
                    # Larger → The sample as a whole deviates from error prototypes → Good
                    # Smaller → Closer to error prototypes → Bad
                    add_scalar_ddp("loss/avg_separation_cost", avg_separation_cost)

                    # Prototypes L2 regularization
                    l21 = p_vectors.norm(p=2)
                    add_scalar_ddp("loss/l21", l21)

                    # Regularization
                    if model_config.use_l_mask:
                        # Punish the connections of the prototype to **incorrect** categories by using L2 regularization to make the cross-category connections -> 0
                        # Preserve the connections of the prototype to **correct** categories (without punishing)
                        # 
                        # p_class_identity -> [num_prototypes, num_classes] -> [num_classes, num_prototypes]
                        l2_mask = 1 - torch.t(p_class_identity).cuda()
                        # last_linear_weight -> [num_prototypes, num_classes] -> [num_classes, num_prototypes]
                        l22 = (last_layer.last_linear_weight.t() * l2_mask).norm(p=2)
                    else:
                        l22 = last_layer.last_linear_weight.t().norm(p=2)
                    add_scalar_ddp("loss/l22", l22)

                    if coefs is not None:
                        total_loss = (
                            float(coefs.coef_cross_entropy) * total_loss
                            # The smaller the `cluster_cost`, the better the intra-class clustering
                            + float(coefs.coef_cluster_cost) * cluster_cost
                            # The bigger the better
                            # `coef_separation_cost` is negative
                            + float(coefs.coef_separation_cost) * separation_cost
                            # Prevent prototype value explosion, promoting smoother and more compact prototypes;
                            # the smaller, the better.
                            + float(coefs.coef_reg_protos) * l21
                            # Penalty for the connection between prototypes and error categories
                            # Encourage prototypes to have strong connections only to their own category, while connections to other categories approach zero
                            # The smaller, the better
                            + float(coefs.coef_reg_last) * l22
                        )
                    else:
                        total_loss = (
                            total_loss
                            + 0.8 * cluster_cost
                            - 0.08 * separation_cost
                            + 1e-3 * l21
                            + 1e-4 * l22
                        )
                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)
                    l21 = p_vectors.norm(p=2)
                    l22 = last_layer.last_linear_weight.t().norm(p=2)

                    add_scalar_ddp("loss/cluster_cost", cluster_cost)
                    add_scalar_ddp("loss/l21", l21)
                    add_scalar_ddp("loss/l22", l22)
                    if coefs is not None:
                        total_loss = (
                            float(coefs.coef_cross_entropy) * total_loss
                            + float(coefs.coef_cluster_cost) * cluster_cost
                            + float(coefs.coef_reg_protos) * l21
                            + float(coefs.coef_reg_last) * l22
                        )
                    else:
                        total_loss = total_loss + 0.8 * cluster_cost + 1e-3 * l21 + 1e-4 * l22

                assert p_layer.cached_prototype_vectors is not None
                cached_p = p_layer.cached_prototype_vectors.view(p_layer.num_prototypes, -1).cpu()
                with torch.no_grad():
                    p_pair_dist = list_of_distances(cached_p, cached_p)
                    p_avg_pair_dist = p_pair_dist[
                        ~torch.eye(p_pair_dist.size(0), dtype=torch.bool)
                    ].mean()
                    # Large distances → prototypes are well separated
                    # Small distances → prototypes collapse / become redundant
                    add_scalar_ddp("loss/p_avg_pair_dist", p_avg_pair_dist)

            state.small_summaries["loss/total_loss"] = total_loss

        # Backward + Step
        state.optimizer.zero_grad()
        total_loss.backward()

        """
        # Use this ONLY when debugging the model!!!
        for name, p in unwrapped_model.named_parameters():
            if p.requires_grad and p.grad is None:
                logging.info(
                    f"{util.get_ddp_msg()}NO GRAD: {name}"
                )
            if p.device.type != "cuda":
                logging.info(
                    f"{util.get_ddp_msg()}[PARAM NOT CUDA] {name} -> {p.device}"
                )
            if p.grad is not None and p.grad.device.type != "cuda":
                logging.info(
                    f"{util.get_ddp_msg()}[GRAD NOT CUDA] {name} -> {p.grad.device}"
                )
        for _, group in enumerate(state.optimizer.param_groups):
            for p in group["params"]:
                if p not in state.optimizer.state:
                    continue
                opt_state = state.optimizer.state[p]
                for k, v in opt_state.items():
                    if torch.is_tensor(v) and v.device != p.device:
                        logging.info(
                            f"{util.get_ddp_msg()}"
                            f"[OPT STATE MISMATCH] param={p.shape} "
                            f"opt_state={k} opt_state.device={v.device} param.device={p.device}"
                        )
        """

        state.optimizer.step()
        assert state.scheduler is not None
        state.scheduler.step()

        # Push prototype
        if model_config.use_variable_proto:
            push_epochs = [
                i for i in range(train_config.train_steps)
                if i % variable_proto_config.push_freq == 0
            ]
            if (
                state.global_step+step_updated >= variable_proto_config.push_start
            ) and (
                state.global_step+step_updated in push_epochs
            ):
                assert state.vp_state is not None
                (
                    update_prototype_vectors,
                    p_shape,
                    p_class_identity,
                    p_ids_per_class,
                ) = push.push_prototypes(
                    variable_proto_config,
                    train_batch,
                    model_ddp=state.model,
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=os.path.join(state.checkpoint_dir, "prototypes"),
                    epoch_number=step_updated,
                    prototype_img_filename_prefix="prototype-img",
                    prototype_self_act_filename_prefix="prototype-self-act",
                    proto_bound_boxes_filename_prefix="bb",
                    save_prototype_class_identity=True,
                    percentile=99,
                )

                if update_prototype_vectors:
                    step_updated += state.global_step

                    assert state.vp_state is not None
                    assert p_shape is not None
                    assert p_class_identity is not None
                    assert p_ids_per_class is not None
                    state.vp_state.step_updated = step_updated
                    state.vp_state.prototype_shape = p_shape
                    state.vp_state.prototype_class_identity = p_class_identity
                    state.vp_state.prototype_ids_per_class = p_ids_per_class

                    state.save()
                    break

        # Test accuracy
        if not common_flags.PRETRAIN_SHARED_FEATURE.value:
            state.model.eval()
            with torch.no_grad():
                test_predictions = state.model(
                    test_batch.transformer_images,
                    test_batch.transformer_labels,
                    test_batch.cnn_images,
                    mask=test_batch.transformer_masks,
                    deterministic_inference=False,
                )
            test_pred_labels = torch.argmax(test_predictions, dim=-1)
            test_accuracy = (
                (test_pred_labels == test_batch.cnn_labels)
                .float()
                .sum()
                / train_batch.cnn_labels.numel()
            )
            state.small_summaries["accuracy/test_accuracy"] = test_accuracy

        if (
            (state.global_step-start_step) % 100 == 1 and
            (not FLAGS.use_ddp or global_rank == 0)
        ):
            logging.info(
                f"{util.get_ddp_msg()}"
                f"Step: {state.global_step}, "
                f"Time Per Step: {(time.time()-starting_time) / (state.global_step-start_step)}, "
                f"Push Step Updated: {step_updated}"
            )

        if (
            train_config.steps_between_saves > 0 and
            state.global_step % train_config.steps_between_saves == 0
        ):
            state.save()

        if state.global_step % train_config.small_summaries_every == 0:
            add_scalars_ddp(state.small_summaries)
        if state.global_step % train_config.large_summaries_every == 0:
            add_scalars_ddp(state.large_summaries)

        state.global_step += 1

    if not update_prototype_vectors:
        # Final save
        state.save()

    return (
        update_prototype_vectors,
        step_updated,
        p_shape,
        p_class_identity,
        p_ids_per_class,
    )


def init_training(state: TrainState, updating_vp: bool = False) -> bool:
    """Initializes the training loop by creating writer if needed
    and try restoring latest checkpoint.

    Returns:
       `True` if checkpoint was found and `False` otherwise.
    """
    if state.summary_writer is None:
        state.summary_writer = util.MultiFileWriter(state.checkpoint_dir)

    if updating_vp:
        # Skip loading model state when updating variable prototype
        return True

    restored, ckpt_path = state.load_latest()
    if not restored:
        logging.info(f"{util.get_ddp_msg()}No checkpoint found.")
    else:
        logging.info(f"{util.get_ddp_msg()}Restored from {ckpt_path}, step {state.global_step}")
    return restored
