# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_vp_training(log_dirs: list[str], save_path: str, trim_step=50000, smooth_window=12):
    tag_to_plot = 'accuracy/accuracy'

    all_data = []
    for run_path in log_dirs:
        print(f"\nrun_path: {run_path}")
        if not os.path.isdir(run_path):
            continue

        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()

        scalar_tags = ea.Tags()['scalars']
        print("Scalars tags in this run:")
        for tag in scalar_tags:
            print(tag)

        if tag_to_plot not in ea.Tags()['scalars']:
            continue

        scalar_data = ea.Scalars(tag_to_plot)
        for x in scalar_data:
            all_data.append({
                'step': x.step,
                'value': x.value,
                'run': os.path.basename(run_path),
            })

    df = pd.DataFrame(all_data)
    print(df.head())
    df = df[df['step'] <= trim_step]
    df['value_smooth'] = df.groupby('run')['value'].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())

    sns.set_theme(style='whitegrid')
    # plt.figure(figsize=(8,6))
    palette = sns.color_palette("viridis", n_colors=df['run'].nunique())
    sns.lineplot(
        data=df,
        x='step',
        y='value',
        hue='run',
        style='run',
        palette=palette,
        markers=False, dashes=False,
        linewidth=1,
        alpha=0.3, # The original curve is semi-transparent.
        legend=False, # Do not display the legend repeatedly
    )
    sns.lineplot(
        data=df,
        x='step',
        y='value_smooth',
        hue='run',
        style='run',
        palette=palette,
        markers=False, dashes=False,
        linewidth=2,
    )

    # plt.title('Accuracy across multiple runs')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend(title='Run')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# %%
log_dirs_1push = [
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simnagle-nomf-seq30-ratio1e-5/train-0',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simnagle-nomf-seq30-ratio1e-5/train-10000',
]
save_path_1push = 'vp_training_1push.png'
plot_vp_training(log_dirs_1push, save_path_1push, smooth_window=6)


# %%
log_dirs_2push = [
    './cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1/train-0',
    './cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1/train-43000',
    './cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1/train-54000',
]
save_path_2push = 'vp_training_2push.png'
plot_vp_training(log_dirs_2push, save_path_2push, 30000, smooth_window=6)


# %%
log_dirs_3push = [
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows6-seq30-ratio1e-3/train-0',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows6-seq30-ratio1e-3/train-10000',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows6-seq30-ratio1e-3/train-26000',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows6-seq30-ratio1e-3/train-54000',
]
save_path_3push = 'vp_training_3push.png'
plot_vp_training(log_dirs_3push, save_path_3push, 30000, smooth_window=6)
