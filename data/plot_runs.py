# %%
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# %%
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


# %%
def plot_usevp(
    log_dirs: list[str],
    save_path: str,
    trim_step=50000,
    smooth_window=12,
):
    tag_to_plot = 'accuracy/accuracy'

    all_data = []
    for idx, run_path in enumerate(log_dirs):
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
                'run': 'novp' if idx==0 else 'usevp',
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
log_dirs_vp_cic = [
    './cicdarknet/logs/1-exps_other_methods/cicdarknet-baseline-novp-noaug-nomf-seq30-ratio1e-3/train',
    './cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1e-3/train-8000',
]
plot_usevp(log_dirs_vp_cic, 'cic-usevp_ratio1e-3.png', 30000, smooth_window=6)

# %%
log_dirs_vp_cic = [
    './cicdarknet/logs/1-exps_other_methods/cicdarknet-baseline-novp-noaug-nomf-seq30-ratio1e-5/train',
    './cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1e-5/train-32000',
]
plot_usevp(log_dirs_vp_cic, 'cic-usevp_ratio1e-5.png', 40000, smooth_window=6)

# %%
log_dirs_vp_tcub = [
    './tcub/logs/1-exps_other_methods/tcub-baseline-novp-noaug-nomf-seq30-ratio1e-3/train',
    './tcub/logs/3-exps_variable_prototypes/tcub-usevp-noaug-nomf-seq30-ratio1e-3/train-55000',
]
plot_usevp(log_dirs_vp_tcub, 'tcub-usevp_ratio1e-3.png', 35000, smooth_window=6)


# %%
log_dirs_vp_tcub = [
    './tcub/logs/1-exps_other_methods/tcub-baseline-novp-noaug-nomf-seq30-ratio1e-5/train',
    './tcub/logs/3-exps_variable_prototypes/tcub-usevp-noaug-nomf-seq30-ratio1e-5/train-35000',
]
plot_usevp(log_dirs_vp_tcub, 'tcub-usevp_ratio1e-5.png', 35000, smooth_window=6)

# %%
log_dirs_vp_safesurf = [
    './safesurf2025/logs/1-exps_other_methods/safesurf2025-baseline-novp-noaug-nomf-seq30-ratio1e-3/train',
    './safesurf2025/logs/3-exps_variable_prototypes/safesurf2025-usevp-noaug-nomf-seq30-ratio1e-3/train-0',
]
plot_usevp(log_dirs_vp_safesurf, 'safesurf2025-usevp_ratio1e-3.png', 35000, smooth_window=6)


# %%
log_dirs_vp_safesurf = [
    './safesurf2025/logs/1-exps_other_methods/safesurf2025-baseline-novp-noaug-nomf-seq30-ratio1e-5/train',
    './safesurf2025/logs/3-exps_variable_prototypes/safesurf2025-usevp-noaug-nomf-seq30-ratio1e-5/train-55000',
]
plot_usevp(log_dirs_vp_safesurf, 'safesurf2025-usevp_ratio1e-5.png', 35000, smooth_window=6)


# %%
log_dirs_vp_safesurf = [
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-novp-simnagle-nomf-seq30-ratio1e-3/train',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simnagle-nomf-seq30-ratio1e-3/train-42000',
]
plot_usevp(log_dirs_vp_safesurf, 'safesurf2025-usevp_simnagle_ratio1e-3.png', 27000, smooth_window=6)

# %%
log_dirs_vp_safesurf = [
    './safesurf2025/logs/3-exps_variable_prototypes/safesurf2025-novp-simnagle-flows3-seq30-ratio1e-3/train',
    './safesurf2025/logs/4-exps_multi_flows/safesurf2025-usevp-simnagle-flows3-seq30-ratio1e-3/train-0',
]
plot_usevp(log_dirs_vp_safesurf, 'safesurf2025-usevp_simnagle_flows3_ratio1e-3.png', 28000, smooth_window=6)

# %%
log_dirs_vp_safesurf = [
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-novp-simnagle-nomf-seq30-ratio1e-5/train',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simnagle-nomf-seq30-ratio1e-5/train-8000',
]
plot_usevp(log_dirs_vp_safesurf, 'safesurf2025-usevp_simnagle_ratio1e-5.png', 35000, smooth_window=6)


# %%
def plot_usemf(
    log_dirs: list[str],
    save_path: str,
    trim_step=50000,
    smooth_window=12,
):
    tag_to_plot = 'accuracy/accuracy'

    all_data = []
    for idx, run_path in enumerate(log_dirs):
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
        if idx == 0:
            run_name = "1 flow"
        elif idx == 1:
            run_name = "3 flows"
        elif idx == 2:
            run_name = "6 flows"
        else:
            run_name = "9 flows"
        for x in scalar_data:
            all_data.append({
                'step': x.step,
                'value': x.value,
                'run': run_name,
            })

    df = pd.DataFrame(all_data)
    print(df.head())
    df = df[df['step'] <= trim_step]
    df['value_smooth'] = df.groupby('run')['value'].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())

    sns.set_theme(style='whitegrid')
    # plt.figure(figsize=(8,6))
    # YlOrRd, viridis, magma, YlGnBu, RdBu_r, coolwarm
    palette = sns.color_palette("magma", n_colors=df['run'].nunique())
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
log_dirs_mf_safesurf = [
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simnagle-nomf-seq30-ratio1e-3/train-42000',
    './safesurf2025/logs/4-exps_multi_flows/safesurf2025-usevp-simnagle-flows3-seq30-ratio1e-3/train-0',
    './safesurf2025/logs/4-exps_multi_flows/safesurf2025-usevp-simnagle-flows6-seq30-ratio1e-3/train-22000',
    './safesurf2025/logs/4-exps_multi_flows/safesurf2025-usevp-simnagle-flows9-seq30-ratio1e-3/train-13000',
]
plot_usemf(log_dirs_mf_safesurf, 'safesurf2025-usemf_ratio1e-3.png', 35000, smooth_window=6)

# %%
log_dirs_mf_safesurf = [
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simnagle-nomf-seq30-ratio1e-5/train-8000',
    './safesurf2025/logs/4-exps_multi_flows/safesurf2025-usevp-simnagle-flows3-seq30-ratio1e-5/train-0',
    './safesurf2025/logs/4-exps_multi_flows/safesurf2025-usevp-simnagle-flows6-seq30-ratio1e-5/train-39000',
    './safesurf2025/logs/4-exps_multi_flows/safesurf2025-usevp-simnagle-flows9-seq30-ratio1e-5/train-12000'
]
plot_usemf(log_dirs_mf_safesurf, 'safesurf2025-usemf_ratio1e-5.png', 35000, smooth_window=9)


# %%
log_dirs_mf_tcub = [
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simnagle-nomf-seq30-ratio1e-3/train-55000',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows3-seq30-ratio1e-3/train-56000',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows6-seq30-ratio1e-3/train-54000',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows9-seq30-ratio1e-3/train-0',
]
plot_usemf(log_dirs_mf_tcub, 'tcub-usemf_ratio1e-3.png', 30000, smooth_window=6)

# %%
log_dirs_mf_tcub = [
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simnagle-nomf-seq30-ratio1e-5/train-53000',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows3-seq30-ratio1e-5/train-0',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows6-seq30-ratio1e-5/train-17000',
    './tcub/logs/4-exps_multi_flows/tcub-usevp-simnagle-flows9-seq30-ratio1e-5/train-45000'
]
plot_usemf(log_dirs_mf_tcub, 'tcub-usemf_ratio1e-5.png', 30000, smooth_window=6)


# %%
def plot_seqlen(
    log_dirs: list[str],
    save_path: str,
    trim_step=50000,
    smooth_window=12,
):
    tag_to_plot = 'accuracy/accuracy'

    all_data = []
    for idx, run_path in enumerate(log_dirs):
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
        if idx == 0:
            run_name = "seq30"
        elif idx == 1:
            run_name = "seq60"
        else:
            run_name = "seq90"
        for x in scalar_data:
            all_data.append({
                'step': x.step,
                'value': x.value,
                'run': run_name,
            })

    df = pd.DataFrame(all_data)
    print(df.head())
    df = df[df['step'] <= trim_step]
    df['value_smooth'] = df.groupby('run')['value'].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())

    sns.set_theme(style='whitegrid')
    # plt.figure(figsize=(8,6))
    # YlOrRd, viridis, magma, YlGnBu, RdBu_r, coolwarm
    palette = sns.color_palette("YlGnBu", n_colors=df['run'].nunique())
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
log_dirs_seqlen_safesurf = [
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simnagle-nomf-seq30-ratio1e-3/train-42000',
    './safesurf2025/logs/5-exps_seq_len/safesurf2025-usevp-simnagle-nomf-seq60-ratio1e-3/train-28000',
    './safesurf2025/logs/5-exps_seq_len/safesurf2025-usevp-simnagle-nomf-seq90-ratio1e-3/train-0',
]
plot_seqlen(log_dirs_seqlen_safesurf, 'safesurf2025-seqlen_ratio1e-3.png', 30000, smooth_window=9)


# %%
def plot_tcpaug(
    log_dirs: list[str],
    save_path: str,
    trim_step=50000,
    smooth_window=12,
):
    tag_to_plot = 'accuracy/accuracy'

    all_data = []
    for idx, run_path in enumerate(log_dirs):
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
        if idx == 0:
            run_name = "noaug"
        elif idx == 1:
            run_name = "simall"
        elif idx == 2:
            run_name = "simloss"
        elif idx == 3:
            run_name = "simnagle"
        else:
            run_name = "simshift"
        for x in scalar_data:
            all_data.append({
                'step': x.step,
                'value': x.value,
                'run': run_name,
            })

    df = pd.DataFrame(all_data)
    print(df.head())
    df = df[df['step'] <= trim_step]
    df['value_smooth'] = df.groupby('run')['value'].transform(lambda x: x.rolling(smooth_window, min_periods=1).mean())

    sns.set_theme(style='whitegrid')
    # plt.figure(figsize=(8,6))
    # YlOrRd, viridis, magma, YlGnBu, RdBu_r, coolwarm
    palette = sns.color_palette("magma", n_colors=df['run'].nunique())
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

    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    idx = df.groupby('run')['value_smooth'].idxmax()
    df_max = df.loc[idx]
    df_max_simple = df_max[['run', 'value_smooth']].reset_index(drop=True)

    def last_region_mean(x):
        n = int(len(x) * 0.35)
        return x.tail(n).mean()
    df_mean = df.groupby('run')['value_smooth'].apply(last_region_mean)
    return df_max_simple, df_mean

# %%
log_dirs_tcpaug_cic = [
    './cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1e-3/train-8000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simall-nomf-seq30-ratio1e-3/train-12000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simloss-nomf-seq30-ratio1e-3/train-8000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simnagle-nomf-seq30-ratio1e-3/train-35000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simshift-nomf-seq30-ratio1e-3/train-46000',
]
df_runs_max, df_runs_mean = plot_tcpaug(log_dirs_tcpaug_cic, 'cicdarknet-tcpaug_ratio1e-3.png', 100000, smooth_window=3)
print(df_runs_max)
"""cicdarknet-tcpaug_ratio1e-3
        run  value_smooth
0     noaug      0.901701
1    simall      0.938095
2   simloss      0.980952
3  simnagle      0.961905
4  simshift      0.942857
"""
print(df_runs_mean)

# %%
log_dirs_tcpaug_cic = [
    './cicdarknet/logs/3-exps_variable_prototypes/cicdarknet-usevp-noaug-nomf-seq30-ratio1e-5/train-32000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simall-nomf-seq30-ratio1e-5/train-48000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simloss-nomf-seq30-ratio1e-5/train-50000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simnagle-nomf-seq30-ratio1e-5/train-10000',
    './cicdarknet/logs/2-exps_tcp_augmentation/cicdarknet-usevp-simshift-nomf-seq30-ratio1e-5/train-55000',
]
df_runs_max, df_runs_mean = plot_tcpaug(log_dirs_tcpaug_cic, 'cicdarknet-tcpaug_ratio1e-5.png', 100000, smooth_window=5)
print(df_runs_max)
"""cicdarknet-tcpaug_ratio1e-5
        run  value_smooth
0     noaug      0.888175
1    simall      0.954286
2   simloss      0.937143
3  simnagle      0.977143
4  simshift      0.917143
"""
print(df_runs_mean)


# %%
log_dirs_tcpaug_tcub = [
    './tcub/logs/3-exps_variable_prototypes/tcub-usevp-noaug-nomf-seq30-ratio1e-3/train-55000',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simall-nomf-seq30-ratio1e-3/train-45000',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simloss-nomf-seq30-ratio1e-3/train-0',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simnagle-nomf-seq30-ratio1e-3/train-55000',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simshift-nomf-seq30-ratio1e-3/train-25000',
]
df_runs_max, df_runs_mean = plot_tcpaug(log_dirs_tcpaug_tcub, 'tcub-tcpaug_ratio1e-3.png', 100000, smooth_window=5)
print(df_runs_max)
"""tcub-tcpaug_ratio1e-3
        run  value_smooth
0     noaug      0.905714
1    simall      0.945714
2   simloss      0.945714
3  simnagle      0.937143
4  simshift      0.914286
"""
print(df_runs_mean)

# %%
log_dirs_tcpaug_tcub = [
    './tcub/logs/3-exps_variable_prototypes/tcub-usevp-noaug-nomf-seq30-ratio1e-5/train-35000',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simall-nomf-seq30-ratio1e-5/train-11000',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simloss-nomf-seq30-ratio1e-5/train-19000',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simnagle-nomf-seq30-ratio1e-5/train-53000',
    './tcub/logs/2-exps_tcp_augmentation/tcub-usevp-simshift-nomf-seq30-ratio1e-5/train-0',
]
df_runs_max, df_runs_mean = plot_tcpaug(log_dirs_tcpaug_tcub, 'tcub-tcpaug_ratio1e-5.png', 100000, smooth_window=5)
print(df_runs_max)
"""tcub-tcpaug_ratio1e-5
        run  value_smooth
0     noaug      0.886571
1    simall      0.965714
2   simloss      0.925714
3  simnagle      0.925714
4  simshift      0.948571
"""
print(df_runs_mean)


# %%
log_dirs_tcpaug_safesurf = [
    './safesurf2025/logs/3-exps_variable_prototypes/safesurf2025-usevp-noaug-nomf-seq30-ratio1e-3/train-50000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simall-nomf-seq30-ratio1e-3/train-38000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simloss-nomf-seq30-ratio1e-3/train-53000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simnagle-nomf-seq30-ratio1e-3/train-42000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simshift-nomf-seq30-ratio1e-3/train-54000',
]
df_runs_max, df_runs_mean = plot_tcpaug(log_dirs_tcpaug_safesurf, 'safesurf2025-tcpaug_ratio1e-3.png', 100000, smooth_window=5)
print(df_runs_max)
"""safesurf2025-tcpaug_ratio1e-3
        run  value_smooth
0     noaug      0.973016
1    simall      0.985714
2   simloss      0.972431
3  simnagle      0.983214
4  simshift      0.973016
"""
print(df_runs_mean)

# %%
log_dirs_tcpaug_safesurf = [
    './safesurf2025/logs/3-exps_variable_prototypes/safesurf2025-usevp-noaug-nomf-seq30-ratio1e-5/train-55000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simall-nomf-seq30-ratio1e-5/train-32000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simloss-nomf-seq30-ratio1e-5/train-48000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simnagle-nomf-seq30-ratio1e-5/train-8000',
    './safesurf2025/logs/2-exps_tcp_augmentation/safesurf2025-usevp-simshift-nomf-seq30-ratio1e-5/train-57000',
]
df_runs_max, df_runs_mean = plot_tcpaug(log_dirs_tcpaug_safesurf, 'safesurf2025-tcpaug_ratio1e-5.png', 100000, smooth_window=500)
print(df_runs_max)
"""safesurf2025-tcpaug_ratio1e-5
        run  value_smooth
0     noaug      0.783187
1    simall      0.897619
2   simloss      0.980586
3  simnagle      0.940208
4  simshift      0.819299
"""
print(df_runs_mean)
