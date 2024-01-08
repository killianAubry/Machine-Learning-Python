import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import gridspec
import warnings
warnings.filterwarnings("ignore")
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], wspace=0.3, hspace=0.3)
def load_results(log_filename):
    with open(log_filename, 'r') as log_file:
        results = json.load(log_file)
    return results
def print_progress_bar(percentage):
   bar_length = 30
   num_blocks = int(round(bar_length * percentage))
   progress = "[" + "█" * num_blocks + "-" * (bar_length - num_blocks) + "]"
   percentage_str = f"{percentage * 100:.2f}%"
   print(f"\r{progress.ljust(bar_length + 2)}{percentage_str.ljust(6)}", end="")
   if percentage == 1.0:
       print()  # Move to the next line after the progress bar is complete

def plot_time(data):
    ax_time = plt.subplot(gs[0, 2])
    ax_time.set_title('Average Time per Epoch')
    df_time = pd.DataFrame(data, columns=['average_time_per_epoch', 'Marker'])
    sns.violinplot(x='Marker', y='average_time_per_epoch', data=df_time, ax=ax_time, hue="Marker",split=True, inner="box", fill=False,palette={"legacy": "blue", "proposed": "orange"})

def plot_f1(data):
    ax_f1 = plt.subplot(gs[0, 3])
    ax_f1.set_title('F1 Score')
    df_f1 = pd.DataFrame(data, columns=['f1_score', 'Marker'])
    sns.violinplot(x='Marker', y='f1_score', data=df_f1, ax=ax_f1, hue="Marker",split=True, inner="box", fill=False,palette={"legacy": "blue", "proposed": "orange"})

def plot_results(results_list):
    sns.set_theme(style="dark")
    # Create subplots for each metric

    # Initialize color map based on marker type
    marker_colors = {"proposed": "magma", "legacy": "viridis"}

    # ...
    sum_dataLegacy = np.array([[0, 0],[0, 0]])
    sum_dataProposed = np.array([[0, 0],[0, 0]])
    legAdd = 0
    propAdd = 0
    avg_LossL =  []
    avg_LossP =  []
    count = 0
    timePlotData = []
    f1PlotData = []
    muted_palette = sns.color_palette("deep", 2, desat=0.3)
    legend_elements = [
        plt.Line2D([0], [0], color='blue', label='legacy'),
        plt.Line2D([0], [0], color='orange', label='proposed')
    ]
    for i, data in enumerate(results_list):
        print_progress_bar(count/len(results_list))
        count += 1
        marker_type = data["marker_type"]

        # Plot training loss
        
        ax_loss = plt.subplot(gs[0, :2])
        ax_loss.set_title('Training Loss')
        df_loss = pd.DataFrame({'Epochs': range(len(data["training_loss"])), 'Loss': data["training_loss"], 'Marker': marker_type})
        df_loss['EMA'] = df_loss.groupby('Marker')['Loss'].transform(lambda x: x.ewm(alpha=0.1).mean())
        # Use a custom palette with dark blue for "legacy" and orange for "proposed"
        if marker_type == 'legacy':
            if avg_LossL == []:
                avg_LossL = [list(range(len(data["training_loss"]))), data["training_loss"]]
            else:
                avg_LossL = np.mean(np.array([avg_LossL, [list(range(len(data["training_loss"]))), data["training_loss"]]]), axis=0 )
        else:
            if avg_LossP == []:
                avg_LossP = [list(range(len(data["training_loss"]))), data["training_loss"]]
            else:
                avg_LossP = np.mean(np.array([avg_LossP, [list(range(len(data["training_loss"]))), data["training_loss"]]]), axis=0 )
        sns.lineplot(x='Epochs', y='EMA', hue='Marker', data=df_loss, linestyle='--', ax=ax_loss, palette={ "legacy": muted_palette[0], "proposed": muted_palette[1] }, legend=False, linewidth=0.25)
        if count == len(results_list)-1:
            df_lossP = pd.DataFrame({'Epochs': avg_LossP[0], 'Loss': avg_LossP[1]})
            df_lossP['EMA'] = df_lossP['Loss'].transform(lambda x: x.ewm(alpha=0.1).mean())
            df_lossL = pd.DataFrame({'Epochs': avg_LossL[0], 'Loss': avg_LossL[1]})
            df_lossL['EMA'] = df_lossL['Loss'].transform(lambda x: x.ewm(alpha=0.1).mean())
            sns.lineplot(x='Epochs', y='EMA', data=df_lossP, linestyle='-', ax=ax_loss, color="#FF8C00", legend=False, linewidth=2)
            sns.lineplot(x='Epochs', y='EMA', data=df_lossL, linestyle='-', ax=ax_loss, color="blue", legend=False, linewidth=2)
        plt.legend(handles=legend_elements)
        timePlotData.append([data["average_time_per_epoch"], marker_type])
        f1PlotData.append([data["f1_score"], marker_type])
        confusion_matrix = data["confusion_matrix"]
        total_samples = confusion_matrix["true_negatives"] + confusion_matrix["false_positives"] + confusion_matrix["false_negatives"] + confusion_matrix["true_positives"]
        matrix_data = np.array([[((confusion_matrix["true_negatives"]*100)/ total_samples)/100 , ((confusion_matrix["false_positives"]*100)/ total_samples)/100 ],
                                [((confusion_matrix["false_negatives"]*100)/ total_samples)/100 , ((confusion_matrix["true_positives"]*100)/ total_samples)/100 ]])
        normalized_matrix_data = matrix_data #/ total_samples 
        if marker_type == "proposed":
            ax_cm = plt.subplot(gs[1, :2])
            normalized_matrix_data = np.add(sum_dataProposed, normalized_matrix_data)
            sum_dataProposed = normalized_matrix_data
            propAdd += 1

        else:
            ax_cm = plt.subplot(gs[1, 2:])
            normalized_matrix_data = np.add(sum_dataLegacy, normalized_matrix_data)
            sum_dataLegacy = normalized_matrix_data
            legAdd += 1
    
    ax_cm = plt.subplot(gs[1, :2])
    ax_cm.set_title('Confusion Matrix (Smooth Heatmap)')
    ax_cm.set_xticklabels(['Porange Negative', 'Porange Positive'])
    ax_cm.set_yticklabels(['Act Negative', 'Act Positive'])
    ax_cm.set_xlabel(f'Marker Type: {marker_type.capitalize()}')
    sns.heatmap(np.divide(sum_dataProposed.tolist(),(propAdd/100)), cmap='viridis', annot=True, fmt=".1f",
                    cbar=True, ax=ax_cm, linewidths=0, annot_kws={"size": 14}, vmin=0, vmax=100)
    ax_cm = plt.subplot(gs[1, 2:])
    ax_cm.set_title('Confusion Matrix (Smooth Heatmap)')
    ax_cm.set_xticklabels(['Porange Negative', 'Porange Positive'])
    ax_cm.set_yticklabels(['Act Negative', 'Act Positive'])
    ax_cm.set_xlabel(f'Marker Type: {marker_type.capitalize()}')
    sns.heatmap(np.divide(sum_dataLegacy.tolist(),(legAdd/100)), cmap='viridis', annot=True, fmt=".1f",
                    cbar=True, ax=ax_cm, linewidths=0, annot_kws={"size": 14}, vmin=0, vmax=100)
    plot_f1(f1PlotData)
    plot_time(timePlotData)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == "__main__":
    log_filename = "combined.json"  # Assuming this is the correct filename
    results_list = load_results(log_filename)
    plot_results(results_list)
