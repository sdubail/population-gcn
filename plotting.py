import matplotlib.pyplot as plt
import numpy as np

def plot_graph_connectivity(graph:np.ndarray):
    # Plot the adjacency matrix
    plt.figure(figsize=(10, 10))
    plt.title("Adjacency Matrix Connecitvity")
    plt.matshow(graph, fignum=1, cmap="viridis")  # Use 'viridis' or 'Blues' for contrast
    plt.colorbar(label="Edge Presence")
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.show()
    
def plot_acc_auc_boxplot(filenames_for_experience_with_rd_graph_support, df_result):
    

    # Set useful params depending on the number of experience
    N_RD_EXP = len(filenames_for_experience_with_rd_graph_support)
    xticks_list = np.arange(2 + N_RD_EXP) +1
    xticks_lables = ['GCN'] + [f'rd{k+1} GCN' for k in range(N_RD_EXP)] + ['Ridge classifier']

    # Set colors for each boxplot
    colors = ['skyblue', 'lightgreen'] * (N_RD_EXP + 1)*2

    # Create the first plot for lin and acc
    plt.figure(figsize=(12, 6))

    # Boxplot for lin and acc
    plt.subplot(1, 2, 1)
    Y = [df_result['GCN_acc']] + [df_result[f'rd{k+1}_GCN_acc'] for k in range(N_RD_EXP)] + [df_result['ridge_classifier_acc']]
    bp1 = plt.boxplot(Y, patch_artist=True, widths=0.5)

    # Color the boxes for the first subplot
    for patch, color in zip(bp1['boxes'], colors[:2]):
        patch.set_facecolor(color)

    # Add red square for the mean
    for i, median in enumerate(bp1['medians']):
        median_value = median.get_ydata()[0]
        plt.scatter(i+1, median_value, color='red', marker='s', zorder=5)

    # Set titles and labels
    plt.title('ABIDE accuracy')
    plt.xticks(xticks_list, xticks_lables)

    # Create the second plot for lin_auc and auc
    plt.subplot(1, 2, 2)
    Y = [df_result['GCN_auc']] + [df_result[f'rd{k+1}_GCN_auc'] for k in range(N_RD_EXP)] + [df_result['ridge_classifier_auc']]
    bp2 = plt.boxplot(Y, patch_artist=True, widths=0.5)

    # Color the boxes for the second subplot
    for patch, color in zip(bp2['boxes'], colors[2:]):
        patch.set_facecolor(color)

    # Add red square for the mean
    for i, median in enumerate(bp2['medians']):
        median_value = median.get_ydata()[0]
        plt.scatter(i+1, median_value, color='red', marker='s', zorder=5)

    # Set titles and labels
    plt.title('ABIDE AUC')
    plt.xticks(xticks_list, xticks_lables)

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plots
    plt.show()
