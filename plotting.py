import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.colors as mcolors


rcParams = {
    'font.size': 12,            # General font size
    'axes.titlesize': 20,       # Title font size of the plot
    'axes.labelsize': 18,       # Font size of x and y labels
    'xtick.labelsize': 12,      # Font size of x tick labels
    'ytick.labelsize': 14,      # Font size of y tick labels
    'legend.fontsize': 12,      # Font size of legend text
    'figure.titlesize': 20,     # Font size of the figure title
    #'figure.figsize': (10, 6)   # Default figure size
}

plt.rcParams.update(rcParams)

def plot_graph_connectivity(graph:np.ndarray, subtitle:str="", saving_filename:str=None):
    # Plot the adjacency matrix
    plt.figure(figsize=(10, 10))
    plt.title("Adjacency Matrix Connecitvity"+subtitle)
    plt.matshow(graph, fignum=False, cmap="viridis")  # Use 'viridis' or 'Blues' for contrast
    plt.colorbar(label="Edge Presence")
    plt.xlabel("Node")
    plt.ylabel("Node")
    if saving_filename != None:
        plt.savefig(saving_filename)
        plt.close()
    else:
        plt.show()

def build_dic_for_plot_summary_adj_matrix_construction(phenotypic_graph, features_similarity_graph, adjacency_matrix):
    dic = {}
    dic['phenotypic_graph'] = phenotypic_graph
    dic['features_similarity_graph'] = features_similarity_graph
    dic['adjacency_matrix'] = adjacency_matrix
    return dic

def plot_summary_adj_matrix_construction(dic:dict, saving_filename:str=None):
    graphs = [dic['phenotypic_graph'], dic['features_similarity_graph'], dic['adjacency_matrix']]  # Replace with actual matrices
    subtitles = ["Phenotypic Graph", "Features similarity", "Adjacency Matrix"]

    # Determine the global min and max values across all graphs
    vmin = min(np.min(graph) for graph in graphs)
    vmax = max(np.max(graph) for graph in graphs)

    # Create a figure with gridspec to manage spacing
    fig = plt.figure(figsize=(15, 5))
    spec = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1])  # Last column for the colorbar

    # Plot each graph
    axes = [fig.add_subplot(spec[i]) for i in range(3)]
    for i, ax in enumerate(axes):
        im = ax.matshow(graphs[i], cmap="viridis", vmin=vmin, vmax=vmax)  # Use global vmin and vmax
        ax.set_title(subtitles[i])
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")

    # Add a single colorbar in the designated column
    cbar_ax = fig.add_subplot(spec[3])  # Colorbar in the last column
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')

    plt.tight_layout()
    if saving_filename != None:
        plt.savefig(saving_filename)
        plt.close()
    else:
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

def plot_adjacency_matrix(matrix, threshold=0.5):
    """
    Plots the adjacency matrix of a graph and shows the density of the graph as a bar chart.
    
    Parameters:
    - matrix: 2D numpy array representing the adjacency matrix of the graph.
    - threshold: float value to define zero vs non-zero. 
                 Values below or equal to this threshold are considered zero (no edge).
    """
    # Create a mask for the zero and non-zero elements based on the threshold.
    mask = matrix > threshold
    
    # Calculate the density of the adjacency matrix.
    num_elements = matrix.size  # Total number of elements in the matrix
    num_edges = np.sum(mask)/2  # Number of non-zero elements (edges)
    num_nodes = matrix.shape[0]  # Number of nodes (N)
    
    # Density calculation for an undirected graph
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    
    # Set the color map: 'white' for zero and 'black' for non-zero.
    cmap = mcolors.ListedColormap(['white', 'black'])
    bounds = [0, 1, 2]  # Define the bounds of the color map
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the adjacency matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(mask, cmap=cmap, norm=norm)
    # plt.axis('off')  # Hide the axes for better visualization
    plt.title(f'Density of Graph: {density:.2f}')
    
    # Plot the density as a bar chart
    # plt.subplot(2, 1, 2)  # Plot the density in the second subplot
    # plt.bar(['Density of Graph'], [density], color='black')
    plt.ylabel('Node')
    plt.xlabel('Node')

    plt.tight_layout()
    plt.show()
 
def plot_histo_nonzero_edges_value(graph_adj_matrix):
    # Exemple de données à afficher
    data = np.abs(graph_adj_matrix[np.triu_indices_from(graph_adj_matrix, k=1)])
    data = data[data>0]
    num_bins = 60
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), num_bins) 

    # Création de l'histogramme
    plt.hist(data, bins=60, color='lightblue', edgecolor='black', density=True)

    # Ajout des titres et des labels
    plt.xlabel('value of edges')
    plt.ylabel('density')
    plt.xscale('log')

    # Calcul des statistiques
    median_value = np.median(data)
    q1 = np.percentile(data, 25)  # Premier quartile (Q1)
    q3 = np.percentile(data, 75)  # Troisième quartile (Q3)
    deciles = [np.percentile(data, i * 10) for i in range(1, 10)]  # Déciles de D1 à D9
    min_value = data.min()
    max_value = data.max()

    # Tracé de la ligne rouge pour la médiane sur l'axe x (log)
    plt.axvline(median_value, color='red', linestyle='--', linewidth=2, label=f'Median = {median_value:.2f}')

    # Tracé des lignes pour Q1 et Q3 en couleur différente (par exemple, bleu pour Q1 et orange pour Q3)
    plt.axvline(q1, color='blue', linestyle=':', linewidth=2, label=f'Q1 = {q1:.2f}')
    plt.axvline(q3, color='orange', linestyle='-.', linewidth=2, label=f'Q3 = {q3:.2f}')

    colors = ['purple', 'brown', 'pink', 'cyan', 'lime', 'magenta', 'yellow', 'grey', 'black']
    for i, d in enumerate(deciles):
        if (i ==0) or (i==8):   
            plt.axvline(d, color=colors[i], linestyle='--', linewidth=1, label=f'D{i+1} = {d:.2f}')

    # Tracé des lignes pour Q1 et Q3 en couleur différente (par exemple, bleu pour Q1 et orange pour Q3)
    plt.axvline(min_value, color='brown', linestyle='--', linewidth=2, label=f'min = {min_value:.2f}')
    plt.axvline(max_value, color='black', linestyle='--', linewidth=2, label=f'max = {max_value:.2f}')

    # Affichage de l'histogramme
    plt.legend()
    plt.show()
    

def plot_mosaic_adj_matrix(dic:dict, saving_filename:str="../figures/construction_of_adjacency_matrix_and_sparcity.pdf"):
    graphs = [dic['phenotypic_graph'], dic['features_similarity_graph'], dic['adjacency_matrix']]  # Replace with actual matrices
    subtitles = ["a) Phenotypic graph", "b) Subjects similarity", "c) Adjacency matrix"]

    # Determine the global min and max values across all graphs
    vmin = min(np.min(graph) for graph in graphs)
    vmax = max(np.max(graph) for graph in graphs)

    fig = plt.figure(figsize=(15, 10), layout='constrained')
    axs = fig.subplot_mosaic([["graphPheno", "graphPheno", "graphPheno", "graphSim", "graphSim", "graphSim", "graphAdj", "graphAdj", "graphAdj", "cbar"],
                          ["graphDensity", "graphDensity", "graphDensity", "graphDensity", "graphDensity", "EdgeRepartition", "EdgeRepartition", "EdgeRepartition", "EdgeRepartition", "EdgeRepartition"]])

    ## plot graphPheno:
    axs["graphPheno"].set_title(subtitles[0])
    im = axs["graphPheno"].matshow(graphs[0], cmap="viridis", vmin=vmin, vmax=vmax)
    axs["graphPheno"].set_xlabel("Node")
    axs["graphPheno"].set_ylabel("Node")

    ## plot graphSim:
    axs["graphSim"].set_title(subtitles[1])
    axs["graphSim"].matshow(graphs[1], cmap="viridis", vmin=vmin, vmax=vmax)
    axs["graphSim"].set_xlabel("Node")
    axs["graphSim"].set_ylabel("Node")

    ## plot graphAdj:
    axs["graphAdj"].set_title(subtitles[2])
    axs["graphAdj"].matshow(graphs[2], cmap="viridis", vmin=vmin, vmax=vmax)
    axs["graphAdj"].set_xlabel("Node")
    axs["graphAdj"].set_ylabel("Node")
    
    ## Add a single colorbar in the designated column
    fig.colorbar(im, cax=axs["cbar"], orientation='vertical')

    ## add graph density representation
    threshold = 10**-16
    matrix = dic['adjacency_matrix']
    mask = matrix > threshold
    # Calculate the density of the adjacency matrix.
    num_elements = matrix.size  # Total number of elements in the matrix
    num_edges = np.sum(mask)/2  # Number of non-zero elements (edges)
    num_nodes = matrix.shape[0]  # Number of nodes (N)
    # Density calculation for an undirected graph
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    # Set the color map: 'white' for zero and 'black' for non-zero.
    cmap = mcolors.ListedColormap(['white', 'black'])
    bounds = [0, 1, 2]  # Define the bounds of the color map
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # Plot the adjacency matrix
    axs["graphDensity"].imshow(mask, cmap=cmap, norm=norm)
    axs["graphDensity"].axis('off')  # Hide the axes for better visualization
    axs["graphDensity"].set_title(f'd) Density of adjacency matrix: {density:.2f}')
    axs["graphDensity"].set_ylabel('Node')
    axs["graphDensity"].set_xlabel('Node')
    # axs["graphDensity"].tight_layout()
    
    # plot repartition of edges values
    graph_adj_matrix = dic['adjacency_matrix']
    data = np.abs(graph_adj_matrix[np.triu_indices_from(graph_adj_matrix, k=1)])
    data = data[data>0]
    num_bins = 60

    # histogram
    axs["EdgeRepartition"].hist(data, bins=num_bins, color='lightblue', edgecolor='black', density=True)

    # Ajout des titres et des labels
    axs["EdgeRepartition"].set_xlabel('value of non-zero edges')
    axs["EdgeRepartition"].set_ylabel('density')
    axs["EdgeRepartition"].set_xscale('log')
    axs["EdgeRepartition"].set_title('e)')

    # Calcul des statistiques
    median_value = np.median(data)
    q1 = np.percentile(data, 25)  # Premier quartile (Q1)
    q3 = np.percentile(data, 75)  # Troisième quartile (Q3)
    deciles = [np.percentile(data, i * 10) for i in range(1, 10)]  # Déciles de D1 à D9
    min_value = data.min()
    max_value = data.max()

    # Tracé de la ligne rouge pour la médiane sur l'axe x (log)
    axs["EdgeRepartition"].axvline(median_value, color='red', linestyle='--', linewidth=2, label=f'Median = {median_value:.2f}')

    # Tracé des lignes pour Q1 et Q3 en couleur différente (par exemple, bleu pour Q1 et orange pour Q3)
    axs["EdgeRepartition"].axvline(q1, color='blue', linestyle=':', linewidth=2, label=f'Q1 = {q1:.2f}')
    axs["EdgeRepartition"].axvline(q3, color='orange', linestyle='-.', linewidth=2, label=f'Q3 = {q3:.2f}')

    colors = ['purple', 'brown', 'pink', 'cyan', 'lime', 'magenta', 'yellow', 'grey', 'black']
    for i, d in enumerate(deciles):
        if (i ==0) or (i==8):   
            axs["EdgeRepartition"].axvline(d, color=colors[i], linestyle='--', linewidth=1, label=f'D{i+1} = {d:.2f}')

    # Tracé des lignes pour Q1 et Q3 en couleur différente (par exemple, bleu pour Q1 et orange pour Q3)
    axs["EdgeRepartition"].axvline(min_value, color='brown', linestyle='--', linewidth=2, label=f'min = {min_value:.2f}')
    axs["EdgeRepartition"].axvline(max_value, color='black', linestyle='--', linewidth=2, label=f'max = {max_value:.2f}')

    # Affichage de l'histogramme
    axs["EdgeRepartition"].legend()
    plt.savefig(saving_filename, dpi = 250)
    # plt.savefig("../figures/construction_of_adjacency_matrix_and_sparcity.png", dpi = 250)

def create_horizontal_boxplots(dfs_list, labels, colname='GCN_acc', filename="boxplots_ADJ_matrix_sensitivity", x_label='accuracy'):
    """
    Create a graph with n horizontal boxplots.

    Parameters:
        n (int): The number of boxplots to create.
    """
    # Generate random data for each boxplot
    data = [dfs_list[label][colname] for label in labels]
    # df_average = average_several_dfs(list(dfs_list.values()))
    # data.append(df_average['ridge_classifier_acc'])

    # Create the plot
    plt.figure(figsize=(17, 6))
    boxplot = plt.boxplot(data, vert=False, patch_artist=True)

    # Add mean values next to each boxplot
    for i, dataset in enumerate(data, start=1):
        mean_value = np.mean(dataset)
        plt.text(mean_value, i, f'{mean_value:.3f}', ha='left', va='center', fontsize=14, color='black')


    # Label the y-axis
    # labels_y_sticks = list(labels) + ['ridge accuracy']
    cols_name = ['classic', 'worst', '1', 'random']
    cols_labels_name = ['classic', 'worst', 'no', 'random']
    cols_labels_dic = {str(col):label for col, label in zip(cols_name, cols_labels_name)}
    rows_name = ['classic', 'G', 'As', 'worst', 'all', 'random']
    rows_labels_name = ['gender + site', 'gender only', 'site only', 'worst', 'all', 'random']
    rows_labels_dic = {col:label for col, label in zip(rows_name, rows_labels_name)}

    labels_y_sticks = [f"{rows_labels_dic[label.split('_')[1]]} phenom. / {cols_labels_dic[str(label.split('_')[3])]} simil." for label in labels]
    
    plt.yticks(ticks=range(1, len(data) + 1), labels=labels_y_sticks)

    # Add title and labels
    plt.xlabel(x_label)
    # plt.ylabel('Adjacency matrix type')
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"../figures/{filename}.png", dpi=250)
    plt.savefig(f"../figures/{filename}.pdf", dpi=250)


def create_vertical_boxplots(dfs_list, labels, colname='GCN_acc', filename="boxplots_ADJ_matrix_sensitivity", y_label='accuracy', y_legnth=8):
    """
    Create a graph with n horizontal boxplots.

    Parameters:
        n (int): The number of boxplots to create.
    """
    # Generate random data for each boxplot
    data = [dfs_list[label][colname] for label in labels]
    # df_average = average_several_dfs(list(dfs_list.values()))
    # data.append(df_average['ridge_classifier_acc'])

    # Create the plot
    plt.figure(figsize=(10, y_legnth))
    boxplot = plt.boxplot(data, vert=True, patch_artist=True)

    # Add mean values next to each boxplot
    for i, dataset in enumerate(data, start=1):
        mean_value = np.mean(dataset)
        plt.text(i-0.33, mean_value, f'{mean_value:.3f}', ha='left', va='center', fontsize=14, color='black')


    # Label the y-axis
    # labels_y_sticks = list(labels) + ['ridge accuracy']
    cols_name = ['classic', 'worst', '1', 'random']
    cols_labels_name = ['classic', 'worst', 'no', 'random']
    cols_labels_dic = {str(col):label for col, label in zip(cols_name, cols_labels_name)}
    rows_name = ['classic', 'G', 'As', 'worst', 'all', 'random']
    rows_labels_name = ['gender + site', 'gender only', 'site only', 'worst', 'all', 'random']
    rows_labels_dic = {col:label for col, label in zip(rows_name, rows_labels_name)}

    labels_x_sticks = [f"{rows_labels_dic[label.split('_')[1]]} phenom. / {cols_labels_dic[str(label.split('_')[3])]} simil." for label in labels]
    
    plt.xticks(ticks=range(1, len(data) + 1), labels=labels_x_sticks, rotation=90)


    # Add title and labels
    plt.ylabel(y_label)
    # plt.ylabel('Adjacency matrix type')
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"../figures/{filename}.png", dpi=250)
    plt.savefig(f"../figures/{filename}.pdf", dpi=250)

