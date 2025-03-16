import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.patches as patches
import seaborn as sns

from .general_utils import get_eval_filename, get_full_model_name

def parallelCoordinatesPlot(title, N, data, category, ynames, colors=None, category_names=None, savepath=None):
    """
    A legend is added, if category_names is not None.

    :param title: The title of the plot.
    :param N: Number of data sets (i.e., lines).
    :param data: A list containing one array per parallel axis, each containing N data points.
    :param category: An array containing the category of each data set.
    :param category_names: Labels of the categories. Must have the same length as set(category).
    :param ynames: The labels of the parallel axes.
    :param colors: A colormap to use.
    :return:
    """

    fig, host = plt.subplots(figsize=(24, 8))

    # organize the data
    ys = np.dstack(data)[0]
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=7)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()
    host.set_title(title, fontsize=15)

    if colors is None:
        colors = plt.cm.tab10.colors
    if category_names is not None:
        legend_handles = [None for _ in category_names]
    else:
        legend_handles = [None for _ in set(category)]
    for j in range(N):
        # to just draw straight lines between the axes:
        # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

        # create bezier curves
        # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
        #   at one third towards the next axis; the first and last axis have one less control vertex
        # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
        # y-coordinate: repeat every point three times, except the first and last only twice
        verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                         np.repeat(zs[j, :], 3)[1:-1]))
        # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=colors[category[j]])
        legend_handles[category[j]] = patch
        host.add_patch(patch)

        if category_names is not None:
            host.legend(legend_handles, category_names,
                        loc='lower center', bbox_to_anchor=(0.5, -0.18),
                        ncol=len(category_names)//2, fancybox=True, shadow=True)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()



def barPlot(title, metric_name, directions, results_per_model, colors, width=0.05, savepath=None):

    x = np.arange(len(directions))  # the label locations
    width = width  # the width of the bars
    multiplier = 0
    nb_model = len(results_per_model)

    fig, ax = plt.subplots(layout='constrained', figsize=(24, 8))

    for i, (model, results) in enumerate(results_per_model.items()):
        offset = width * multiplier
        rects = ax.bar(x + offset, results["mean_score"], width, label=model, yerr=results["std_unbias_score"], align='center', ecolor='black', capsize=2, color = colors[i])
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(f"{metric_name} score")
    ax.set_ylim(0)
    ax.set_title(title, fontsize=15)
    ax.set_xticks(x + (nb_model//2)*width, directions)
    ax.legend(loc='upper center', ncols=len(results_per_model)//2, fancybox=True, shadow=True)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    plt.close()


def concatenate_results_parrPlot(directions, models, model_sizes, datasets, reduce_sizes, metrics_names, agg_keys, additionnal_names=None, verbose=False):
    """
    agg_keys should be a list containing keys present in output dictonnary for every metrics desired
    for all metrics, can be only ["mean_score", "std_score", "std_unbias_score"] (or less)
    """
    metrics_names2metrics = {"ROUGE-1": "rouge1",
                             "ROUGE-2": "rouge2",
                             "ROUGE-L": "rougeL",
                             "ROUGE-Lsum": "rougeLsum",
                             "BLEU": "bleu",
                             "SacreBLEU": "sacrebleu",
                             "chrF": "chrf",
                             "chrF++": "chrfplusplus",
                             "COMET": "comet",
                             "BLEURT": "bleurt",
                             "BERTscore": "bertscore",
                             "METEOR": "meteor"}
    metrics = [metrics_names2metrics[name] for name in metrics_names] # Want something ordered, don't only take dico.values()
    
    data = {key: [[] for _ in range(len(metrics))] for key in agg_keys}
    
    print("Extracting and concatenating metrics...")
    additionnal_names = [None]*len(models) if additionnal_names is None else additionnal_names
    for dataset_name, reduce_size in zip(datasets, reduce_sizes):
        for model_name, model_size, additionnal_name in zip(models, model_sizes, additionnal_names):
            for direction in directions:
                eval_filename = get_eval_filename(direction, dataset_name, model_name, model_size, reduce_size, additionnal_name)
                if verbose:
                    print(eval_filename)
                with open(eval_filename, "rb") as f:
                    evaluations = pickle.load(f)
                for i, m in enumerate(metrics):
                    for key in agg_keys:
                        data[key][i].append(evaluations[m][key])
    return data

def make_parallel_plot(directions,
                       models, model_sizes,
                       datasets, reduce_sizes,
                       metrics_names,
                       list_colors_per, 
                       additionnal_names = None, colors=None, verbose=False, savepath=None):
    # Aggregate eval data
    data = concatenate_results_parrPlot(directions, models, model_sizes, datasets, reduce_sizes, metrics_names, agg_keys=["mean_score"], additionnal_names=additionnal_names, verbose=verbose)
    data = data["mean_score"]

    # Generate plot categories
    ## Precompute categories names
    print(f"Generating categories based {list_colors_per} type ('list_colors_per' param)...")

    dataset_name2real_name = {"wnt23": "WNT23", "flores": "FLORES+"}
    dataset_name2real_name_and_reduction = {}
    for dataset_name, reduce_size in zip(datasets, reduce_sizes):
        dataset_name2real_name_and_reduction[dataset_name] = dataset_name2real_name[dataset_name] + f" - reduct to {reduce_size} samples"
    category_names_data = [dataset_name2real_name_and_reduction[dataset_name] for dataset_name in datasets] if "dataset" in list_colors_per else []
    category_names_direction = directions if "direction" in list_colors_per else []

    additionnal_names = [None]*len(models) if additionnal_names is None else additionnal_names
    category_names_models = ([get_full_model_name(model_name, model_size, additionnal_name)
                              for model_name, model_size, additionnal_name in zip(models, model_sizes, additionnal_names)]
                              if "model" in list_colors_per
                              else [])
    print(category_names_models)

    ## Generate all combinaisons of categories
    category_names = []
    for cat_data in (category_names_data if len(category_names_data)>0 else [""]):
        is_text = len(category_names_data)>0 and (len(category_names_direction)>0 or len(category_names_models)>0)
        cat1 = cat_data + (" - " if is_text else "")
        for cat_model in (category_names_models if len(category_names_models)>0 else [""]):
            is_text = len(category_names_models)>0 and len(category_names_direction)>0
            cat2 = cat1 + cat_model + (" - " if is_text else "")
            for cat_dir in (category_names_direction if len(category_names_direction)>0 else [""]):
                cat3 = cat2 + cat_dir
                category_names.append(cat3)
    elem2cat = {cat_name: i for i, cat_name in enumerate(category_names)}

    ## Get category name per element
    category = []
    for dataset_name in datasets:
        for model_name, model_size, additionnal_name in zip(models, model_sizes, additionnal_names):
            for direction in directions:
                is_text = len(category_names_data)>0 and (len(category_names_direction)>0 or len(category_names_models)>0)
                cat_name = (dataset_name2real_name_and_reduction[dataset_name] if len(category_names_data)>0 else "") + (" - " if is_text else "")
                is_text = len(category_names_models)>0 and len(category_names_direction)>0
                cat_name = (cat_name
                            + (get_full_model_name(model_name, model_size, additionnal_name)
                                if len(category_names_models)>0
                                else "")
                            + (" - " if is_text else ""))
                cat_name = cat_name + (direction if len(category_names_direction)>0 else "")
                category.append(elem2cat[cat_name])

    if colors is None and len(list_colors_per)==1:
        if "dataset" in list_colors_per:
            colors = plt.cm.Accent.colors
        elif "direction" in list_colors_per:
            colors = plt.cm.tab20.colors
        else:
            colors = plt.cm.Dark2.colors + plt.cm.tab10.colors[0:7] + plt.cm.tab10.colors[8:]
    elif colors is None:
        colors = plt.cm.Dark2.colors + plt.cm.tab10.colors[0:7] + plt.cm.tab10.colors[8:]

    # Plot
    print("Plotting in parallel coordinates plot...")
    n_datasets, n_directions, n_models = len(directions), len(models), len(datasets)
    parallelCoordinatesPlot(title = f"Influence of {list_colors_per} on translation performances",
                            N = n_datasets*n_directions*n_models,
                            data = data,
                            category = category,
                            category_names = category_names,
                            ynames = metrics_names,
                            colors=colors,
                            savepath=savepath)

def concatenate_results_barPlot(directions, models, model_sizes, dataset_name, reduce_size, metric_name, additionnal_names=None, verbose=False):
    """
    for all metrics, can be only ["mean_score", "std_score", "std_unbias_score"] (or less)
    """
    metrics_names2metrics = {"ROUGE-1": "rouge1",
                             "ROUGE-2": "rouge2",
                             "ROUGE-L": "rougeL",
                             "ROUGE-Lsum": "rougeLsum",
                             "BLEU": "bleu",
                             "SacreBLEU": "sacrebleu",
                             "chrF": "chrf",
                             "chrF++": "chrfplusplus",
                             "COMET": "comet",
                             "BLEURT": "bleurt",
                             "BERTscore": "bertscore",
                             "METEOR": "meteor"}
    metric = metrics_names2metrics[metric_name]
    additionnal_names = [None]*len(models) if additionnal_names is None else additionnal_names
    results_per_model = {get_full_model_name(model_name, model_size, additionnal_name): {"mean_score":[], "std_unbias_score":[]} for model_name, model_size, additionnal_name in zip(models, model_sizes, additionnal_names)}
    
    print("Extracting and concatenating metrics...")
    for model_name, model_size, additionnal_name in zip(models, model_sizes, additionnal_names):
        for direction in directions:
            eval_filename = get_eval_filename(direction, dataset_name, model_name, model_size, reduce_size, additionnal_name)
            if verbose:
                print(eval_filename)
            with open(eval_filename, "rb") as f:
                evaluations = pickle.load(f)
            results_per_model[get_full_model_name(model_name, model_size, additionnal_name)]["mean_score"].append(evaluations[metric]["mean_score"])
            results_per_model[get_full_model_name(model_name, model_size, additionnal_name)]["std_unbias_score"].append(evaluations[metric]["std_unbias_score"])
    return results_per_model

def make_bar_plot(directions,
                    model_names, model_sizes,
                    dataset_name, reduce_size,
                    metric_names,
                    additionnal_names = None,
                    cmap=None,
                    width=0.05,
                    savepath = None):
    for metric_name in metric_names:
        title = f"{metric_name} translation evaluation on dataset {dataset_name} (mean score with unbiased std)"
        results_per_model = concatenate_results_barPlot(directions, model_names, model_sizes, dataset_name, reduce_size, metric_name, additionnal_names=additionnal_names, verbose=False)
        cmap = "Spectral" if cmap is None else cmap
        cmap_perso = ListedColormap(sns.color_palette(cmap, len(results_per_model)).as_hex())
        barPlot(title,
                metric_name,
                directions,
                results_per_model,
                width=width,
                colors = cmap_perso.colors,
                savepath = (savepath+f"_{metric_name}" if savepath is not None else None))