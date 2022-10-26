from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np


def my_formatter(x, pos):
    val_str = "{:g}".format(x)
    return val_str


def select_itemsets(
    df_patterns: pd.DataFrame, itemsets: list, itemset_col_name: str = "itemsets"
):
    """Slice the frequent pattern info to get only the itemsets of interest.
    Args:
        df_patterns (pd.DataFrame): the dataframe with patterns and their info
        itemsets (list): itemsets to select
        itemset_col_name (str): name of the columns corresponding to the itemset name
    Returns:
        pd.DataFrame: the slice of the dataset of only the itemsets in input
    """

    if type(itemsets) == frozenset:
        itemsets = [itemsets]

    return df_patterns.loc[df_patterns[itemset_col_name].isin(itemsets)]


def slice_by_itemset(df: pd.DataFrame, itemset) -> pd.DataFrame:
    """Slice the dataFrame to select the instances satisfying the itemset
    Args:
        df (pd.DataFrame): the input table
        itemset (frozenset): the itemset
    Returns:
        pd.DataFrame: the slice of the data satifying the itemset
    """

    indexes = df.index
    for item in itemset:
        s = item.split("=")
        attr, value = s[0], "=".join(s[1:])
        indexes = df.loc[indexes].loc[df[attr].astype(str) == value].index
    return df.loc[indexes]


def plotComparisonShapleyValues(
    sh_score_1,
    sh_score_2,
    title=[],
    sharedAxis=False,
    height=0.8,
    linewidth=0.8,
    sizeFig=(7, 7),
    saveFig=False,
    nameFig=None,
    labelsize=10,
    titlesize=10,
    pad=0.5,
    subcaption=True,
    metrics_name=None,
    formatTicks=False,
    deltaLim=None,
    show_figure=True,
    sort="by_value",
    show_both_label=True
):

    """TODO. Plot the Shapley value of two itemsets side by side
    Args:
        sh_score_1 (dict): Shapley value of itemset 1
        sh_score_2 (dict): Shapley value of itemset 2
        title (list): titles of the two figures
        height (float): height of the bar plot
        linewidth (float): linewidth of the bar plot
        sizeFig (tuple): size of the figure
        saveFig (bool): True to save the figure
        nameFig (str): path and name of the figure to save
        sort (str): sort "by_value" as default. specify "alp" for alphabetical order
        todo
    """

    h1, h2 = (height[0], height[1]) if type(height) == list else (height, height)

    import matplotlib.pyplot as plt

    sh_score_1 = {str(",".join(list(k))): v for k, v in sh_score_1.items()}
    sh_score_2 = {str(",".join(list(k))): v for k, v in sh_score_2.items()}

    if sort == "by_value":
        sh_score_1 = { k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[1]) }
        sh_score_2 = { k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[1]) }

    if sort == "alp":
        sh_score_1 = { k: v for k, v in sorted(sh_score_1.items(), key=lambda item: item[0], reverse=True) }
        sh_score_2 = { k: v for k, v in sorted(sh_score_2.items(), key=lambda item: item[0], reverse=True) }

    if sort == "paper":
        sh_score_1 = { k: v*100 for k, v in (sh_score_1.items()) }
        sh_score_2 = { k: v*100 for k, v in (sh_score_2.items()) }


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=sizeFig, dpi=100, sharex=sharedAxis)

    ax1.barh(
        range(len(sh_score_1)),
        sh_score_1.values(),
        align="center",
        color="C01",
        height=h1,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )

    ax1.set_yticks(range(len(sh_score_1)))
    ax1.set_yticklabels(list(sh_score_1.keys()))
    ax1.tick_params(axis="x", labelsize=labelsize)

    if len(title) > 1:
        ax1.set_title(title[0], fontsize=titlesize)

    ax2.barh(
        range(len(sh_score_2)),
        sh_score_2.values(),
        align="center",
        color="C01",
        height=h2,
        linewidth=linewidth,
        edgecolor="#0C4A5B",
    )
    ax2.set_yticks(range(len(sh_score_2)))
    ax2.set_yticklabels(list(sh_score_2.keys()))
    ax2.tick_params(axis="x", labelsize=labelsize)

    if len(title) > 1:
        ax2.set_title(f"{title[1]}", fontsize=titlesize)
    fig.tight_layout(pad=pad)
    if len(title) > 1:
        plt.title(title[1], fontsize=titlesize)
    
    ax1.tick_params(axis="y", labelsize=labelsize)
    
    if show_both_label:
        ax2.tick_params(axis="y", labelsize=labelsize)
        ax2.set_yticks(range(len(sh_score_2)))
        ax2.set_yticklabels(list(sh_score_2.keys()))
    else:
        ax2.set_yticklabels([None]*len(sh_score_2))
    """
    if sharedAxis:
        sh_scores = list(sh_score_1.values()) + list(sh_score_2.values())
        if deltaLim:
            min_x, max_x = min(sh_scores) - deltaLim, max(sh_scores) + deltaLim
        else:
            min_x, max_x = (
                min(sh_scores) + min(0.01, min(sh_scores)),
                max(sh_scores) + min(0.01, max(sh_scores)),
            )
        ax1.set_xlim(min_x, max_x)
        ax2.set_xlim(min_x, max_x)
    """
    s1 = "(a)" if subcaption else ""
    s2 = "(b)" if subcaption else ""

    ax1.set_xlabel(f"{s1}", size=labelsize)
    ax2.set_xlabel(f"{s2}", size=labelsize)

    """
    if formatTicks:
        major_formatter = FuncFormatter(my_formatter)
        ax1.xaxis.set_major_formatter(major_formatter)
        ax2.xaxis.set_major_formatter(major_formatter)
    """
    if saveFig:
        nameFig = "./shap.pdf" if nameFig is None else f"{nameFig}.pdf"
        plt.savefig(nameFig, format="pdf", bbox_inches="tight")
    if show_figure:
        plt.show()
    plt.close()


def plot_true_pred(df_sel_input, column, title=None, groupby="prediction", figsize=(2, 2), show_fig = True):
    """Grouped bar plot
    df_sel (pd.DataFrame) : the input data
    column (str) : define the sets
    groupby (str) : define the color of the sets
    show_fig(bool) : show the figure
    """

    df_sel = df_sel_input.copy()
    if column == "speakerId" and "speakerId" not in df_sel_input.columns:
        
        df_sel['speakerId'] = df_sel.index.map(lambda x: x.split("/")[2])

    import matplotlib.pyplot as plt

    labels = sorted(list(df_sel[column].unique()))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=figsize)

    groupbyvalues = sorted(df_sel[groupby].unique())
    n_groups = len(groupbyvalues)
    ### prepare for grouping the bars
    total_width = 0.5  # 0 ≤ total_width ≤ 1
    d = 0.05  # gap between bars, as a fraction of the bar width, 0 ≤ d ≤ ∞
    width = total_width / (n_groups + (n_groups - 1) * d)
    offset = -total_width / 2

    for i in groupbyvalues:
        df_sel_i = df_sel.loc[df_sel[groupby] == i]
        vc = df_sel_i[column].value_counts()
        values = [vc[k] if k in vc else 0 for k in labels]
        s = -1 if i == 0 else 1
        ax.bar(x + offset, values, width, align="edge", label=i)
        offset += (1 + d) * width
    ax.set_xticks(x, labels, rotation=90)
    if title:
        plt.title(title)
    plt.legend()
    if show_fig:
        plt.show()
    plt.close()


def attributes_in_itemset(itemset, attributes, alls=True):
    """Check if attributes are in the itemset (all or at least one)
    Args:
        itemset (frozenset): the itemset
        attributes (list): list of itemset of interest
        alls (bool): If True, check if ALL attributes of the itemset are the input attributes.
        If False, check AT LEAST one attribute of the itemset is in the input attributes.
    """
    # Avoid returning the empty itemset (i.e., info of entire dataset)
    if itemset == frozenset() and attributes:
        return False

    for item in itemset:
        # Get the attribute
        attr_i = item.split("=")[0]

        # If True, check if ALL attributes of the itemset are the input attributes.
        if alls:
            # Check if the attribute is present. If not, the itemset is not admitted
            if attr_i not in attributes:
                return False
        else:
            # Check if least one attribute. If yes, return True
            if attr_i in attributes:
                return True
    if alls:
        # All attributes of the itemset are indeed admitted
        return True
    else:
        # Otherwise, it means that we find None
        return False


def filter_itemset_df_by_attributes(
    df: pd.DataFrame, attributes: list, alls=True, itemset_col_name: str = "itemsets"
) -> pd.DataFrame:
    """Get the set of itemsets that have the attributes in the input list (all or at least one)
    Args:
        df (pd.DataFrame): the input itemsets (with their info).
        attributes (list): list of itemset of interest
        alls (bool): If True, check if ALL attributes of the itemset are the input attributes.
        If False, check AT LEAST one attribute of the itemset is in the input attributes.
        itemset_col_name (str) : the name of the itemset column, "itemsets" as default
    Returns:
        pd.DataFrame: the set of itemsets (with their info)
    """

    return df.loc[
        df[itemset_col_name].apply(
            lambda x: attributes_in_itemset(x, attributes, alls=alls)
        )
    ]


def attributes_in_itemset(itemset, attributes, alls = True):
    """ Check if attributes are in the itemset (all or at least one)
    
    Args:
        itemset (frozenset): the itemset
        attributes (list): list of itemset of interest
        alls (bool): If True, check if ALL attributes of the itemset are the input attributes. 
        If False, check AT LEAST one attribute of the itemset is in the input attributes.
        
    """
    # Avoid returning the empty itemset (i.e., info of entire dataset)
    if itemset == frozenset() and attributes:
        return False
    
    for item in itemset:
        # Get the attribute
        attr_i = item.split("=")[0]
        
        #If True, check if ALL attributes of the itemset are the input attributes.
        if alls:
            # Check if the attribute is present. If not, the itemset is not admitted
            if attr_i not in attributes:
                return False
        else:
            # Check if least one attribute. If yes, return True
            if attr_i in attributes:
                return True
    if alls:
        # All attributes of the itemset are indeed admitted
        return True
    else:
        # Otherwise, it means that we find None
        return False
    
def filter_itemset_df_by_attributes(df: pd.DataFrame, attributes: list, alls = True, itemset_col_name: str = "itemsets") -> pd.DataFrame:
    """Get the set of itemsets that have the attributes in the input list (all or at least one)
    
    Args:
        df (pd.DataFrame): the input itemsets (with their info). 
        attributes (list): list of itemset of interest
        alls (bool): If True, check if ALL attributes of the itemset are the input attributes. 
        If False, check AT LEAST one attribute of the itemset is in the input attributes.
        itemset_col_name (str) : the name of the itemset column, "itemsets" as default
        
    Returns:
        pd.DataFrame: the set of itemsets (with their info)
    """

    return df.loc[df[itemset_col_name].apply(lambda x: attributes_in_itemset(x, attributes, alls = alls))]
    
    


def plotShapleyValue(
        itemset=None,
        shapley_values=None,
        sortedF=True,
        metric="",
        nameFig=None,
        saveFig=False,
        height=0.5,
        linewidth=0.8,
        sizeFig=(4, 3),
        labelsize=10,
        titlesize=10,
        title=None,
        abbreviations={},
        xlabel=False,
        show_figure=True,
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=sizeFig, dpi=100)

        if shapley_values is None and itemset is None:
            print("Error")
            return -1

        if shapley_values is None and itemset:
            shapley_values = self.computeShapleyValue(itemset)

        if abbreviations:
            shapley_values = abbreviateDict(shapley_values, abbreviations)

        sh_plt = {str(",".join(list(k))): v for k, v in shapley_values.items()}
        metric = f"{div_name}_{{{self.metric_name}}}" if metric is None else metric

        if sortedF:
            sh_plt = {k: v*100 for k, v in sorted(sh_plt.items(), key=lambda item: item[1])}

        ax.barh(
            range(len(sh_plt)),
            sh_plt.values(),
            height=height,
            align="center",
            color="C01",
            linewidth=linewidth,
            edgecolor="#0C4A5B",
        )
        ax.set_yticks(range(len(sh_plt)), minor=False)
        ax.set_yticklabels(list(sh_plt.keys()), minor=False)
        ax.tick_params(axis="y", labelsize=labelsize)
        ax.tick_params(axis="x", labelsize=labelsize)

        if xlabel:
            ax.set_xlabel(f"${div_name}({i_name}|{p_name})$", size=labelsize)

        title = "" if title is None else title
        title = f"{title} ${metric}$" if metric != "" else title  

        ax.set_title(title, fontsize=titlesize)
        if saveFig:
            nameFig = "./shap.pdf" if nameFig is None else nameFig
            plt.savefig(
                f"{nameFig}",
                bbox_inches="tight",
                #pad=0.05,
                facecolor="white",
                transparent=False,
            )
        if show_figure:
            plt.show()
            plt.close()


def order_by_key(d, order):
  return [d[k]*100 for k in sorted(order, key=order.get)]



def plotMultipleSV(
        shapley_values_1=None,
        shapley_values_2=None,
        sortedF=True,
        height=0.2,
        linewidth=0.8,
        sizeFig=(6,4),
        title=None,
        labelsize=10,
        titlesize=10,
        abbreviations={},
        xlabel=False,
        show_figure=True,
        saveFig=True,
        nameFig=None,
        legend=True,
        paper_exp=True,
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=sizeFig, dpi=100)

        if paper_exp:
            sh_plt_1 = {str(",".join(list(k))): v*100 for k, v in shapley_values_1.items()}
            sh_plt_2 = {str(",".join(list(k))): v*100 for k, v in shapley_values_2.items()}
        else:
            sh_plt_1 = {str(",".join(list(k))): v for k, v in shapley_values_1.items()}
            sh_plt_2 = {str(",".join(list(k))): v for k, v in shapley_values_2.items()}

        if sortedF:
            sh_plt_1 = {k: v for k, v in sorted(sh_plt_1.items(), key=lambda item: item[1])}

        ax.barh(
            np.arange(0.45,len(sh_plt_1)+0.45,1),
            sh_plt_1.values(),
            height=height,
            align="center",
            color="C01",
            linewidth=linewidth,
            edgecolor="#0C4A5B",
            label='w2v2-b to w2v2-l'
        )

        ax.barh(
            range(0,len(sh_plt_1),1),
            order_by_key(shapley_values_2, shapley_values_1),
            height=height,
            align="center",
            color="#83C4FA",
            hatch="//",
            linewidth=linewidth,
            edgecolor="#0C4A5B",
            label='w2v2-b to hub-b'
        )

        ax.set_yticks(np.arange(0.25,len(sh_plt_1)+0.25,1), minor=False)
        ax.set_yticklabels(sh_plt_1.keys(), minor=False)
        ax.tick_params(axis="y", labelsize=labelsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        
        if legend:
            ax.legend(fontsize=12)

        if xlabel:
            ax.set_xlabel(f"${div_name}({i_name}|{p_name})$", size=labelsize+2)

        title = "" if title is None else title

        ax.set_title(title, fontsize=titlesize)

        if saveFig:
            nameFig = "./shap.pdf" if nameFig is None else nameFig
            
            plt.savefig(
                f"{nameFig}",
                bbox_inches="tight",
                facecolor="white",
                transparent=False,
            )

        if show_figure:
            plt.show()
            plt.close()