import numpy as np

def discretize(
    dfI,
    bins=3,
    attributes=None,
    strategy="quantile",
    adaptive=True,
    round_v=0,
    min_distinct = 10
):
    attributes = dfI.columns if attributes is None else attributes
    
    X_discretized = KBinsDiscretizer_continuos(
        dfI,
        attributes,
        bins=bins,
        strategy=strategy,
        adaptive=adaptive,
        round_v=round_v,
        min_distinct=min_distinct,
    )
    for attribute in dfI.columns:
        if attribute not in X_discretized:
            X_discretized[attribute] = dfI[attribute]
    return X_discretized

def KBinsDiscretizer_continuos(
    dt, attributes=None, bins=3, strategy="quantile", adaptive=True, round_v=0, min_distinct = 10
):
    
    def _get_edges(input_col, bins, round_v=0):
        from sklearn.preprocessing import KBinsDiscretizer

        est = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy=strategy)
        est.fit(input_col)
        # edges = [i.round() for i in est.bin_edges_][0]
        # edges = [int(i) for i in edges][1:-1]
        edges = [i for i in est.bin_edges_][0]
        edges = [round(i, round_v) for i in edges][1:-1]

        if len(set(edges)) != len(edges):
            edges = [
                edges[i]
                for i in range(0, len(edges))
                if len(edges) - 1 == i or edges[i] != edges[i + 1]
            ]
        return edges

    attributes = dt.columns if attributes is None else attributes
    continuous_attributes = [a for a in attributes if dt.dtypes[a] != object]
    X_discretize = dt[attributes].copy()
    for col in continuous_attributes:
        if len(dt[col].value_counts()) > min_distinct:

            if adaptive:
                msg = None
                found = False
                for increased in range(0, 5):
                    edges = _get_edges(dt[[col]], bins + increased, round_v=round_v)
                    if (len(edges) + 1) != bins:
                        msg = f"Not enough data in the bins for attribute {col}--> bin size is increased from {bins} to {bins+increased}"
                    else:
                        found = True
                        break
                if found == False:
                    edges = _get_edges(dt[[col]], bins, round_v=round_v)
                    msg = f"Not enough data in the bins & adaptive failed for attribute {col}. Discretized with lower #of bins ({len(edges)} vs {bins})"
                if msg:
                    import warnings

                    warnings.warn(msg)
            else:
                edges = _get_edges(dt[[col]], bins, round_v=round_v)

            for i in range(0, len(edges)):
                if i == 0:
                    data_idx = dt.loc[dt[col] <= edges[i]].index
                    X_discretize.loc[data_idx, col] = f"<={edges[i]:.{round_v}f}"
                if i == len(edges) - 1:
                    data_idx = dt.loc[dt[col] > edges[i]].index
                    X_discretize.loc[data_idx, col] = f">{edges[i]:.{round_v}f}"

                data_idx = dt.loc[
                    (dt[col] > edges[i - 1]) & (dt[col] <= edges[i])
                ].index
                X_discretize.loc[
                    data_idx, col
                ] = f"({edges[i-1]:.{round_v}f}-{edges[i]:.{round_v}f}]"
            ### IMPO: added check if no discretization is performed.
            # In this case, the attribute is dropped.
            if edges == []:
                import warnings

                msg = f"No discretization is performed for attribute '{col}'. The attribute {col} is removed. \nConsider changing the size of the bins or the strategy.'"
                warnings.warn(msg)
                X_discretize.drop(columns=[col], inplace=True)
        else:
            X_discretize[col] = X_discretize[col].astype("object")
    return X_discretize

