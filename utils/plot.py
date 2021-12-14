import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


COLORS = list(colors.TABLEAU_COLORS.values())


def plot(df: pd.DataFrame, ranks: pd.DataFrame, title="", annotate=False):

    _, axs = plt.subplots()
    
    axs.set_title(title)
    
    for i in np.unique(ranks):
        index = df[ranks["rank"] == i]
        index.plot(
            x="PerdasT",
            y="Mativa",
            kind="scatter",
            marker="o",
            ax=axs,
            label=f"rank {i}",
            color=COLORS[i%len(COLORS)]
        )
        if annotate:
            frame = index[["PerdasT", "Mativa"]]
            for k, v in frame.iterrows():
                # import ipdb; ipdb.set_trace()
                axs.annotate(k, v)
    

__all__ = ["plot"]