# -*- coding: utf-8 -*-
"""Plotting"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

def boxplot_models(accuracy):
    """
    Journal type boxplot

    :param accuracy: cross validation accuracies
    :return: boxplot
    """

    # Load style file
    plt.style.use('PaperDoubleFig.mplstyle')

    # Create a figure instance
    fig = plt.figure(1)

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Colors
    k = '#000000'
    r = '#D76D62'
    b = '#6d62d7'

    # Create the boxplot
    bp = ax.boxplot(accuracy, vert=False, showmeans=True, meanprops={"marker":"d","markerfacecolor":b})

    ## Custom x-axis labels
    labels = ['A','B','C','D','E','F','G','H','I','J']
    ax.set_yticklabels(labels[:len(accuracy)])

    ## Axes
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.yaxis.grid(color='grey', linestyle='-', dashes=(2, 7))
    ax.xaxis.set_minor_locator(MultipleLocator(0.002))
    ax.xaxis.set_tick_params(direction='in', which='both')
    ax.set_xlabel('Cross Validation Accuracy [%]')
    ax.set_ylabel('Selected Model')

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color=k, linewidth=1)

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color=k, linewidth=1)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color=k, linewidth=1)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color=r, linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color=r, alpha=0.5)

    # Save the figure
    fig.savefig('../figures/boxplot.pdf', dpi=300)

    return plt.show()