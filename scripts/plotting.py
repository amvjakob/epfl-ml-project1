# -*- coding: utf-8 -*-
"""Plotting"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator, LogFormatter)
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def boxplot_models(accuracy, save=False):
    """
    Journal type boxplot

    :param accuracy: cross validation accuracies
    :param save: save the figure
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
    ax.xaxis.set_minor_locator(MultipleLocator(0.4))
    ax.xaxis.set_tick_params(direction='in', which='both')
    ax.set_xlabel('Cross Validation Accuracy [%]', labelpad=10)
    ax.set_ylabel('Selected Model', labelpad=10)

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
    if save:
        fig.savefig('../figures/boxplot.pdf', dpi=300)

    return plt.show()

def surface3d_model(degrees, lambdas, accuracies, ytl, save=False):
    """
    Journal type Surface Plot

    :param degrees: polynomial expansion
    :param lambdas: hyperparameters tested
    :param accuracies: results
    :param ytl: YAxis Tick Labels for log lambda
    :param save: save the figure
    :return: surface plot
    """

    # Load style file
    plt.style.use('PaperDoubleFig.mplstyle')

    # Create fig and ax instances
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(degrees, lambdas, accuracies, cmap=cm.rainbow)

    # Beauty aspects
    ax.yaxis.grid(color='grey', linestyle='-', dashes=(2, 7))
    ax.zaxis.set_minor_locator(MultipleLocator(0.2))
    ax.zaxis.set_major_locator(MultipleLocator(1))
    ax.zaxis.set_tick_params(direction='in', which='both')
    ax.set_ylabel('Hyperparameter $\lambda$',labelpad=15)
    ax.set_xlabel('Degree $d$', labelpad=10)
    ax.set_zlabel('Accuracy [%]',labelpad=5)
    #ax.yaxis.set_ticklabels(ytl)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.4, aspect=10, ticks=[81.8, 82.2, 82.6, 83], spacing='proportional')

    # elevation and angle
    ax.view_init(15, 50)

    # Save the figure
    if save:
        fig.savefig('../figures/surfaceplot.pdf', dpi=300)

    return plt.show()