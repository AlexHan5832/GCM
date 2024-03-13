# -*- coding:utf-8 -*-
"""
    @Description:
    @Author: Yu Han
    @Date: 2023/05/22 15:29
    @Company: 
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def get_accuracy_from_csv(file_name: str, method_name: list = None) -> dict:
    """
    Get the accuracy from csv file
    :param method_name: method name
    :param file_name: csv name
    :return: accuracy is a dict, key is the category name, value is the accuracy
    """
    # read csv
    with open(file_name, "r") as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
        raw_data = rows[2:]

        all_accuracy = {}
        for single_data in raw_data:
            acc_name = single_data[0]

            if method_name is not None and acc_name not in method_name:
                continue

            illumination_accuracy = single_data[1:11]
            if illumination_accuracy[0] == "":
                break
            illumination_features = single_data[11]
            viewpoint_accuracy = single_data[13:23]
            viewpoint_features = single_data[23]
            overall_accuracy = single_data[25:35]
            overall_features = single_data[35]

            # str to float
            illumination_accuracy = list(map(float, illumination_accuracy))
            viewpoint_accuracy = list(map(float, viewpoint_accuracy))
            overall_accuracy = list(map(float, overall_accuracy))

            single_accuracy = {"illumination": illumination_accuracy,
                               "viewpoint": viewpoint_accuracy,
                               "overall": overall_accuracy,
                               "overall_features": overall_features}
            assert acc_name not in all_accuracy.keys(), "accuracy name must be unique"
            all_accuracy[acc_name] = single_accuracy

        return all_accuracy


def plot_accuracy(all_accuracy: dict, colors=None, linestyles=None) -> None:
    """
    Plot the accuracy of the three categories
    :param linestyles: linestyles is a list, each element is the linestyle of the corresponding model
    :param colors: colors is a list, each element is the color of the corresponding model
    :param all_accuracy: accuracy is a dict, key is the model name, value is the all accuracy
    :return: None
    """
    if linestyles is None:
        linestyles = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--', '-', '--']

    if colors is None:
        colors = ['black', 'black', 'orange', 'orange', 'red', 'red', 'blue', 'blue', 'purple', 'purple', 'green', 'green', 'pink', 'pink', 'magenta', 'magenta', "yellow", "yellow"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    idx = 0
    for name in all_accuracy.keys():
        single_accuracy = all_accuracy[name]
        # Get the accuracy of the three categories
        overall_accuracy = single_accuracy["overall"]
        illumination_accuracy = single_accuracy["illumination"]
        viewpoint_accuracy = single_accuracy["viewpoint"]
        features = single_accuracy["overall_features"]
        name = name + " #" + features
        # Plot the point plots on each of the three graphs
        ax[0].plot(np.arange(1, 11), overall_accuracy , marker='None', color=colors[idx],
                   linestyle=linestyles[idx])
        ax[1].plot(np.arange(1, 11), illumination_accuracy , marker='None', color=colors[idx],
                   linestyle=linestyles[idx])
        ax[2].plot(np.arange(1, 11), viewpoint_accuracy, marker='None', color=colors[idx],
                   linestyle=linestyles[idx], label=name)
        idx += 1

    ax[2].legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad=0.)

    ax[0].set_title("Overall")
    ax[1].set_title("Illumination")
    ax[2].set_title("Viewpoint")


    ax[0].set_xlabel("Pixel threshold")
    ax[1].set_xlabel("Pixel threshold")
    ax[2].set_xlabel("Pixel threshold")
    ax[0].set_ylabel("MMA")
    # ax[1].set_ylabel("MMA")
    # ax[2].set_ylabel("MMA")
    ax[0].set_ylim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[0].set_xticks(np.arange(1, 11))
    ax[1].set_xticks(np.arange(1, 11))
    ax[2].set_xticks(np.arange(1, 11))
    ax[0].yaxis.set_major_locator(MultipleLocator(0.1))
    ax[1].yaxis.set_major_locator(MultipleLocator(0.1))
    ax[2].yaxis.set_major_locator(MultipleLocator(0.1))
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    plt.subplots_adjust(right=0.7)
    plt.subplots_adjust(bottom=0.5)
    plt.show()


if __name__ == '__main__':
    accuracy = get_accuracy_from_csv(r"D:\hy\Pycharm\MtResearch\Results\paper_fig\figure.csv")

    plot_accuracy(accuracy)
