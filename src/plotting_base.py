import pickle

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
from util import *
import scipy
import plotly.graph_objects as go
from jinja2 import Template
from plotting_util import generate_xticks
from tex_plots import *


def par2_contours(df, x_name, y_name, z_names, zero_lvl="value", **kwargs):
    """
    Plot contours of a 2D parameter space. The first element must be the reference point.

    :param df: DataFrame containing the x, y, and z data.
    :param x_name: Name of the column in the DataFrame representing the x-axis data.
    :param y_name: Name of the column in the DataFrame representing the y-axis data.
    :param z_names: List of names of the columns in the DataFrame representing the z-axis data.
    :param zero_lvl: Method to normalize z data. Options: "value", "mean", "median", "0-ref". Default: "value".
    :param kwargs: Additional keyword arguments to pass to the plotting function.
    :return: Tuple containing the figure and axis of the contour plot.

    """
    x_data, y_data, = df[x_name].values, df[y_name].values

    x_data = vectorized_roundSF(x_data, 5)
    y_data = vectorized_roundSF(y_data, 5)

    ref_point = [x_data[0], y_data[0], 0]

    # drop the reference point
    x_data = x_data[1:]
    y_data = y_data[1:]

    # figure size
    fig, ax = plt.subplots()

    CPs = []
    hs = []

    # make a unique color and linestyle for each z_name
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    if len(z_names) > len(linestyles):
        raise ValueError('Too many z_names to plot. Not enough linestyles.')
    samples = len(np.unique(x_data))

    for iz, z_name in enumerate(z_names):
        z_data = df[z_name].values
        z_data = vectorized_roundSF(z_data, 5)
        ref_point[2] = z_data[0]

        z_data = z_data[1:]

        ax = plt.gca()

        # Create a grid to interpolate the data onto
        if 1 == 0:

            xi_lin, yi_lin = np.linspace(x_data.min(), x_data.max(), samples), np.linspace(y_data.min(),
                                                                                           y_data.max(),
                                                                                           samples)

            xi, yi = np.meshgrid(xi_lin, yi_lin)

            # Interpolate the z_data onto the grid
            zi = scipy.interpolate.griddata((x_data, y_data), z_data, (xi, yi),
                                            method='linear')
        else:
            # create the image by arranging the data in a grid
            # drop the reference point
            xi_lin, yi_lin = np.unique(x_data), np.unique(y_data)

            # convert the x_data and y_data to

            # sort the xi_lin and yi_lin
            xi_lin = np.sort(xi_lin)
            yi_lin = np.sort(yi_lin)

            xi, yi = np.meshgrid(xi_lin, yi_lin)
            zi = np.zeros((len(xi_lin), len(yi_lin)))

            for i, x in enumerate(xi_lin):
                for j, y in enumerate(yi_lin):
                    # print(x, y)
                    # print(z_data[(x_data == roundSF(x,5)) & (y_data == roundSF(y,5))])

                    found_element = z_data[(x_data == roundSF(
                        x, 5)) & (y_data == roundSF(y, 5))]
                    if len(found_element) == 0:
                        found_element = np.nan
                    zi[j, i] = found_element

        # Create the contour plot
        if zero_lvl == "value":
            max_val = np.max(
                np.abs([np.nanpercentile(zi, 5), np.nanpercentile(zi, 95)]))
            cmap_type = 'Blues'
            color_labl = "z value"

            vmin, vmax = np.nanpercentile(zi, 4), np.nanpercentile(zi, 96)
        elif zero_lvl == 'mean':
            zi = np.divide(zi - np.mean(zi), np.mean(zi))
            max_val = np.max(
                np.abs([np.nanpercentile(zi, 5), np.nanpercentile(zi, 95)]))
            cmap_type = 'RdBu_r'
            color_labl = "Fractional from mean point"
            vmin, vmax = -max_val, max_val
        elif zero_lvl == 'median':
            zi = np.divide(zi - np.median(zi), np.median(zi))
            max_val = np.max(
                np.abs([np.nanpercentile(zi, 5), np.nanpercentile(zi, 95)]))
            cmap_type = 'RdBu_r'
            color_labl = "Fractional from median point"
            vmin, vmax = -max_val, max_val
        elif zero_lvl == '0-ref':
            # using the reference point (the first point in the list)
            zi = np.divide(zi - ref_point[2], ref_point[2])
            max_val = np.max(
                np.abs([np.nanpercentile(zi, 5), np.nanpercentile(zi, 95)]))
            cmap_type = 'RdBu_r'
            color_labl = "Fractional difference from ref point"
            vmin, vmax = -max_val, max_val
        else:
            raise ValueError('zero_lvl not understood.')

        # set the x and y limits
        if 1 == 1:
            if len(z_names) == 1:

                im = plt.imshow(zi, cmap=cmap_type, origin='lower',
                                vmin=vmin,
                                vmax=vmax, alpha=.5)

                cbar = fig.colorbar(im)

                # Set the label of the colorbar
                cbar.set_label(str_latex(color_labl), fontsize=6)

                # make the colobar same size as the plot
                plt.gcf().subplots_adjust(bottom=0.3, top=.7)

                # set the x and y limits
                plt.xlim(0, samples - 2)
                plt.ylim(0, samples - 2)

            else:
                line1 = mlines.Line2D([], [], color='k', linewidth=0.5, linestyle=linestyles[iz], label=str_latex(
                    z_name))
                hs.append(line1)

        xTicks = generate_xticks(xi_lin)
        xTick_locs = np.interp(xTicks, xi_lin, range(xi_lin.size))
        plt.xticks(xTick_locs, xTicks, rotation=0)
        yTicks = generate_xticks(yi_lin)
        yTick_locs = np.interp(yTicks, yi_lin, range(yi_lin.size))
        plt.yticks(yTick_locs, yTicks, rotation=0)

        plt.xlabel(str_latex(x_name))
        plt.ylabel(str_latex(y_name))
        plt.title(str_latex(z_name + ' : ' + zero_lvl))

        if zero_lvl == '0-ref' or 1 == 1:
            # linear interp the index of x_data[0], y_data[0] on the grid
            x0 = np.interp(ref_point[0], xi_lin, range(xi_lin.size))
            y0 = np.interp(ref_point[1], yi_lin, range(yi_lin.size))

            plt.scatter(x0, y0, marker='*', color='k', s=20)

        if zero_lvl == '0-ref':
            levels = np.linspace(-.5, .5, 21)
        else:
            levels = np.linspace(np.min(vmin), np.max(vmax), 7)

        # levels = generate_xticks(min_value=vmin,max_value= vmax)

        levels = [roundSF(_, 3) for _ in levels]

        # check if the levels are all the same
        if 1 == 1:
            if np.all(levels == levels[0]):
                levels = None
            try:
                CPs.append(plt.contour(zi, levels=levels,
                           colors="k", linestyles=linestyles[iz]))
                # CPs.append(plt.contour(zi, colors="k", linestyles=linestyles[iz]))
                CPs[iz].clabel(fontsize=6, inline=1)
            except:
                pass  # no contours to plot
        return fig, ax


def histall(df, x, group_by=None, **kwargs):
    """
    Plot a histogram of all the columns of a 2D array.
    """

    fig, ax = plt.subplots()

    if group_by is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.hist(df[x].values, **kwargs)
        plt.xlabel(str_latex(x))
    else:
        group_unq = df[group_by].unique()

        # color each histogram based on the group_unq
        colors = plt.cm.jet(np.linspace(0, 1, len(group_unq)))
        colors = [tuple(color) for color in colors]

        for i, group in enumerate(group_unq):
            df_selection = df[df[group_by] == group]
            ax.hist(df_selection[x].values, histtype="step",
                    label=group, color=colors[i], **kwargs)
        plt.xlabel(str_latex(x))

        # add a colorbar to the right of the plot to show the group_unq
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=len(group_unq)))
        sm._A = []
        cbar = plt.colorbar(sm, ticks=np.linspace(
            0, len(group_unq), len(group_unq)))
        cbar.ax.set_yticklabels(np.round(group_unq, 2))

        plt.title("Grouped by: " + str_latex(group_by))
    plt.axis('tight')
    plt.grid(False)

    return fig, ax


def pareto_front(dfS, x, y):
    # Sort the DataFrame based on 'x' and then 'y'
    df_sorted = dfS.sort_values(by=[x, y])

    # Compute the cumulative minimum of 'y' to identify points on the Pareto front
    df_sorted[y+'_cummin'] = df_sorted[y].cummin()

    # The Pareto front points are those where 'y' is equal to its cumulative minimum
    pareto_df = df_sorted[df_sorted[y] == df_sorted[y+'_cummin']]

    # Drop the helper column before returning
    pareto_df = pareto_df.drop(columns=[y+'_cummin'])

    return pareto_df


def minYforBinnedX(dfS, x, y, num_bins=20):

    # find the min y for binned x. bin in log space
    dfS[x+'_bin'] = pd.cut(dfS[x], bins=np.logspace(np.log10(dfS[x].min()),
                           np.log10(dfS[x].max()), num_bins))
    min_y_for_each_bin = dfS.loc[dfS.groupby(x+'_bin')[y].idxmin()]

    # smooth the data using mean
    # min_y_for_each_bin[y] = min_y_for_each_bin[y].rolling(window=3).mean()

    return min_y_for_each_bin


def scatterGroupedColor(df, x, y, group_by=None, logx=True, logy=True, pareto=True, **kwargs):
    """
    Plot a histogram of all the columns of a 2D array.
    """

    fig, ax = plt.subplots(figsize=(4, 4))

    if group_by is None:
        fig, ax = plt.subplots()
        ax.scatter(df[x].values, df[y].values, **kwargs)
        plt.xlabel(str_latex(x))
    else:
        group_unq = df[group_by].unique()

        # color each histogram based on the group_unq

        if len(group_unq) > 10:
            colors = plt.cm.jet(np.linspace(0, 1, len(group_unq)))
            colors = [tuple(color) for color in colors]

        elif len(group_unq) > 5:
            colors = plt.cm.tab10(np.linspace(0, 1, len(group_unq)))
            colors = [tuple(color) for color in colors]

        elif len(group_unq) <= 3:
            colors = ["r", "b", "g"]

        for i, group in enumerate(group_unq):
            df_selection = df[df[group_by] == group]
            ax.scatter(df_selection[x].values, df_selection[y].values, label=group_by + ": " + str(group), color=colors[i], s=.2,
                       marker='o', **kwargs)

            # if pareto: # find the pareto front
            # sort the data by x
            dfs = df_selection[[x, y]].astype(float)
            # pareto_front_df = pareto_front(dfs, x, y)
            pareto_front_df = minYforBinnedX(dfs, x, y)

            ax.plot(pareto_front_df[x], pareto_front_df[y], color=colors[i], label=group_by + ": " + str(group) + (' '
                                                                                                                   'Minimum'),
                    lw=2)

        plt.legend(fontsize=8)

        set_latex_labels(ax, str_latex(x), str_latex(
            y), str_latex("Grouped by: " + str_latex(group_by)))

        if logx:
            plt.xscale("log")
        if logy:
            plt.yscale("log")

        # add a colorbar to the right of the plot to show the group_unq
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=len(group_unq)))
        sm._A = []
        # cbar = plt.colorbar(sm, ticks=np.linspace(0, len(group_unq), len(group_unq)))
        # cbar.ax.set_yticklabels(np.round(group_unq, 2))

    plt.axis('tight')

    return fig, ax


def stacked_hist(data0, numeric_variable, category_variable, num_bins=100, multi_plot=False, zmax=100,
                 max_category_bins=7,
                 numeric_variable_weights=None, histtype="step",
                 **kwargs):
    # trim the zmax percetnile of the data
    upper_bounds = np.nanpercentile(data0[numeric_variable], zmax)
    lower_bounds = np.nanpercentile(data0[numeric_variable], 0)

    data = data0[(data0[numeric_variable] >= lower_bounds) & (data0[numeric_variable] <= upper_bounds)].reset_index(
        drop=True)

    if numeric_variable_weights is None:
        data["WEIGHTS"] = 1
    else:
        data["WEIGHTS"] = data[numeric_variable_weights]

    if type(data[category_variable][0]) == str:
        data_type = 'str'
    else:
        data_type = 'num'
    cat_count = len(data[category_variable].unique())
    # if too many unqiue cats, then create binned version of the variable
    if cat_count > max_category_bins:
        data[category_variable +
             '_binned'] = pd.cut(data[category_variable], bins=max_category_bins)
        category_variable = category_variable + '_binned'

    grouped_data = data.groupby(category_variable, observed=True)[
        [numeric_variable, "WEIGHTS"]]

    # if numerical use jet, if categorical use tab20
    if data_type == 'str':
        # get tab20 colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(grouped_data)))
    else:
        colors = plt.cm.plasma(np.linspace(0, 1, len(grouped_data)))
        
    colors = [tuple(color) for color in colors]

    # Create a list of unique categories

    # Determine the number of subplots (rows)
    num_rows = len(grouped_data)

    if multi_plot:
        # Set the size of the entire figure (width, height)
        fig_width = 10
        fig_height = 5 * num_rows

        # make a separate subplot for each category, and plot the histogram on each
        fig, axes = plt.subplots(num_rows, 1, figsize=(
            fig_width, fig_height), sharex=True)
        for i, _ in enumerate(grouped_data):
            category, group = _
            category = roundSF(category, 2)
            group.plot.hist(ax=axes[i], label=category, color=colors[i],
                            alpha=1, bins=num_bins, histtype="step", lw=1)
            axes[i].set_title(f'{category_variable}: {category}')
            axes[i].set_ylabel('Frequency')

    else:
        fig, ax = plt.subplots()
        for i, _ in enumerate(grouped_data):
            category, group = _

            group[numeric_variable].plot.hist(ax=ax, label=category, alpha=.8, color=colors[i], bins=num_bins,
                                              lw=1, weights=group["WEIGHTS"], histtype=histtype, **kwargs)

    ax.set_xlabel(str_latex(numeric_variable))
    ax.set_ylabel("Count")
    ax.legend(fontsize=6)

    # grid off
    ax.grid(False)

    ax.set_title(str_latex(f'{category_variable.replace("_binned","")}'))
    return fig, ax


# sensitivity fractional analysis for each variable
def sensitivity_analysis(df, dfEX, var, ax=None, **kwargs):
    """
    Plot a sensitivity analysis of a parameter.
    """

    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(dfEX[var], **kwargs)
    ax.axvline(df[var].v, color='k', linestyle='dashed', linewidth=1)
    ax.set_xlabel(str_latex(var))
    return fig, ax


def scatter(df, x, y, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(df[x], df[y], **kwargs)
    ax.set_xlabel(str_latex(x))
    ax.set_ylabel(str_latex(y))
    return fig, ax


def plot(df, x, y, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(df[x], df[y], **kwargs)
    ax.set_xlabel(str_latex(x))
    ax.set_ylabel(str_latex(y))
    return fig, ax


def scatterColor(df, x, y, z, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(df[x], df[y], c=df[z])
    ax.set_xlabel(str_latex(x))
    ax.set_ylabel(str_latex(y))
    return fig, ax


def subselect_df_ND(df: pd.DataFrame, columns_to_exclude: list) -> pd.DataFrame:
    """
    Find rows in a DataFrame with identical values in all columns except the specified ones.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to search for identical rows.
    columns_to_exclude : list
        A list of column names to exclude from the comparison.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the rows with identical values in all columns except the specified ones.

    Example
    -------
    >>> data = {'A': [1, 2, 3, 4, 1], 'B': [5, 6, 7, 8, 5], 'C': [9, 10, 11, 12, 13]}
    >>> df = pd.DataFrame(data)
    >>> columns_to_exclude = ['C']
    >>> identical_rows = find_identical_rows_except_columns(df, columns_to_exclude)
    >>> print(identical_rows)
       A  B   C
    0  1  5   9
    4  1  5  13
    """

    # Make a copy of the DataFrame without the selected columns
    df_without_selected = df.drop(columns_to_exclude, axis=1)

    # Find duplicate rows, considering all columns except the selected ones
    duplicates = df_without_selected.duplicated(keep=False)

    # Get the rows with identical values in all columns except the selected ones
    identical_rows = df[duplicates]

    return identical_rows


def load_summary_data(data_folder):
    df = pd.read_csv(pjoin(data_folder, "summary.csv"))
    # create a dictionary for the head data (variable name, function, unit)
    if df.loc[0, "Data Type"] == "Function":
        col_data = df.iloc[[0, 1]]
    else:
        raise ValueError(
            "The summary.csv does not contain Function and Unit information.")

    # drop the "Function" and "Unit" rows
    df = df.drop([0, 1]).reset_index(drop=True)

    # convert all the data to float where possible
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass

    return df, col_data


def load_summary_data_fast(data_folder, metadata_rows=0, file_name="summary.csv"):
    file_path = pjoin(data_folder, file_name)

    # Load the CSV file skipping metadata rows
    df = pd.read_csv(file_path, skiprows=[
                     1, 2], header=0, engine='c', on_bad_lines='skip')
    print(df.head())
    # Read the metadata rows separately
    metadata = []
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if i < metadata_rows:
                metadata.append(line.strip().split(','))
            else:
                break

    summary_data = {"infs": np.sum(np.isinf(df['Energy System LCOE'].values)),
                    "nans": np.sum(np.isnan(df['Energy System LCOE'].values)),
                    "total": len(df['Energy System LCOE'].values)}

    # sort by _id
    df = df.sort_values(by=['_id'], ).reset_index(drop=True)

    if np.any(df['Energy System LCOE'] == np.inf):
        print("Summary Data")
        print(summary_data)
        print("DROPPING infs - errors likely")
        df = df[df['Energy System LCOE'] != np.inf]
    return df, metadata


def parallel_coordinates_plot(df, columnsToDisplay=None, colorBy=None, log_color=False, colorscale='Viridis', **kwargs):

    if colorBy is not None:
        columnsToDisplay = columnsToDisplay + [colorBy]

    df_filtered = df[columnsToDisplay].copy()

    if log_color:
        df_filtered[colorBy] = np.log10(df_filtered[colorBy])

    # Find categorical columns and convert them to numerical representations
    cat_columns = df_filtered.select_dtypes(
        include=['object', 'category']).columns
    cat_mappings = {}
    for col in cat_columns:
        df_filtered[col] = df_filtered[col].astype('category').cat.codes
        cat_mappings[col] = dict(
            enumerate(df[col].astype('category').cat.categories))

    if colorBy is not None:
        # Create a dictionary of line properties for each category
        linedict = dict(
            color=df_filtered[colorBy],
            colorscale=colorscale,

            showscale=True,
            cmin=df_filtered[colorBy].min(),
            cmax=df_filtered[colorBy].max(),
        )
    else:
        linedict = dict(
        )
    # Create the parallel coordinates plot
    fig = go.Figure(data=go.Parcoords(

        line=linedict,
        dimensions=[
            {
                'label': col,
                'values': df_filtered[col],
                'tickvals': list(cat_mappings[col].keys()) if col in cat_mappings else None,
                'ticktext': list(cat_mappings[col].values()) if col in cat_mappings else None,
            }
            for col in df_filtered.columns
        ]
    ))

    # Show the plot
    return fig


def sensitivity1d(data_folder, par=[], parz='Energy System LCOE', file_name="summary.csv"):
    """
    Standard routine for outputting the x and y and a sensitivity at the reference point, assume to be the first point.


    :param data_folder:
    :param par:
    :param parz:
    :param file_name:
    :return:
    """

    # par LUT
    par_LUT = {
        "Net Electrical Power Target (MWe)": "Net Electrical Power (MWe)"}

    for p in par:
        if p in par_LUT.keys():
            par[par.index(p)] = par_LUT[p]

    df, col_data = load_summary_data_fast(data_folder, file_name=file_name)

    parz_str = parz.replace("_", " ").replace("/", " per ").replace("$", "\$")
    parx_str = par[0].replace("_", " ").replace(
        "/", " per ").replace("$", "\$")

    # check what kind of variable par[0] is
    val = df[par[0]][0]
    xrefVal, yrefVal = df[par[0]][0], df[parz][0]

    if isinstance(val, str) or isinstance(val, bool) or isinstance(val, np.bool_):
        # categorical variable, return the x and y
        x = df[par[0]]
        y = (df[parz] - yrefVal) / yrefVal
        slope_at_xref_rel = None
    else:
        # convert the id to a number
        df['_id'] = df['_id'].astype(int)
        df = df.sort_values(by=['_id'])

        # remove duplicsates in par[0]
        df = df.drop_duplicates(
            subset=par[0], keep='first').reset_index(drop=True)

        if xrefVal == 0:
            norming_val = 1
        else:
            norming_val = xrefVal
        x = (df[par[0]] - xrefVal) / norming_val
        y = (df[parz] - yrefVal) / yrefVal

        xref_rel, yref_rel = x[0], y[0]

        print(x, y)

        # reorder x and y according to x
        y = y[x.argsort()]
        x = x[x.argsort()]

        # make interpolation function
        f = interp1d(x, y, kind='cubic')

        try:
            # small step size
            dx = 1e-6

            f = interp1d(x, y, kind='cubic')

            # use linear interpolation to find y values at x0-dx and x0+dx
            y_minus = np.interp(xref_rel - dx, x, y)
            y_plus = np.interp(xref_rel + dx, x, y)

            # calculate the slope
            slope_at_xref_rel = (y_plus - y_minus) / (2 * dx)

            print("Slope at uni xref: {0}".format(slope_at_xref_rel))

            # use cubic to find slope
            slope_at_xref_rel2 = (
                f(xref_rel + dx) - f(xref_rel - dx)) / (2 * dx)

            print("Slope at uni xref2: {0}".format(slope_at_xref_rel2))

        except:
            slope_at_xref_rel = 0

        df = df.sort_values(by=[par[0]]).reset_index(drop=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df[par[0]], df[parz], c="k", marker="o", linewidth=1)
        ax.set_ylabel("{0}".format(parz_str))
        ax.set_xlabel("{0}".format(parx_str))
        ax.plot(xrefVal, yrefVal, c="r", marker="o", linewidth=1)
        fig.savefig(pjoin(data_folder, "line{0}.png".format(parz_str)))

        fig2 = plt.figure()
        ax = fig2.add_subplot(111)
        ax.plot(x, y, c="k", marker="o", linewidth=1)
        x_space = np.linspace(x.min(), x.max(), 100)
        # ax.plot(x_space, f(x_space), c="r",linewidth=.5)
        ax.set_ylabel("Factor Diff {0}".format(parz_str))
        ax.set_xlabel("Factor Diff {0}".format(parx_str))
        ax.plot(xref_rel, yref_rel, c="r", marker="o", linewidth=1)
        fig2.savefig(pjoin(data_folder, "line_rel{0}.png".format(parz_str)))

    plt.close("all")

    # categorical vs numerical variable

    return {"x_normed": x, "y_normed": y,
            "x_ref": xrefVal, "y_ref": yrefVal,
            "x": df[par[0]], "y": df[parz],
            "slope": slope_at_xref_rel}


def basic_plot_set(df, par, parz, data_folder,df0=None):
    """
    Standard routine for plotting outputs from population of designs.

    :param data_folder:
    :param par: []
    :param parz:
    :param file_name:
    :return:
    """

    # change the parz str so that latex can be used
    
    parz_str = parz.replace("_", " ").replace("/", " per ")
    bins = max(min(25, len(df) // 20), 20)  # adaptive bin
    if 1==1:
        fig, ax = histall(df, parz, bins=bins, edgecolor='black',
                        linewidth=1, histtype="step")
        # Add vertical lines for the mean and median
        mean_val = df[parz].mean()
        median_val = df[parz].median()
        # Add vertical lines for the mean, median, and reference point
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=1, ymax=0.3)
        ax.text(mean_val, ax.get_ylim()[1]*0.32, 'Mean', rotation=90, color='red', fontsize=7, ha='center', va='bottom')

        ax.axvline(median_val, color='blue', linestyle='-', linewidth=1, ymax=0.1)
        ax.text(median_val, ax.get_ylim()[1]*0.12, 'Median', rotation=90, color='blue', fontsize=7, ha='center', va='bottom')

        if df0 is not None:
            ax.axvline(df0[parz][0], color='green', linestyle='-', linewidth=1, ymax=0.5)
            ax.text(df0[parz][0], ax.get_ylim()[1]*0.52, 'Ref', rotation=90, color='green', fontsize=7, ha='center', va='bottom')
        # Add a text box in the top right summarizing statistics
        stats_text = f'Mean: {roundSF(mean_val, 3)}\nMedian: {roundSF(df[parz].median(), 3)}\n' \
                    f'Mode: {roundSF(df[parz].mode()[0], 3)}\nStd: {roundSF(df[parz].std(), 3)}\n' \
                    f'Skew: {roundSF(df[parz].skew(), 3)}\nKurtosis: {roundSF(df[parz].kurtosis(), 3)}\n' \
                    f'N: {len(df)}'

        if df0 is not None:
            stats_text = f"Ref: {roundSF(df0[parz][0], 3)} \n" + stats_text

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
                va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))
        fig.savefig(pjoin(data_folder, "hist{0}.png".format(
            parz_str)), bbox_inches='tight')

    if len(par) == 1:
        fig, ax = plot(df, par[0], parz, c="k", marker="o", linewidth=1)
        fig.savefig(pjoin(data_folder, "line{0}.png".format(parz_str)))
    if len(par) == 2:
        fig, ax = par2_contours(df, par[0], par[1], [parz], zero_lvl="0-ref")
        fig.savefig(pjoin(data_folder, "contour{0}_ref_.png".format(
            parz_str)), bbox_inches='tight')
        fig, ax = par2_contours(df, par[0], par[1], [parz], zero_lvl="value")
        fig.savefig(pjoin(data_folder, "contour{0}_value_.png".format(
            parz_str)), bbox_inches='tight')

    if len(par) > 2:
        fig = parallel_coordinates_plot(
            df, columnsToDisplay=par, colorBy=parz, log_color=True)
        fig.write_html(pjoin(data_folder, "parallel{0}.html".format(parz_str)))

        for p in par:
            par_str = p.replace("/", "_per_")
            # print(p, parz)
            fig, ax = stacked_hist(df, parz, p, num_bins=bins, zmax=100)
            fig.savefig(pjoin(data_folder, "hist_stacked_{0}_{1}.png".format(
                parz_str, par_str)), bbox_inches='tight')
            plt.close("all")

    if len(par) == 2 or len(par) > 3:

        # make and save scatter plot of each par against parz and give it x and y labels, and a title
        for p in par:
            fig, ax = plt.subplots()
            par_str = p.replace("/", "_per_")
            ax.scatter(df[p], df[parz], s=.2, c='k', alpha=1)

            set_latex_labels(ax, str_latex(p), parz, '')
            # plt.yscale("log")
            # plt.xscale("log")
            plt.savefig(
                pjoin(data_folder, f"scatter{par_str}.png"), bbox_inches='tight')
            plt.close(fig)

        plt.close(fig)

    # close the figures
    plt.close("all")

    return


def str_list_to_float_array(str_list):
    """
    Convert a list of strings number tuples to a numpy array of floats
    :param str_list:
    :return:
    """

    str_list = [list(map(float, s.strip('[]').split(','))) for s in str_list]
    numpy_array = np.array(str_list)
    return numpy_array


def create_table(df, show_par, group_par, show_par2=None):
    # Create clusters for the labels
    labels = df[group_par].unique()

    new_df = pd.DataFrame(columns=labels)

    for label in labels:
        new_df[label] = df[df[group_par] ==
                           label][show_par].reset_index(drop=True)
        if show_par2 is not None:
            new_df[label] = new_df[label].astype(str) + " " + df[df[group_par] == label][show_par2].reset_index(
                drop=True).astype(str)
    new_df = new_df.replace(np.nan, '', regex=True)
    return new_df


def render_html_clean(pivot_df, file_loc):
    """
    Render the pivot table as an html table.

    :param pivot_df: should be grouped pivot
    :param file_loc:
    :return:
    """
    html_template = '''
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
        body {
            font-family: Arial, sans-serif;
            font-size: 8px;
            margin: 0;
            padding: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 25px;
            table-layout: fixed;
        }
        th, td {
            border: 1px solid #ddd;
            text-align: left;
            padding: 4px;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    </head>
    <body>
        <table>
            <thead>
                <tr>
                    {% for col in columns %}
                        <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in rows %}
                    <tr>
                        {% for cell in row %}
                            <td>{{ cell }}</td>
                        {% endfor %}
                    </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    '''

    # Render the HTML table using Jinja2
    template = Template(html_template)
    columns = pivot_df.columns
    rows = pivot_df.to_records(index=False)
    html_table = template.render(columns=columns, rows=rows)

    # Save the generated HTML table to a file
    with open(file_loc, 'w') as f:
        f.write(html_table)


def render_html_hover(df, file_loc):
    """
    Render the pivot table as an html table.

    :param pivot_df: should be grouped pivot
    :param file_loc:
    :return:
    """

    grouped = df.groupby('Function')

    # Define the HTML template with Jinja2
    html_template = '''
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
            font-size: 9px;
            }
            .container {
                display: flex;
                padding-left: 10%;
                padding-right: 10%;
            }
            .left {

                flex-basis: 20%;
            }
            }
            .right {
                            flex-basis: 60%;

            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 25px;
            }
            th, td {
                border: .5px solid #ddd;
                text-align: left;
                padding: 4px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;

            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #3399ff;
                color: white;
            }
            .hidden {
                display: none;
                position: absolute;
            }
        </style>
        <script>
            function showTable(id) {
                document.getElementById(id).style.display = 'block';
            }
            function hideTable(id) {
                document.getElementById(id).style.display = 'none';
            }
        </script>
    </head>
    <body>
        <div class="container">
            <div class="left">
                <table>
                    <thead>
                        <tr>
                            <th>Function</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for group in grouped %}
                            <tr onmouseover="showTable('{{ group[0] }}')" onmouseout="hideTable('{{ group[0] }}')">
                                <td>{{ group[0] }}</td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="right">
                {% for group in grouped %}
                    <table class="hidden" id="{{ group[0] }}">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Value</th>
                                <th>Unit</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in group[1].itertuples() %}
                                <tr>
                                    <td>{{ row.index }}</td>
                                    <td>{{ row.v }}</td>
                                    <td>{{ row.Unit  }}</td>
                                </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    '''

    # Render the HTML table using Jinja2
    template = Template(html_template)
    html_table = template.render(grouped=grouped)

    # Save the generated HTML table to a file
    with open(file_loc, 'w') as f:
        f.write(html_table)

    return


def create_diagram(df, show_par, group_par):
    # repalce : with - to make it a valid node name
    df[show_par] = df[show_par].str.replace(":", "-")
    df[group_par] = df[group_par].str.replace(":", "-")

    dot = Digraph(comment='Grouped Strings')
    dot.attr(compound='true')

    # Create clusters for the labels
    labels = df[group_par].unique()

    # connect the clusters
    for i in range(len(labels) - 1):
        dot.edge(labels[i], labels[i + 1], style='invis')

    for label in labels:
        with dot.subgraph(name=f'cluster_{group_par}') as c:
            c.attr(label=label, style='rounded, filled',
                   fillcolor='lightgrey', rankdir='TB')
            # Create nodes for the strings inside the clusters
            for _, row in df[df[group_par] == label].iterrows():
                # c.node(row[show_par], shape='ellipse')

                c.node(row[show_par], shape='ellipse', style='filled', fillcolor='white', color='black',
                       fontcolor='black')

    return dot


def find_first_inf(row):
    # first_inf is the column name of the first column that has an infinity
    for col_name, value in row.items():
        if value == np.inf:
            return col_name
    return None


# Apply the custom function to the entire DataFrame


def sensitivity1D_analysis(parent_dir):
    with open(pjoin(parent_dir, "sens_1d_data.pkl"), 'rb') as f:
        sens_data = pickle.load(f)

    print(sens_data)

    # sort sens_data by slope, with None to the end

    # for each numeric parameter, add a line to the plot
    simple_table = {}
    plt.figure()
    for par in sens_data.keys():
        p_data = sens_data[par]
        if p_data["slope"] is not None:
            simple_table[par] = {"slope": p_data["slope"]}
            parx_str = par.replace("_", " ").replace(
                "/", " per ").replace("$", "\$")

            plt.plot(p_data["x_normed"], p_data["y_normed"],
                     label=parx_str + ' slope: ' + str(p_data["slope"]))
        else:
            options = p_data["x_normed"]
            results = p_data["y_normed"]
            options_results = dict(zip(options, results))

            # add each option to the simple table
            for option in options_results.keys():
                simple_table[par + " " +
                             str(option)] = {"slope": options_results[option]}

    # small font
    plt.legend(fontsize=8)
    plt.xlim(-5, 5)
    plt.ylim(-1, 1)
    plt.xlabel("Parameter (normalized diff)")
    plt.ylabel("LCOE (normalized diff)")
    plt.savefig(pjoin(parent_dir, "sens_1d.png"), dpi=300)

    # save the simple table as df csv
    print(simple_table)

    df = pd.DataFrame.from_dict(simple_table, orient='index')
    df = df.sort_values(by=['slope'])
    df.to_csv(pjoin(parent_dir, "sens_1d_simple_table.csv"))
    print(df)

    return


if __name__ == "__main__":

    if 1 == 0:
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing3/"
        # data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_2_D_Average-Enrichment_Primary
        # -Operating-Pressure/"
        df = pd.read_csv(pjoin(data_folder, "summary.csv"))

        a = scatter(df, "Average Enrichment", 'Primary Operating Pressure')
        plt.savefig(pjoin(data_folder, "scatter.png"))

        a = scatterColor(df, "Average Enrichment",
                         'Primary Operating Pressure', 'Energy System LCOE')
        plt.savefig(pjoin(data_folder, "scatter.png"))

        a = histall(df, 'Energy System LCOE', "Average Enrichment", bins=20, edgecolor='black', linewidth=1.2,
                    histtype="step")
        plt.savefig(pjoin(data_folder, "hist.png"))

        a = par2_contours(df, "Average Enrichment",
                          'Primary Operating Pressure', ['Energy System LCOE'])
        plt.savefig(pjoin(data_folder, "contour.png"))

    if 1 == 1:
        # color GA results by interest rate or other variable
        data_folder = ('/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/230820_randomSampling_2interestRates'
                       '/GA_results_Full/')
        data_folder = '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing/GA_results_Full'
        # data_folder = ('/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/230819_PopSampling'
        #         '/GA_results_Full/')
        # /Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/230822_popsample_selection/GA_results_Full
        file_name = 'GA_results.csv'
        xs = ['Discharge Burnup [MWd/kg]', 'Core Energy Content (MWyr)',
              "RX power (MW)", "Refueling Interval",
              'Core Power Density [W/cc]',
              # "Core Energy Density (MWyr/ton)",
              "Fuel Power Density [W/cc]",
              'Primary Operating Pressure',
              "Core Volume",
              "Core Diameter (m)",
              ]
        y = 'Energy System LCOE'

        color = 'Debt Interest Rate'

        # make and save a scatter plot with points colored by interest rate as a categorical variable with a legend
        df, col_data = load_summary_data_fast(data_folder, file_name=file_name)
        if 1 == 0:
            df = df[df["Debt Interest Rate"] == 0.1]

        for x in xs:
            print(x)
            parx_str = x.replace("_", " ").replace("/", " per ")

            fig, ax = scatterGroupedColor(
                df, x, y, group_by=color, logx=True, logy=True)
            fig.savefig(pjoin(os.path.dirname(data_folder),
                        "scatter_color{0}.png".format(parx_str)),)

    if 1 == 0:  # 1D plot testing

        sensitivity1D_analysis(
            '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/')
        sensitivity1D_analysis(
            '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/productionNits-MacBook-Pro-3.local/20230531_HR22_M27_S59')
        # sensitivity1d('/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_1_D_Net-Electrical-Power-Target-(MWe)',
        #                ['Net Electrical Power Target (MWe)'], 'Energy System LCOE')

    if 1 == 0:  # 2D plot testing
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_2_D_TRISO-Volume-Packing" \
                      "-Fraction_Average-Enrichment"
        data_folder = "/Users/hans/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_2_D_TRISO-Volume-Packing" \
                      "-Fraction_Average-Enrichment"
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/production/20230425_HR19_M34_S33_2_D_TRISO" \
                      "-Volume-Packing-Fraction_Average-Enrichment"
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/production/20230425_HR20_M16_S22_2_D_Fuel" \
                      "-Compaction-Cost-[$_per_ccFuel]_Burnup-Perf-Factor"
        # data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_2_D_Net-Electrical-Power-Target
        # -(MWe)_Power-Rating-Perf-Factor"

        parzs = ['Energy System LCOE', 'O&M Cost [$/kWehr]', 'Fuel Cost [$/kWehr]', 'Capital Cost [$/kWehr]',
                 'Capital Cost ($/kWe)', 'Core Power Density [W/cc]', 'Core PSAR [W/cm2]', "RX power (MW)"]
        for parz in parzs:
            print(parz)
            basic_plot_set(data_folder, par=["TRISO Volume Packing Fraction", "Average Enrichment"],
                           parz=parz)
    if 1 == 0:  # 2D plot testing
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_2_D_TRISO-Volume-Packing" \
                      "-Fraction_Average-Enrichment"
        data_folder = "/Users/hans/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_2_D_TRISO-Volume-Packing" \
                      "-Fraction_Average-Enrichment"
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/production/20230425_HR19_M34_S33_2_D_TRISO" \
                      "-Volume-Packing-Fraction_Average-Enrichment"
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/productionMacBook-Pro-72.local" \
                      "/20230427_HR23_M32_S53/2_D_Fuel-Compaction-Cost-[$_per_ccFuel]_Burnup-Perf-Factor"
        # data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_2_D_Net-Electrical-Power-Target
        # -(MWe)_Power-Rating-Perf-Factor"

        parzs = ['Energy System LCOE',

                 ]
        for parz in parzs:
            print(parz)
            basic_plot_set(data_folder, par=["Fuel Compaction Cost [$/ccFuel]", "Burnup Perf Factor"],
                           parz=parz)

    if 1 == 0:  # GA Plot testing
        data_folder = '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/230531_randsampl/GA_Results_FULL'
        data_folder = '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing/GA_results_Full'

        # INDEPENDNT VARIABLES
        par = [
            'Safety Class',
            "TRISO Volume Packing Fraction",
            'Primary Operating Pressure',
            'Core Diameter (m)', "RX power (MW)", "Active Core Volume", "Core Energy Density [MJ/kg]",
            'Core Power Density [W/cc]',
            'Discharge Burnup [MWd/kg]',
            'Average Enrichment', 'Operating Model',  'TRISO kernel material',
            'Primary Maximum Fluid Velocity',
            'Net Electrical Power Target (MWe)']
        # DEPENDENT VARIABLES
        parzs = ['Energy System LCOE']

        for parz in parzs:
            print(parz)
            basic_plot_set(data_folder, par=par,
                           parz=parz,
                           file_name="GA_results.csv")

    if 1 == 0:  # ND plot tesing
        data_folder = '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/productionMacBook-Pro-72.local' \
                      '/20201201_HR16_M00_S00/12_D copy'
        data_folder = '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/productionMacBook-Pro-72.local' \
                      '/20230426_HR22_M51_S18/12_D'
        # INDEPENDNT VARIABLES
        #
        manipulated_Var = ['Primary Operating Pressure', 'Operating Model', 'Safety Class', 'Channel Count Factor',
                           'E-beam Welding', "Gen III+ or later", 'Net Electrical Power Target (MWe)',
                           'Average Enrichment', "TRISO Volume Packing Fraction", 'Core Diameter (m)',
                           'Reactor Aspect Ratio', 'Fuel Volume Fraction']
        par = [
            'Safety Class',
            "TRISO Volume Packing Fraction",
            'Primary Operating Pressure',
            'Core Diameter (m)',
            'RX power (MW)',
            'Core Power Density [W/cc]',
            'Core PSAR [W/cm2]',
            'Average Enrichment',
            "Active Core Volume",
            "Active Core Height (m)",
            "Net Electrical Power Target (MWe)",
        ]

        # DEPENDENT VARIABLES
        parzs = ['Energy System LCOE']
        for parz in parzs:
            print(parz)
            basic_plot_set(data_folder, par=par,
                           parz=parz)

    if 1 == 0:
        data_folder = '/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing_8_D'
        df, c = load_summary_data_fast(pjoin(data_folder))

        # how many infinities are there? in the LCOE?

        summary_data = {"infs": np.sum(np.isinf(df['Energy System LCOE'].values)),
                        "nans": np.sum(np.isnan(df['Energy System LCOE'].values)),
                        "total": len(df['Energy System LCOE'].values)}

    if 1 == 0:
        # test the html tables
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing/"
        data_folder = "/Users/nitwit/Dropbox/hans/Omega-13/OMEGA14DATA_OUT/testing/"
        df = pd.read_csv(pjoin(data_folder, "design_table.csv"),
                         converters={"v": convert_to_float})
        df["v"] = [format_float(_, 3, False) for _ in df["v"]]
        df["v"] = format_float(df["v"], 3, False)
        print(df["v"].values.tolist())

        # round all the numerical entries to 2 significant digits

        # for _ in df.iterrows():
        #     try:
        #         _["v"] = "{0:.3f}".format(_["v"])
        #     except:
        #         pass

        render_to_html3(df, pjoin(data_folder, "design_table3.html"))

        df["nameval"] = df["index"] + ":" + df["v"].astype(str)

        # diagram = create_diagram(df,show_par = "index",group_par = "Function")
        # diagram.render(pjoin( data_folder, "design_table"), view=True, format='png')

        table_grouped = create_table(
            df, show_par="nameval", group_par="Function")
        render_to_html2(table_grouped, pjoin(
            data_folder, "design_table_grouped2.html"))
        table_grouped.to_csv(
            pjoin(data_folder, "design_table_grouped.csv"), index=False)
