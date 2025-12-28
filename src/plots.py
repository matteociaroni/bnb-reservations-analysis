import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_monthly_average_metric(
        df: pd.DataFrame,
        value_column: str,
        months: list[int],
        month_labels: dict | list,
        *,
        title: str,
        y_label: str,
        ylim: tuple | None = None,
        grid: bool = True
) -> None:
    """
    Plot the average monthly value across all available years.

    Each month is averaged only over the years that have data.
    Months missing in some years are ignored in the average.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ['year', 'month', value_column].
    value_column : str
        Column to plot (e.g., 'occupied_nights', 'occupancy_pct').
    months : list[int]
        Months to plot (subset of 1..12).
    month_labels : dict or list
        Mapping month number -> label or list of labels.
    title : str
        Plot title.
    y_label : str
        Y-axis label.
    ylim : tuple, optional
        Y-axis limits (min, max).
    grid : bool, default True
        Show horizontal grid.
    """
    # step 1: aggregate per year e mese (sum o mean, dipende dalla metrica)
    per_year_month = df.groupby(["year", "month"])[value_column].mean().reset_index()

    # step 2: media su tutti gli anni per mese
    monthly_avg = per_year_month.groupby("month")[value_column].mean().reindex(months)

    x = np.arange(len(months))
    plt.figure(figsize=(12,6))
    plt.bar(x, monthly_avg)

    # handle month labels
    if isinstance(month_labels, dict):
        labels = [month_labels[m] for m in months]
    else:
        labels = month_labels

    plt.xticks(x, labels)
    plt.xlabel("Month")
    plt.ylabel(y_label)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if grid:
        plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_monthly_metric_by_year(
        df: pd.DataFrame,
        value_column: str,
        years: list[int],
        months: list[int],
        month_labels: dict | list,
        *,
        title: str,
        y_label: str,
        ylim: tuple | None = None,
        grid: bool = True
) -> None:
    """
    Plot a monthly metric with side-by-side bars grouped by year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ['year', 'month', value_column].
    value_column : str
        Column to plot (e.g., 'occupied_nights', 'occupancy_pct').
    years : list[int]
        Years to plot.
    months : list[int]
        Months to plot (subset of 1..12).
    month_labels : dict or list
        Mapping month number -> label or list of labels for the months.
    title : str
        Plot title.
    y_label : str
        Y-axis label.
    ylim : tuple, optional
        Y-axis limits (min, max).
    grid : bool, default True
        Show horizontal grid.
    """
    x = np.arange(len(months))
    bar_width = 0.8 / len(years)

    plt.figure(figsize=(12, 6))

    for i, year in enumerate(sorted(years)):
        # select year and reindex months, fill missing months with 0
        values = (
            df[df["year"] == year]
            .set_index("month")
            .reindex(months)[value_column]
            .fillna(0)
        )

        plt.bar(x + i * bar_width, values, width=bar_width, label=str(year))

    # handle month_labels as dict or list
    if isinstance(month_labels, dict):
        labels = [month_labels[m] for m in months]
    else:
        labels = month_labels

    plt.xticks(x + bar_width * (len(years) - 1) / 2, labels)
    plt.xlabel("Month")
    plt.ylabel(y_label)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)
    if grid:
        plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.legend(title="Year")
    plt.tight_layout()
    plt.show()


def build_day_of_year_heatmap(
        df_daily: pd.DataFrame,
        value_col: str,
        agg_func: str | callable
) -> pd.DataFrame:
    """
    Build a day-of-month x month heatmap matrix.

    Parameters
    ----------
    value_col : column to aggregate
    agg_func  : aggregation function (e.g. 'mean', 'nunique', callable)

    Returns
    -------
    DataFrame indexed by day, columns = months (1..12)
    """
    heatmap_df = (
        df_daily
        .groupby(["day", "month"])[value_col]
        .agg(agg_func)
        .unstack("month")
        .fillna(0)
    )

    # ensure month order
    months = list(range(1, 13))
    heatmap_df = heatmap_df.reindex(columns=months, fill_value=0)
    return heatmap_df

import seaborn as sns

def plot_calendar_heatmap(
        heatmap_df: pd.DataFrame,
        title: str,
        colorbar_label: str,
        cmap: str = "YlGnBu"
) -> None:
    """
    Plot a calendar-style heatmap (day x month).
    """
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    plt.figure(figsize=(15, 8))
    sns.heatmap(
        heatmap_df,
        cmap=cmap,
        cbar_kws={"label": colorbar_label}
    )
    plt.xlabel("Month")
    plt.ylabel("Day of month")
    plt.title(title)
    plt.xticks(
        ticks=[i + 0.5 for i in range(12)],
        labels=month_labels
    )
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def plot_daily_bookings_by_platform(df_daily, year: int):
    """
    Heatmap of daily bookings by platform for a given year.

    Parameters
    ----------
    df_daily : pd.DataFrame
        Daily bookings DataFrame with columns ['date', 'origin'].
    year : int
        Year to filter and plot.
    """
    df_year = df_daily[df_daily["date"].dt.year == year].copy()
    if df_year.empty:
        print(f"No data available for year {year}")
        return

    df_year["day"] = df_year["date"].dt.day
    df_year["month"] = df_year["date"].dt.month

    all_months = np.arange(1, 13)

    platforms = df_year["origin"].unique()
    palette = sns.color_palette("Set2", n_colors=len(platforms))
    color_mapping = dict(zip(platforms, palette))

    heatmap_origins = df_year.pivot_table(
        index="day",
        columns="month",
        values="origin",
        aggfunc="first"
    ).reindex(columns=all_months)

    heatmap_rgb = heatmap_origins.applymap(lambda x: color_mapping.get(x, (1, 1, 1)))

    fig, ax = plt.subplots(figsize=(15, 8))

    n_days = len(heatmap_rgb.index)
    n_months = len(all_months)

    cell_width = 1
    cell_height = 1

    for row_idx, day in enumerate(heatmap_rgb.index):
        for col_idx, month in enumerate(heatmap_rgb.columns):
            color = heatmap_rgb.loc[day, month]
            ax.add_patch(
                plt.Rectangle(
                    (col_idx * cell_width, row_idx * cell_height),
                    cell_width,
                    cell_height,
                    facecolor=color,
                    edgecolor="lightgray"
                )
            )

    ax.set_ylim(n_days, 0)
    ax.set_xlim(0, n_months)

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun", "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(np.arange(n_months) + 0.5)
    ax.set_xticklabels(month_labels)
    ax.set_yticks(np.arange(n_days) + 0.5)
    ax.set_yticklabels(heatmap_rgb.index)

    ax.set_xlabel("Month")
    ax.set_ylabel("Day")
    ax.set_title(f"Daily bookings by platform - {year}")

    patches = [mpatches.Patch(color=color_mapping[p], label=p) for p in platforms]
    ax.legend(handles=patches, title="Platform", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
