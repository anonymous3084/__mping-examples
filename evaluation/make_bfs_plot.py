#!/usr/bin/env python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", help="Input file")
    parser.add_argument("--output_name", help="Output file name", required=True)
    parser.add_argument("--output_format", choices=['pdf', 'pgf', 'both'], default='pdf')
    args = parser.parse_args()
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 7,
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.preamble': "\n".join([
            r"\usepackage{xspace}",
            r"\newcommand{\xxxxx-3ing}{\textsc{xxxxx-5}\xspace}",
        ])
    })

    # remove extension from output_name
    output_name = Path(args.output_name).stem
    output_path = Path(args.output_name).parent
    if not Path(args.INPUT).exists():
        print(f"Input file {args.INPUT} does not exist.")
        sys.exit(1)


    # plt.style.use('tableau-colorblind10')
    # plt.style.use('seaborn-v0_8-colorblind')
    pt = 1 / 72
    fullwidth = 251 * pt
    df = pd.read_csv(args.INPUT, sep=',', header=0)
    df = df[(df.mpi_type == 'intel') & (df.iteration > 0)].sort_values('p')


    style = {
        'mpi': {'label': 'MPI', 'color': 'tab:blue', 'marker': 's'},
        'mpi_neighborhood': {'label': 'MPI neighbor', 'color': 'tab:orange', 'marker': 'o'},
        # 'mpl': {'label': 'MPL', 'color': 'tab:green', 'marker': 'v'},
        'xxxxx-3ing_flattened': {'label': r"\xxxxx-3ing", 'color': 'tab:red', 'marker': '^'},
        # 'rwth_mpi': {'label': 'RWTH-MPI', 'color': 'tab:purple', 'marker': '<'},
        'xxxxx-3ing_sparse': {'label': r"\xxxxx-3ing sparse", 'color': 'tab:brown', 'marker': '>'},
        'xxxxx-3ing_grid': {'label': r"\xxxxx-3ing grid", 'color': 'tab:pink', 'marker': 'x'}
    }
    # df['algorithm'] = df['algorithm'].apply(lambda x: style[x]['label'])

    hue_kws = {
        'color': [args['color'] for args in style.values()],
        'marker': [args['marker'] for args in style.values()],
        'markeredgecolor': [args['color'] for args in style.values()],
    }

    fg = sns.FacetGrid(df, col='graph', hue='algorithm',
                    col_order=['gnm-undirected_permute:false', 'rgg2d_permute:false',
                                # 'rgg3d_permute:false',
                                'rhg_permute:false'],
                    col_wrap=2,
                    hue_order=style.keys(),
                    hue_kws=hue_kws)
    fg.map(sns.lineplot, 'p', 'total_time',
        markersize=3,
        fillstyle='none',
        markeredgewidth=.5,
        linestyle='--',
        linewidth=.5,
        err_style='band',
        err_kws={'edgecolor': None},
        errorbar='sd')
    figure = fg.figure
    for ax in figure.axes:
        ax.set_yscale('log')
        ax.set_xscale('log', base=2)
        [i.set_linewidth(.1) for i in ax.spines.values()]
        ylim = ax.get_ylim()
        # print(ylim)
        # ax.set_ylim(ylim[0], 60)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.tick_params(axis="y",direction="in", which='both', width=.1)
        ax.tick_params(axis="x",direction="out", which='both', width=.1)
        ax.grid('both', linewidth=.5, alpha=.5, linestyle='dotted')
        ax.xaxis.set_major_locator(plt.LogLocator(base=2, numticks=5))
    # figure.axes[0].set_ylabel('Total time (s)')
    figure.set_size_inches(.75 * fullwidth, .7 * fullwidth)
    figure.axes[0].set_title('GNM', y=.75)
    figure.axes[1].set_title('RGG-2D', y=.75)
    # figure.axes[2].set_title('RGG-3D')
    figure.axes[2].set_title('RHG', y=.75)
    figure.supylabel('Total time (s)', x=-0.1)
    figure.supxlabel('number of processors', y=-0.06)
    new_labels = [style[algo]['label'] for algo in style.keys()]
    figure.legend(handles=figure.axes[0].lines, labels=new_labels, bbox_to_anchor=(1, .1), loc='lower right', handletextpad=.5)

    output_format = args.output_format
    if output_format == 'pdf' or output_format == 'both':
        figure.savefig(output_path / f'{output_name}.pdf', bbox_inches='tight')
    if output_format == 'pgf' or output_format == 'both':
        figure.savefig(output_path / f'{output_name}.pgf', bbox_inches='tight')

if __name__ == "__main__":
    main()
