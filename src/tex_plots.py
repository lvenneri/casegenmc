import re
import numpy as np

def setup_tex_plots(fontsize=12, usetex=True, style='default',texfonts=True,figsize=[8.0, 8.0]):
    """
    e.g. setup_text_plots(fontsize=14, usetex=True,style='default')

    :param fontsize:
    :param usetex:
    :param style:
    :return: setup for nice plotting
    """
    import matplotlib

    matplotlib.style.use(style)
    matplotlib.rcParams['savefig.dpi'] = 500
    matplotlib.rc('legend', fontsize=fontsize, handlelength=3)
    matplotlib.rc('axes', titlesize=fontsize)
    matplotlib.rc('axes', titlesize=fontsize)
    matplotlib.rc('axes', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)

    # use latex font
    if usetex:
        matplotlib.rcParams['text.usetex'] = True

    matplotlib.rc('font', size=fontsize, family='serif',
                  style='normal', variant='normal',
                  stretch='normal', weight='normal',
                  serif=['Computer Modern'])

    # default scatter plot marker size
    matplotlib.rcParams['lines.markersize'] = 5
    # default scatter plot line width
    matplotlib.rcParams['lines.linewidth'] = 1
    matplotlib.rcParams['grid.color'] = 'k'
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.linestyle'] = ':'
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['figure.figsize'] = figsize
    matplotlib.rcParams['errorbar.capsize'] = 3
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True
    matplotlib.rcParams['image.interpolation'] = 'nearest'
    matplotlib.rcParams['image.resample'] = False
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['mathtext.rm'] = 'serif'
    matplotlib.rc('lines', linewidth=1.0, color='k')
    matplotlib.rc('legend', fancybox=False, shadow=False, framealpha=1, borderpad=0, frameon=True)

    return


def str_latex(s):
    latex_replacements = {
        "&": r"\&",
        "_": r"\_",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "^": r"\^",
        "~": r"\~",
        "<": r"\<",
        ">": r"\>",
        "|": r"\|",
        " ": r"\ ",
        "\n": r"\n",
    }

    if True:
        for char, replacement in latex_replacements.items():
            s = s.replace(char, replacement)

    return s


def set_latex_labels(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(str_latex(xlabel))
    if ylabel:
        ax.set_ylabel(str_latex(ylabel))
    if title:
        ax.set_title(str_latex(title))

def _latex_sci_notation(number, sig_figs=2):
    sci_notation_str = '{:.{sig_figs}e}'.format(number, sig_figs=sig_figs)
    match = re.match(r"([+-]?\d\.\d{0,10})e([+-]?\d+)", sci_notation_str)
    base, exponent = match.groups()
    latex_str = r'${} \times 10^{{{}}}$'.format(base, exponent)
    return latex_str

def latex_sci_notation(numbers, sig_figs=2):
    if np.isscalar(numbers):
        return _latex_sci_notation(numbers, sig_figs)
    else:
        return [_latex_sci_notation(n, sig_figs) for n in numbers]
