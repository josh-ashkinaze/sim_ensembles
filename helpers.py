import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import random 
import logging
import re
import os 
import chardet


random.seed(42)
np.random.seed(42)


#####################################################################
# DATA WRANGLING
#####################################################################

def clean_vars(s, how='title'):
    """
    Simple function to clean titles for plots

    Params
    - s (str): The string to clean
    - how (str, default='title'): How to return string. Can be either ['title', 'lowercase', 'uppercase']

    Returns
    - cleaned string
    """
    assert how in ['title', 'lowercase', 'uppercase'], "Bad option!! see docs"
    s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s)
    s = s.replace('_', ' ')
    if how == 'title':
        return s.title()
    elif how=='lower':
        return s.lower()
    elif how=='upper':
        return s.upper()


def read_csv_robust(file_path, sep=",", num_bytes=10000):
    """
    A function to robustly read in CSVs when they may contain different kinds of encoding errors

    Params:
        file_path (str): The file path
        sep (str): The string seperator
        num_bytes(int, default=10000): Reads in this sample to get encoding 

    Returns
        pandas df if success else None 
    """
    # Detect the file encoding
    def detect_encoding(file_path, num_bytes):
        with open(file_path, 'rb') as file:
            rawdata = file.read(num_bytes)
            result = chardet.detect(rawdata)
            return result['encoding']

    encoding_detected = detect_encoding(file_path, num_bytes)

    # Try reading the file with the detected encoding
    try:
        df = pd.read_csv(file_path, encoding=encoding_detected, on_bad_lines='skip', sep=sep)
        print(f"File read successfully with encoding: {encoding_detected}")
        return df
    except Exception as e:
        print(f"Failed to read with detected encoding {encoding_detected}. Error: {e}")

        # Fallback to UTF-8
        try:
            print("Attempting to read with UTF-8...")
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', sep=sep)
            print("File read successfully with UTF-8.")
            return df
        except Exception as e:
            print(f"Failed to read with UTF-8. Error: {e}")

            # Second fallback to ISO-8859-1
            try:
                print("Attempting to read with ISO-8859-1...")
                df = pd.read_csv(file_path, encoding='ISO-8859-1', on_bad_lines='skip', sep=sep)
                print("File read successfully with ISO-8859-1.")
                return df
            except Exception as e:
                print(f"Failed to read with ISO-8859-1. Error: {e}")
                raise ValueError("All attempts failed. Please check the file for issues beyond encoding.")



#####################################################################
# LATEX 
#####################################################################

def statsmodels2latex(model, beta_digits=2, se_digits=2, p_digits=3, ci_digits=2, print_sci_not=False):
    """
    This function summarizes the results from a fitted statistical model,
    printing a LaTeX formatted string for each parameter in the model that includes the beta coefficient,
    standard error, p-value, and 95% CI.
    
    Parameters:
    - model: A fitted statistical model with methods to extract parameters, standard errors,
             p-values, and confidence intervals.
    - beta_digits (default = 2): Number of decimal places for beta coefficients.
    - se_digits (default = 2): Number of decimal places for standard errors.
    - p_digits (default = 3): Number of decimal places for p-values.
    - ci_digits (default = 2): Number of decimal places for confidence intervals.
    - print_sci_not: Boolean to print very small p-values (p<0.001) in scientific notation or just write 'p<0.001'
    
    """
    
    summary_strs = []
    # Check if the necessary methods are available in the model
    if not all(hasattr(model, attr) for attr in ['params', 'bse', 'pvalues', 'conf_int']):
        raise ValueError("Model does not have the required methods (params, bse, pvalues, conf_int).")
    
    # Retrieve parameter estimates, standard errors, p-values, and confidence intervals
    params = model.params
    errors = model.bse
    pvalues = model.pvalues
    conf_int = model.conf_int()
    
    for param_name, beta in params.items():
        # Escape LaTeX special characters in parameter names
        safe_param_name = param_name.replace('_', '\\_')
        
        se = errors[param_name]
        p = pvalues[param_name]
        ci_lower, ci_upper = conf_int.loc[param_name]

        if p < 0.001:
            if print_sci_not:
                p_formatted = f"= {p:.2e}"
            else:
                p_formatted = f"<0.001"
        else:
            p_formatted = f"= {p:.{p_digits}f}"

        summary = (f"{safe_param_name}: $\\beta = {beta:.{beta_digits}f}$, "
                   f"$SE = {se:.{se_digits}f}$, $p {p_formatted}$, "
                   f"$95\\% CI = [{ci_lower:.{ci_digits}f}, {ci_upper:.{ci_digits}f}]$")
        print(summary)



def stargazer2latex(star, filename, add_ci=True, display_mod=False):
    """
    Function to process the Stargazer object and save the LaTeX output to a file.

    Params:
    - star: Stargazer object
    - filename: str, the path to save the LaTeX output
    - add_ci: bool, whether to add 95% CIs to the LaTeX output
    - display_mod: bool, whether to display the Stargazer object before saving the LaTeX output


    Example:
        add_words = smf.ols('ai_add_wc ~ IsConstrained*PromptType', data=df).fit()
        remove_words = smf.ols('ai_remove_wc ~ IsConstrained*PromptType', data=df).fit()
        star = Stargazer([add_words, remove_words]) 
        star.custom_columns(["Count of Added Words", "Count of Removed Words"])
        star.title("OLS regressions of additions and removals.") 
        stargazer2latex(star, "../tables/reg_auto_add_rem.tex")
    """
    
    print(f"Starting to process the Stargazer object for {filename}")
    if display_mod:
        display(star)
    base_title = star.title_text if star.title_text else "OLS regressions of additions and removals."
    
    
    # Handle CI and if so append this to title
    if add_ci:
        star.show_confidence_intervals(True)
    ci_string = " 95\% CIs in parentheses." if add_ci else ""
    star.title(base_title + ci_string)
    
    # Set table lable based on filename
    table_label = filename.split("/")[-1].replace(".tex", "")
    star.table_label = table_label
    
    # Stargazer adds "T." to factor variables which looks ugly so I remove these
    # Also, latex does not like underscores unless you're in math mode so remove too
    latex_content = star.render_latex().replace("_", "")
    latex_content = latex_content.replace("T.", "")
    
    with open(filename, "w") as tex_file:
        tex_file.write(latex_content)
    
    print(f"Processed LaTeX saved to {filename}")
    return star




#####################################################################
# PLOTTING
#####################################################################

def make_aesthetic(hex_color_list=None, with_gridlines=False, bold_title=False, save_transparent=False, font_scale=2):
    """Make Seaborn look clean and add space between title and plot"""
    
    # Note: To make some parts of title bold and others not:
    # plt.title(r$'\bf{bolded title}$\nAnd a non-bold subtitle')
    
    sns.set(style='white', context='paper', font_scale=font_scale)
    if not hex_color_list:
        hex_color_list = [
        "#826AED", # Medium slate blue
        "#D41876", # Telemagenta
        "#00A896", # Persian green,
        "#89DAFF", # Pale azure
        "#F7B2AD", # Melon
        "#342E37", # Dark grayish-purple
        "#7DCD85", # Emerald
        "#E87461", # Medium-bright orange
        "#E3B505", # Saffron
        "#2C3531", # Dark charcoal gray with a green undertone
        "#D4B2D8", # Pink lavender
        "#7E6551", # Coyote
        "#F45B69", # Vibrant pinkish-red
        "#020887", # Phthalo Blue
        "#F18805"  # Tangerine
        ]
    sns.set_palette(sns.color_palette(hex_color_list))
    try:
        plt.rcParams['font.family'] = 'Arial'
    except:
        pass
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.titlelocation'] = 'left'
    if bold_title:
        plt.rcParams['axes.titleweight'] = 'bold'
    else:
        plt.rcParams['axes.titleweight'] = 'regular'
    plt.rcParams['axes.grid'] = with_gridlines
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['savefig.transparent'] = save_transparent
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['axes.titlepad'] = 20*(font_scale/1)
    return hex_color_list

    


#####################################################################
# STATISTICS
#####################################################################


def pretty_print_desc_stats(data, n_bootstrap=1000, ci=False, ci_level=0.95, n_digits=2, seed=42):
    """
    Calculate descriptive statistics and print a LaTeX string in APA format.

    Args:
        data (array-like): Array of data to calculate statistics on.
        n_bootstrap (int, optional): Number of bootstrap samples. Default is 1000.
        ci (bool, optional): Whether to include confidence intervals. Default is False.
        ci_level (float, optional): Confidence interval level if ci is True. Default is 0.95.
        n_digits (int, optional): Number of digits to round the values to. Default is 2.
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        str: A formatted LaTeX string with the mean, median, and standard deviation,
             and optionally the confidence interval.
    
    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> print(pretty_print_desc_stats(data, 1000, False, 0.95, 2, 42))
        $M = 3.00, Mdn = 3.00, SD = 1.41$
    """
    data = np.array(data)
    
    # Calculate mean, median, and standard deviation
    mean = np.mean(data)
    median = np.median(data)
    sd = np.std(data, ddof=1)
    
    mean = round(mean, n_digits)
    median = round(median, n_digits)
    sd = round(sd, n_digits)
    
    if ci:
        bootstrap_results = bootstrap_mean(data, n_bootstrap, ci_level, seed)
        lower = round(bootstrap_results['lower'], n_digits)
        upper = round(bootstrap_results['upper'], n_digits)
        latex_string = f"$M = {mean}, Mdn = {median}, SD = {sd}, 95\\% \\text{{CI}} = [{lower}, {upper}]$"
    else:
        latex_string = f"$M = {mean}, Mdn = {median}, SD = {sd}$"
    
    return latex_string


def pretty_print_ci(stats_dict, n_digits=2):
    """
    Format the confidence interval statistics into a LaTeX-compatible string.

    Args:
        stats_dict (dict): Dictionary containing 'mean', 'lower', and 'upper' keys.
        n_digits (int, optional): Number of digits to round the values to. Default is 2.

    Returns:
        str: A formatted LaTeX string with the mean and confidence interval.
    
    Example:
        >>> stats = {'mean': 1.23456, 'lower': 0.98765, 'upper': 1.54321}
        >>> print(pretty_print_ci(stats, 2))
        $M = 1.23, 95\\% \\text{CI} = [0.99, 1.54]$
    """
    mean = round(stats_dict['mean'], n_digits)
    lower = round(stats_dict['lower'], n_digits)
    upper = round(stats_dict['upper'], n_digits)
    
    latex_string = f"$M = {mean}, 95\\% \\text{{CI}} = [{lower}, {upper}]$"
    return latex_string


def bootstrap_mean(array, n_bootstrap=1000, ci=95, seed=42):
    """
    Generate bootstrap confidence interval for the mean of the input data.
    
    Args:
        array: The input data array.
        n_bootstrap: The number of bootstrap samples to generate. Default is 1000.
        ci: The confidence interval percentage. Default is 95%.
        seed: The random seed for reproducibility. Default is 416.
    
    Returns:
        A dict with keys 'mean', 'lower', and 'upper'
    """
    np.random.seed(seed)  
    bootstrap_means = np.array([
        np.mean(np.random.choice(array, size=len(array), replace=True)) for _ in range(n_bootstrap)
    ])
    data_mean = np.mean(array)
    
    lower_bound = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return {'mean': data_mean, 'lower': lower_bound, 'upper': upper_bound}



def bootstrap_statistic(df, group, val, func='mean', n_bootstrap=5000, ci=0.95, seed=42, digits=2, pretty_print=False):
    """
    Perform bootstrapping on a given dataframe and compute a statistic.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group (str or None): Column name to group by. If None, bootstrap on the entire dataframe.
        val (str): Column name of the values to bootstrap.
        func (str or function): Statistic function to apply to each bootstrap sample. 
                                Can be 'mean', 'median', 'sd', 'var', 'min', 'max', 'sum' or a custom function.
        n_bootstrap (int, optional): Number of bootstrap samples. Default is 5000.
        ci (float, default=0.95): Confidence interval.
        seed (int, default=42): Random seed for reproducibility
        digits (int, default=2): Number of digits if print rounded string for latex
        pretty_print (boolean, default=False): If True then print results as latex string

    Returns:
        dict: A dictionary containing the sample mean, lower and upper confidence intervals, 
              and the bootstrapped distribution. If `group` is not None, returns a dictionary 
              of such dictionaries keyed by group values.

    Example:
        >>> df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'val': [1, 2, 3, 4]})
        >>> result = bootstrap_statistic(df, 'group', 'val', 'mean')
        >>> print(result)
        {'A': {'mean': 1.5, 'lower': 1.0, 'upper': 2.0, 'dist': array([1., 1., 2., ..., 1., 2., 2.])},
         'B': {'mean': 3.5, 'lower': 3.0, 'upper': 4.0, 'dist': array([4., 3., 3., ..., 3., 3., 4.])}}
    """
    
    def bootstrap_sample(data, func, n_bootstrap):
        n = len(data)
        samples = np.random.choice(data, size=(n_bootstrap, n), replace=True)
        stats = np.apply_along_axis(func, 1, samples)
        return stats

    # Map string functions to numpy functions
    func_map = {
        'mean': np.mean,
        'median': np.median,
        'sd': np.std,
        'var': np.var,
        'min': np.min,
        'max': np.max,
        'sum': np.sum
    }

    if isinstance(func, str):
        func = func_map.get(func)
        if func is None:
            raise ValueError("Function string not recognized. Only support strings if they are 'mean', 'median', 'sd', 'var', 'min', 'max', or 'sum'.")

    np.random.seed(seed)

    if group is not None:
        results = {}
        groups = df[group].unique()
        for g in groups:
            data = df[df[group] == g][val].values
            stats = bootstrap_sample(data, func, n_bootstrap)
            mean = np.mean(stats)
            lower = np.percentile(stats, (1 - ci) / 2 * 100)
            upper = np.percentile(stats, (1 + ci) / 2 * 100)
            results[g] = {'mean': mean, 'lower': lower, 'upper': upper, 'dist': stats}
        return results
    else:
        data = df[val].values
        stats = bootstrap_sample(data, func, n_bootstrap)
        mean = np.mean(stats)
        lower = np.percentile(stats, (1 - ci) / 2 * 100)
        upper = np.percentile(stats, (1 + ci) / 2 * 100)
        res = {'mean': mean, 'lower': lower, 'upper': upper, 'dist': stats}
        if pretty_print:
            pretty_print_ci(res)
        return res


