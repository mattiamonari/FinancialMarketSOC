import matplotlib.pyplot as plt
import numpy as np
from wavelet import compute_pdf
from scipy.stats import norm
import powerlaw
import networkx as nx
from matplotlib import animation


__all__ = ['plot_returns', 'plot_market_price', 'plot_ratio_buyers_sellers', 'plot_weighted_volumes', 'plot_returns_vs_time', 
           'plot_returns_distribution', 'plot_original_vs_filtered_log_returns_pdf', 'plot_avalanches_on_log_returns', 'plot_avalanche_sizes',
           'histogram_log_bins', 'draw_3d_network']

def plot_returns_vs_time(returns, saveFig=False, squared=False):
    plt.plot(returns)
    plt.xlabel('Time Steps')
    plt.ylabel('Returns')
    plt.title('Price Returns')

    if squared:
        plt.title('Squared Price Returns')
    
    if saveFig:
        plt.savefig('images/returns_vs_time' + ('_squared_' if squared else '') + '.pdf')
        plt.cla()
    else:
        plt.show()

def plot_returns_distribution(returns, fitted_dist=False, saveFig=False, log_returns=False, squared=False):
    
    plt.hist(returns, bins=30,density=True, alpha=0.6, color='g')
    plt.xlabel('Returns')
    plt.ylabel('Probability Density')
    title = 'Log Returns' if log_returns else 'Returns' + (' Squared' if squared else '')

    if fitted_dist:
        mu, std = norm.fit(returns)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 200)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        if log_returns:
            plt.title(title + ": mu = %.2f,  std = %.2f" % (mu, std))
        else:
            plt.title(title + ": mu = %.2f,  std = %.2f" % (mu, std))

    plt.title(title)

    if saveFig:
        plt.savefig('images/' + ('log_' if log_returns else '') + ('squared_' if squared else '') + 'returns_distribution.pdf')
        plt.cla()
    else:
        plt.show()

def plot_returns(returns, saveFig=False, squared=False, log_returns=False):
    plot_returns_vs_time(returns, saveFig, squared)
    plot_returns_distribution(returns, fitted_dist=True, saveFig=saveFig, log_returns=log_returns, squared=squared)

def plot_market_price(prices, moving_avg, profiler_view=False, saveFig=False, num_features=None, labeled_array=None):
    if profiler_view:
        plt.subplot(2, 1, 1)
    else :
        plt.figure(figsize=(12, 6))

    plt.plot(prices, label='Market Price', color='black')
    plt.plot(moving_avg, label='Simple Moving Aerage - 200 days', color='red', alpha=0.6)

    if num_features is not None:
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            plt.axvspan(indices[0], indices[-1], color='red', alpha=0.3, label="Avalanche" if i == 1 else None)

    plt.xlabel('Time Steps')
    plt.ylabel('Market Price')
    plt.title('Market Price Evolution')
    plt.legend()

    if saveFig:
        plt.savefig('images/market_price.pdf')
        plt.cla()
    elif not profiler_view:
        plt.show()
        plt.cla()

def plot_ratio_buyers_sellers(ratio, profiler_view=False, saveFig=False,  num_features=None, labeled_array=None):
    if profiler_view:
        plt.subplot(2, 1, 2)
    else :
        plt.figure(figsize=(8, 6))
    plt.plot(ratio, label='Buyers/Sellers', color='green')

    if num_features is not None:
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            plt.axvspan(indices[0], indices[-1], color='red', alpha=0.3, label="Avalanche" if i == 1 else None)

    plt.xlabel('Time Steps')
    plt.ylabel('Ratio')
    plt.title('Buyers/Sellers Ratio')
    plt.legend()
    plt.tight_layout()

    if saveFig:
        plt.savefig('images/ratio_buyers_sellers.pdf')
        plt.cla()
    elif not profiler_view:
        plt.show()

def plot_weighted_volumes(weighted_volumes, profiler_view=False, saveFig=False):
    if profiler_view:
        plt.subplot(3, 1, 3)
    else:
        plt.figure(figsize=(8, 6))
    
    plt.plot(weighted_volumes)
    plt.ylim(0, 1)
    plt.xlabel('Time Steps')
    plt.ylabel('Volumes Proportion (Buy)')
    
    if saveFig:
        plt.savefig('images/weighted_volumes.pdf')
        plt.cla()
    else: # In this case since this is the last plot we show it even if profileview is True
        plt.show()

def plot_original_vs_filtered_log_returns_pdf(log_returns, filtered_log_returns, bins=50, fit_gaussian_filtered = False, saveFig=False):
    original_bin_centers, original_pdf = compute_pdf(log_returns, bins=bins)
    filtered_bin_centers, filtered_pdf = compute_pdf(filtered_log_returns, bins=bins)

    if fit_gaussian_filtered:
        # Fit a Gaussian to the filtered log returns
        mu_filtered, std_filtered = norm.fit(filtered_log_returns)
        gaussian_x = np.linspace(min(log_returns), max(log_returns), 500)
        filtered_gaussian = norm.pdf(gaussian_x, mu_filtered, std_filtered)
        plt.plot(gaussian_x, filtered_gaussian, '--', label="Gaussian Fit (Filtered)", color="green", alpha=0.8)

    # Plot original log returns PDF
    plt.plot(original_bin_centers, original_pdf, '^', label="Original Log Returns", alpha=0.7)

    # Plot filtered log returns PDF
    plt.plot(filtered_bin_centers, filtered_pdf, 'o', label="Filtered Log Returns", alpha=0.7)

    plt.yscale("log")  # Logarithmic scale for better visibility of tails
    plt.xlabel("Log Returns")
    plt.ylabel("Probability Density")
    plt.title("PDF of Logarithmic Returns Before and After Filtering")
    plt.legend()
    plt.grid(alpha=0.3)

    if saveFig:
        plt.savefig('images/original_vs_filtered_log_returns.pdf')
        plt.cla()
    else:
        plt.show()

def plot_avalanches_on_log_returns(log_returns, residual_signal, filtered_log_returns, labeled_array, num_features, saveFig=False):
    plt.figure(figsize=(12, 6))
    plt.plot(log_returns, label="Original Log Returns", alpha=0.6)
    plt.plot(residual_signal, label="Residual Signal", alpha=0.8)
    plt.plot(filtered_log_returns, label="Filtered Signal", alpha=0.8)
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        plt.axvspan(indices[0], indices[-1], color='red', alpha=0.3, label="Avalanche" if i == 1 else None)
    plt.legend()
    plt.title("Avalanches in Log Returns (Residual Signal)")
    plt.xlabel("Time Steps")
    plt.ylabel("Log Returns")

    if saveFig:
        plt.savefig('images/avalanches_on_log_returns.pdf')
        plt.cla()
    else:
        plt.show()  

def histogram_log_bins(x, x_min=None, x_max=None, num_of_bins=20, min_hits=1):
    """
    Generate histogram with logarithmically spaced bins.
    """
    if not x_min:
        x_min = np.min(x)
    if not x_max:
        x_max = np.max(x)

    # This is the factor that each subsequent bin is larger than the next.
    growth_factor = (x_max / x_min) ** (1 / (num_of_bins + 1))
    # Generates logarithmically spaced points from x_min to x_max.
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), num=num_of_bins + 1)
    # We don't need the second argument (which are again the bin edges).
    # It's conventional to denote arguments you don't intend to use with _.
    bin_counts, _ = np.histogram(x, bins=bin_edges)
    total_hits = np.sum(bin_counts)
    bin_counts = bin_counts.astype(float)

    # Rescale bin counts by their relative sizes.
    significant_bins = []
    for bin_index in range(np.size(bin_counts)):
        if bin_counts[bin_index] >= min_hits:
            significant_bins.append(bin_index)

        bin_counts[bin_index] = bin_counts[bin_index] / (growth_factor ** bin_index)

    # Is there a better way to get the center of a bin on logarithmic axis? There probably is, please figure it out.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # You can optionally rescale the counts by total_hits if you want to get a density.
    return bin_counts[significant_bins], bin_centers[significant_bins], total_hits

def draw_network(num_nodes=50, edge_funds=3, random_trader_ratio=0.25):

    G = nx.barabasi_albert_graph(num_nodes, edge_funds)
    num_random_traders = int(num_nodes * random_trader_ratio)
    random_trader_indices = np.random.choice(range(edge_funds, num_nodes), size=num_random_traders, replace=False)

    edge_funds_nodes = []
    random_traders_nodes = []
    traders_nodes = []
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True) 
    for idx, (node, degree) in enumerate(sorted_nodes):
        if node < edge_funds:
            G.nodes[node]['type'] = 'hedge_fund'
            G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold
            G.nodes[node]['trade_size'] = np.random.uniform(1, 2)  # Random trade size
            edge_funds_nodes.append(node)
        elif node in random_trader_indices:
            G.nodes[node]['type'] = 'random_trader'
            G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)  # Random trade size
            random_traders_nodes.append(node)
        else:
            G.nodes[node]['type'] = 'trader'
            G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold (not used at the moment)
            G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)  # Random trade size
            traders_nodes.append(node)

        G.nodes[node]['last_update_time'] = 0
        G.nodes[node]['position'] = 'buy' if np.random.random() < 0.5 else 'sell'


    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    nx.draw(G, pos, node_size=10, edge_color="gray", alpha=0.15, with_labels=False)

    # we highlight the hedge funds (high-degree nodes)
    nx.draw_networkx_nodes(G, pos, nodelist=edge_funds_nodes, node_size=50, node_color=colors[0], label="Hedge Funds")
    nx.draw_networkx_nodes(G, pos, nodelist=random_traders_nodes, node_size=50, node_color=colors[1], label="Random Traders")
    nx.draw_networkx_nodes(G, pos, nodelist=traders_nodes, node_size=50, node_color=colors[2], label="Traders")

    plt.title("Scale-Free Network (Barabási–Albert Model)")
    plt.legend(loc="best")
    plt.show()

def draw_3d_network(num_nodes=100, edge_funds=3, random_trader_ratio=0.25):
    G = nx.barabasi_albert_graph(num_nodes, edge_funds)
    num_random_traders = int(num_nodes * random_trader_ratio)
    random_trader_indices = np.random.choice(range(edge_funds, num_nodes), size=num_random_traders, replace=False)
    color_map = plt.cm.copper(np.linspace(0, 1, 3))
    
    edge_funds_nodes = []
    random_traders_nodes = []
    traders_nodes = []
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True) 
    for (node, degree) in enumerate(sorted_nodes):
        if node < edge_funds:
            G.nodes[node]['type'] = 'hedge_fund'
            edge_funds_nodes.append(node)
        elif node in random_trader_indices:
            G.nodes[node]['type'] = 'random_trader'
            random_traders_nodes.append(node)
        else:
            G.nodes[node]['type'] = 'trader'
            traders_nodes.append(node)
    
    pos = nx.spring_layout(G, dim=3, seed=42, scale=7)
    nodes = np.array([pos[v] for v in G])
    edges = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    
    colors = []
    size = []
    for node in G.nodes():
        if node in edge_funds_nodes:
            colors.append(color_map[0])
            size.append(75)
        elif node in random_traders_nodes:
            colors.append(color_map[1])
            size.append(25)
        else:
            colors.append(color_map[2])
            size.append(25)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def init():
        ax.scatter(*nodes.T, alpha=0.8, s=size, c=colors)
        for vizedge in edges:
            ax.plot(*vizedge.T, color='gray', alpha=0.15)
        ax.grid(False)
        ax.set_axis_off()
        plt.tight_layout()
        return
    
    def _frame_update(index):
        ax.view_init(index * 0.2, index * 0.5)
        return
    
    ani = animation.FuncAnimation(
        fig, _frame_update, init_func=init, interval=50, cache_frame_data=False, frames=90
    )
    ani.save('images/3d_network.gif', fps=30, dpi=300)


def plot_curve_exponential(x_fit, y_fit, x_plot, y_plot, bin_centers, counts, 
                           lam_fit, lam_fit_err, xlabel, title, savefig=False, 
                           name=None):
    plt.figure(figsize=(8, 5))

    # Plot the binned data
    plt.scatter(bin_centers, counts, s=40, color='blue', label="Log-binned data")

    # Highlight the region we used for the fit
    plt.scatter(x_fit, y_fit, s=60, color='orange', edgecolor='k',
                zorder=3, label="Data used for fit")

    # Overlay the exponential fit
    plt.plot(x_plot, y_plot, 'r--', 
             label=f"Exponential fit\nlambda={lam_fit:.3f} ± {lam_fit_err:.3f}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel("Density (counts / bin_width, log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(f"Images/{name}.pdf")
    plt.show()


def plot_curve_power_law(x_fit_line, y_fit_line, bin_centers, counts, fit_mask, 
                         alpha, alpha_err, xlabel, title, savefig=False, name=None):
    plt.figure(figsize=(8, 6))

    # 1) Plot all data (log-binned)
    plt.scatter(
        bin_centers, counts,
        color='blue', s=30, label='Log-binned Data'
    )

    # 2) Highlight the portion used for fitting
    plt.scatter(
        bin_centers[fit_mask], counts[fit_mask],
        color='orange', edgecolors='k', s=60, zorder=3, label='Points used for fit'
    )

    # 3) Plot the best-fit line
    label_fit = (
        f'alpha = {alpha:.3f}±{alpha_err:.3f}'
    )
    plt.plot(x_fit_line, y_fit_line, 'r--', label=label_fit)

    # Log scales
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel("Density (counts / bin width)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savefig:
        plt.savefig(f"Images/{name}.pdf")
    plt.show()
