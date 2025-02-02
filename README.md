# Financial Markets Simulation

This is a Python package which recreates a financial market via a Scale-Free network instantiated using the Barabasi-Albert algorithm (via the [networkx package](https://github.com/networkx/networkx)). The packages focuses on the analysis of the self-organized criticality in a simulated market compliant with the financial stylized facts and the Efficient Market Hypothesis. In particular, a price time series is created and wide price movements (i.e. avalanches) are studied in their duration, size and laminar time. The avalanches are extracted using the Wavelet Method as described in *"Self-organized criticality and stock market dynamics: an empirical study"* (2005, Bartolozzi et al.) and plotted in log-log scale, fitting a curve to understand the behavior of the distribution.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Repositoty structure


### Prerequisites

You will need Python 3.6 or higher, and the following packages:

* `networkx`
* `matplotlib`
* `scipy`
* `pywt`
* `powerlaw`

### Installing

You can install the package using pip:


