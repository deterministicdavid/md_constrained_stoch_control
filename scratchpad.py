import numpy as np
import matplotlib.pyplot as plt

def plot_eta_to_n(eta=0.5, n_min=1, n_max=100):
    """
    Plots n -> eta^n on a log-log scale.
    """
    if not (0 < eta < 1):
        raise ValueError("eta must be strictly between 0 and 1.")

    # Generate n values using logspace for even distribution on a log scale
    # (Starting from 1 because log(0) is undefined)
    n_values = np.logspace(np.log10(n_min), np.log10(n_max), num=500)
    
    # Calculate y values
    y_values = eta ** n_values

    # Initialize the plot
    plt.figure(figsize=(8, 6))
    
    # Create the log-log plot
    plt.loglog(n_values, y_values, label=rf'$\eta = {eta}$', color='#1f77b4', linewidth=2)

    # Formatting and labels
    plt.title(rf'Log-Log Plot of $n \mapsto \eta^n$ for $\eta \in (0, 1)$')
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\eta^n$')
    
    # Add grid lines for both major and minor ticks
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    
    # Display the plot
    plt.savefig('scratchpad.png')
    plt.close()

if __name__ == "__main__":
    # You can easily test different values of eta here
    plot_eta_to_n(eta=0.99, n_max=100)