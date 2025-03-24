# ========================
# 🚀 PART 2: PARALLEL SIMULATION & VISUALIZATION (KUBEFLOW MPI)
# ========================

from mpi4py import MPI
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load environment variables
load_dotenv()
home = os.getenv("HOME")
DATA_PATH = os.path.join(home, "test-rwx", "historical_data")
PLOTS_DIR = os.path.join(home, "test-rwx", "plots")
CSV_OUTPUT = os.path.join(home, "test-rwx", "nasdaq_simulation_summary.csv")

# Ensure directories exist
if rank == 0:
    os.makedirs(PLOTS_DIR, exist_ok=True)
comm.Barrier()

# CONFIG
N_SIMULATIONS = 10000
N_DAYS = 252  # ~1 trading year

# Load list of symbols (broadcast from rank 0)
if rank == 0:
    nasdaq_df = pd.read_csv(os.path.join(home, "test-rwx", "nasdaq_companies.csv"))
    symbols = nasdaq_df['symbol'].tolist()
else:
    symbols = None
symbols = comm.bcast(symbols, root=0)

# Divide work
my_symbols = symbols[rank::size]

# Run Monte Carlo simulation for a symbol
def run_simulation(symbol):
    try:
        df = pd.read_csv(f"{DATA_PATH}/{symbol}.csv", parse_dates=["date"])
        prices = df['close']
        returns = prices.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        last_price = prices.iloc[-1]

        simulations = np.zeros((N_DAYS, N_SIMULATIONS))
        for i in range(N_SIMULATIONS):
            price = last_price
            for t in range(N_DAYS):
                price *= (1 + np.random.normal(mu, sigma))
                simulations[t, i] = price
        return simulations
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None

# Plot final price distribution
def plot_distribution(symbol, ending_prices):
    plt.figure(figsize=(10, 5))
    plt.hist(ending_prices, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
    plt.title(f"{symbol} - Final Price Distribution (1 Year)")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{symbol}_distribution.png")
    plt.close()

# Plot all simulation paths for a symbol
def plot_simulation_paths(symbol, simulations):
    plt.figure(figsize=(12, 6))
    for i in range(min(100, simulations.shape[1])):  # Limit to 100 paths
        plt.plot(simulations[:, i], linewidth=0.5, alpha=0.6)
    plt.title(f"{symbol} - Monte Carlo Simulation Paths (1 Year)")
    plt.xlabel("Trading Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{symbol}_paths.png")
    plt.close()

# Start timer
start_time = datetime.now()

# Simulate assigned symbols
results = []
final_prices = []
simulations_all = []

for symbol in my_symbols:
    print(f"[Rank {rank}] Simulating: {symbol}")
    sims = run_simulation(symbol)
    if sims is None:
        continue

    ending_prices = sims[-1, :]
    initial_price = sims[0, 0]
    expected_price = np.mean(ending_prices)
    expected_return = (expected_price - initial_price) / initial_price
    std_dev = np.std(ending_prices)

    results.append([symbol, initial_price, expected_price, expected_return, std_dev])
    final_prices.append(ending_prices)
    simulations_all.append(sims)

    plot_distribution(symbol, ending_prices)
    plot_simulation_paths(symbol, sims)

# Gather results to rank 0
gathered_results = comm.gather(results, root=0)
gathered_final_prices = comm.gather(final_prices, root=0)
gathered_simulations = comm.gather(simulations_all, root=0)

# Final processing at rank 0
if rank == 0:
    flat_results = [item for sublist in gathered_results for item in sublist]
    flat_final_prices = [item for sublist in gathered_final_prices for item in sublist]
    flat_simulations = [item for sublist in gathered_simulations for item in sublist]

    df = pd.DataFrame(flat_results, columns=["symbol", "initial_price", "expected_price", "expected_return_1yr", "std_dev_1yr"])
    df.sort_values(by="expected_return_1yr", ascending=False, inplace=True)
    df.to_csv(CSV_OUTPUT, index=False)

    # Portfolio distribution
    if flat_final_prices:
        portfolio_distribution = np.mean(flat_final_prices, axis=0)
        plt.figure(figsize=(12, 6))
        plt.hist(portfolio_distribution, bins=60, color='lightgreen', edgecolor='black', alpha=0.8)
        plt.title("Equal-Weighted NASDAQ Portfolio – Final Price Distribution (1 Year)")
        plt.xlabel("Portfolio Value (Average of All Stocks)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/nasdaq_portfolio_distribution.png")
        plt.close()

    # Simulated index paths
    if flat_simulations:
        aggregate_index = np.mean(flat_simulations, axis=0)
        plt.figure(figsize=(12, 6))
        for i in range(min(100, aggregate_index.shape[1])):
            plt.plot(aggregate_index[:, i], linewidth=0.5, alpha=0.6)
        plt.title("Simulated NASDAQ Index (Equal-Weighted, 1 Year)")
        plt.xlabel("Trading Days")
        plt.ylabel("Index Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/nasdaq_simulated_index_paths.png")
        plt.close()

    # Save run time to a file
    duration = datetime.now() - start_time
    runtime_str = f"Finished in: {duration}\n"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(PLOTS_DIR, f"runtime_{timestamp}.txt"), "w") as f:
        f.write(runtime_str)

    print(f"\n✅ All MPI simulations complete in {duration}. Output saved to {PLOTS_DIR}.")
