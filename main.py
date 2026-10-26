import numpy as np
import matplotlib.pyplot as plt
from BlackScholes import price_call, calc_delta
from DataStructures import HedgeResult, SimulationConfig, IntervalResult
from PriceGenerator import generate_prices
from scipy.stats import gaussian_kde

# init_price = float(input("Enter initial price of stock:"))
# str_price = float(input("Enter strike price of stock:"))
# exp_years = float(input("Enter number of years until option expiry:"))
# volatility = float(input("Enter the volatility as a float:"))
# int_rate = float(input("Enter the interest rate as a float:"))
# step_num = int(input("Enter the number of steps for each stock pricing:"))
# path_num = int(input("Enter the number of paths to test against:"))
# random_seed = int(input("Enter the random_seed to use"))

comma_separated_intervals = [1, 2, 5, 10, 20, 25, 50, 100, 250, 500] # [int(x) for x in input("Enter a comma separated list of intervals to test").split(",")]

# config = SimulationConfig(init_price, str_price, exp_years, volatility, int_rate, step_num, path_num, random_seed)

config = SimulationConfig(100, 100, 0.5, 0.2, 0.03, 500 , 10000 , 1)

rng = np.random.default_rng(config.random_seed)
paths = np.array([generate_prices(config,rng) for x in range(config.number_of_paths)])
print("Done generating paths")
outputs = {}
call_price = price_call(config)
dt = config.expiry_years / config.number_of_steps
## Main loop
for interval in comma_separated_intervals:
    outputs[interval] = []
    for i in range(config.number_of_paths):
        trade_count = 0
        shares_held = 0
        cash = call_price
        delta = calc_delta(config, paths[i][0], 0)
        # initial hedge
        to_trade = delta - shares_held
        if to_trade != 0:
            trade_count += 1
        shares_held += to_trade
        cash -= (to_trade*paths[i][0])
        for j in range(1,config.number_of_steps):
            
            cash *= np.exp(config.interest_rate * dt)
            if j% interval == 0:
                delta = calc_delta(config, paths[i][j], j)
                to_trade = delta - shares_held
                if to_trade != 0:
                    trade_count += 1
                shares_held += to_trade
                cash -= (to_trade*paths[i][j])
            
        # Expiry
        cash *= np.exp(config.interest_rate * dt)
        cash += shares_held * paths[i][config.number_of_steps]
        trade_count += 1
        shares_held = 0
        cash -= max(paths[i][config.number_of_steps] - config.strike_price, 0)
        final_error = cash
        attempt_output = HedgeResult(final_error,trade_count)
        outputs[interval].append(attempt_output)

results = {}
for interval in comma_separated_intervals:
    hedge_result_array = np.array(outputs[interval])
    final_error_array = np.array([x.final_error for x in hedge_result_array])
    average_error = np.mean(final_error_array)
    standard_dev = np.std(final_error_array)
    rmse = np.sqrt(np.mean(np.square(final_error_array)))
    fifth_percentile = np.percentile(final_error_array, 5)
    average_tc = np.mean([y.trade_count for y in hedge_result_array])
    results[interval] = IntervalResult(interval, average_error, rmse, standard_dev, fifth_percentile, average_tc, final_error_array)

for i in results:
    print(f"Interval {i}:")
    print("The average error: " + str(results[i].mean))
    print("The RMSE: " + str(results[i].rmse))
    print("The standard deviation: " + str(results[i].std))
    print("The 5th percentile: " + str(results[i].perc5))
    print("The trade count average: " + str(results[i].average_trade_count))

# Plotting time

# RMSE over intervals
x = comma_separated_intervals
y = [results[i].rmse for i in results]
fig1, ax1 = plt.subplots()
ax1.plot(x,y, marker="o")
ax1.set_xscale("log")
ax1.set_xticks(x, labels=[str(k) for k in x])
ax1.set_xlabel("Rebalancing every k simulation steps (log scale)")
ax1.set_ylabel("Root mean squared error (£ per option)")
fig1.savefig("plot1.png")

#Final error distributions
fig2, ax2 = plt.subplots()

selected = [1,10,25,100,500]
all_errors = np.concatenate([results[k].error_array for k in selected])
x = np.linspace(all_errors.min(), all_errors.max(), 1000)

for k in selected:
    density = gaussian_kde(results[k].error_array)
    ax2.plot(x, density(x), label=f"Every {k} steps")

ax2.axvline(0, color="black", linestyle="--")
ax2.set_xlabel("Final hedging error (£)")
ax2.set_ylabel("Probability density")
ax2.legend()
fig2.savefig("plot2.png")

# 5th percentile
x = comma_separated_intervals
y = [results[i].perc5 for i in results]
fig3, ax3 = plt.subplots()
ax3.plot(x,y, marker="o")
ax3.set_xscale("log")
ax3.set_xticks(x, labels=[str(k) for k in x])
ax3.set_xlabel("Rebalancing every k simulation steps (log scale)")
ax3.set_ylabel("5th percentile of final error (£/option)")
fig3.savefig("plot3.png")

plt.show()