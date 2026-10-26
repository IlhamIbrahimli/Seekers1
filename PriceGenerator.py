import numpy as np
def generate_prices(config: SimulationConfig, rng: np.Random.Generator) -> np.ndarray:
    output = []
    output.append(config.initial_price)
    v = config.volatility
    r = config.interest_rate # Drift
    dt = config.expiry_years / config.number_of_steps
    for i in range(config.number_of_steps):
        prev_price = output[-1]
        z = rng.standard_normal()

        new_price = prev_price * np.exp((r - ((v**2)/2)) * dt + v*np.sqrt(dt)*z)
        output.append(new_price)
    return np.array(output)



