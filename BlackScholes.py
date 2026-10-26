import numpy as np
from scipy.stats import norm
def price_call(config: SimulationConfig) -> float:
    d1 = (np.log(config.initial_price/config.strike_price) + (config.interest_rate + ((config.volatility**2)/2))*config.expiry_years)/ (config.volatility* np.sqrt(config.expiry_years))
    d2 = d1 - (config.volatility* np.sqrt(config.expiry_years))
    call = config.initial_price * norm.cdf(d1) - config.strike_price*np.exp(-config.interest_rate*config.expiry_years)*norm.cdf(d2)
    return call

def calc_delta(config: SimulationConfig, curr_price: float, curr_step: int) -> float:
    time_to_expiry = config.expiry_years - ((config.expiry_years/config.number_of_steps) * curr_step) # Curr step must then start from 0 index
    d1 = (np.log(curr_price/config.strike_price) + (config.interest_rate + ((config.volatility**2)/2))*time_to_expiry)/ (config.volatility* np.sqrt(time_to_expiry))
    delta  = norm.cdf(d1)
    return delta