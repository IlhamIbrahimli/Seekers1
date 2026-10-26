from dataclasses import dataclass, field

@dataclass(frozen=True)
class SimulationConfig:
    initial_price: float
    strike_price: float
    expiry_years: float
    volatility: float
    interest_rate: float
    number_of_steps: int
    number_of_paths: int
    random_seed: int

@dataclass
class HedgeResult:
    final_error: float
    trade_count: int
    #portfolio_history: list[dict[str, float]]

@dataclass
class IntervalResult:
    interval: int
    mean: float
    rmse: float
    std: float
    perc5: float
    average_trade_count: int
    error_array: np.ndarray