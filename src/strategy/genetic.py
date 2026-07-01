"""
Genetic Algorithm Strategy Optimizer
=======================================
DEAP-based GA for evolving trading strategy parameters.

Each "genome" encodes a strategy configuration:
- Which indicators to use
- Indicator parameters (e.g., SMA period, RSI threshold)
- Entry/exit conditions

Fitness = Sharpe × sqrt(profit_factor) / (1 + max_drawdown)

Uses island model parallelism: multiple populations evolve
independently with periodic migration for diversity.

Usage:
    from src.strategy.genetic import GeneticOptimizer

    optimizer = GeneticOptimizer(
        strategy_class=EMACrossStrategy,
        param_space={
            "fast_period": (5, 50),
            "slow_period": (20, 200),
        },
    )
    best_params, best_fitness = optimizer.evolve(price_data, generations=100)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Type

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_sharpe, compute_max_drawdown, compute_profit_factor
from src.core.logger import get_logger
from src.strategy.templates import BaseStrategy, StrategyParams

logger = get_logger("strategy.genetic")


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm."""
    population_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 5
    elite_pct: float = 0.1
    n_islands: int = 1
    migration_interval: int = 10
    migration_size: int = 5
    random_seed: int = 42


@dataclass
class Individual:
    """A single genome in the population."""
    params: dict[str, Any]
    fitness: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    generation: int = 0


@dataclass
class EvolutionResult:
    """Results from a GA evolution run."""
    best_individual: Individual
    best_fitness_history: list[float]
    mean_fitness_history: list[float]
    all_individuals: list[Individual]
    generations: int
    total_evaluations: int


class GeneticOptimizer:
    """
    DEAP-inspired genetic algorithm for strategy parameter optimization.

    Evolves a population of strategy configurations to maximize
    risk-adjusted returns.

    Args:
        strategy_class: The strategy class to optimize.
        param_space: Dict of parameter_name → (min_val, max_val) or list of choices.
        fitness_fn: Custom fitness function. Default uses Sharpe × sqrt(PF) / (1 + DD).
        config: GA configuration (population size, generations, etc.).
    """

    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_space: dict[str, tuple | list],
        fitness_fn: Callable[[pd.Series, list[float]], float] | None = None,
        config: GAConfig | None = None,
    ) -> None:
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.fitness_fn = fitness_fn or self._default_fitness
        self.config = config or GAConfig()

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

    def _default_fitness(self, returns: pd.Series, trade_pnls: list[float]) -> float:
        """
        Default fitness function: Sharpe × sqrt(profit_factor) / (1 + max_drawdown).

        Penalizes high drawdown and rewards consistent profitability.
        """
        if len(returns) < 20 or len(trade_pnls) < 5:
            return -1.0

        sharpe = compute_sharpe(returns)
        pf = compute_profit_factor(trade_pnls)
        equity = (1 + returns).cumprod()
        max_dd, _ = compute_max_drawdown(equity)

        # Clamp values
        sharpe = max(sharpe, -5.0)
        pf = max(min(pf, 10.0), 0.0)  # Cap extreme PF
        max_dd = abs(max_dd)

        fitness = sharpe * np.sqrt(pf) / (1.0 + max_dd)

        # Penalize too few trades
        if len(trade_pnls) < 10:
            fitness *= 0.5

        return float(fitness)

    def _random_individual(self, generation: int = 0) -> Individual:
        """Create a random individual from the parameter space."""
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, tuple) and len(space) == 2:
                lo, hi = space
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = random.randint(lo, hi)
                else:
                    params[name] = random.uniform(lo, hi)
            elif isinstance(space, list):
                params[name] = random.choice(space)
            else:
                params[name] = space

        return Individual(params=params, generation=generation)

    def _crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """Uniform crossover: randomly swap genes between parents."""
        child1_params = {}
        child2_params = {}

        for key in self.param_space:
            if random.random() < 0.5:
                child1_params[key] = parent1.params.get(key)
                child2_params[key] = parent2.params.get(key)
            else:
                child1_params[key] = parent2.params.get(key)
                child2_params[key] = parent1.params.get(key)

        return Individual(params=child1_params), Individual(params=child2_params)

    def _mutate(self, individual: Individual) -> Individual:
        """Mutate: randomly perturb one gene."""
        params = individual.params.copy()
        gene = random.choice(list(self.param_space.keys()))
        space = self.param_space[gene]

        if isinstance(space, tuple) and len(space) == 2:
            lo, hi = space
            current = params[gene]
            range_size = hi - lo

            if isinstance(lo, int) and isinstance(hi, int):
                delta = random.randint(-max(1, int(range_size * 0.2)), max(1, int(range_size * 0.2)))
                params[gene] = max(lo, min(hi, current + delta))
            else:
                delta = random.gauss(0, range_size * 0.1)
                params[gene] = max(lo, min(hi, current + delta))
        elif isinstance(space, list):
            params[gene] = random.choice(space)

        return Individual(params=params)

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """Tournament selection: pick the best from a random subset."""
        candidates = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(candidates, key=lambda x: x.fitness)

    def _evaluate_individual(
        self,
        individual: Individual,
        price_data: pd.DataFrame,
    ) -> Individual:
        """Evaluate an individual's fitness by backtesting its strategy."""
        try:
            strategy_params = StrategyParams(
                name=f"ga_{self.strategy_class.__name__}",
                params=individual.params,
            )
            strategy = self.strategy_class(strategy_params)
            signals = strategy.generate_signals(price_data)

            # Simple signal → returns conversion
            # Signal: 1 = long, -1 = short, 0 = hold
            returns = price_data["close"].pct_change()

            # Map signals to positions
            positions = signals.map({1: 1, -1: -1, 0: 0}).fillna(0).shift(1).fillna(0)
            strategy_returns = (positions * returns).dropna()

            # Identify trades (position changes)
            trade_pnls = []
            position = 0
            entry_price = 0.0

            for i, (ts, row) in enumerate(price_data.iterrows()):
                current_pos = int(positions.get(ts, 0))
                price = row["close"]

                if current_pos != position:
                    if position != 0:
                        pnl = (price - entry_price) / entry_price * position
                        trade_pnls.append(pnl)
                    if current_pos != 0:
                        entry_price = price
                    position = current_pos

            # Compute fitness
            individual.fitness = self.fitness_fn(strategy_returns, trade_pnls)
            individual.sharpe = compute_sharpe(strategy_returns) if len(strategy_returns) > 20 else 0
            individual.profit_factor = compute_profit_factor(trade_pnls) if trade_pnls else 0
            equity = (1 + strategy_returns).cumprod()
            dd, _ = compute_max_drawdown(equity) if len(equity) > 0 else (0, 0)
            individual.max_drawdown = dd
            individual.total_trades = len(trade_pnls)

        except Exception as e:
            logger.warning("ga_evaluation_error", error=str(e), params=individual.params)
            individual.fitness = -999.0

        return individual

    def evolve(
        self,
        price_data: pd.DataFrame,
        n_generations: int | None = None,
        verbose: bool = True,
    ) -> EvolutionResult:
        """
        Run the genetic algorithm evolution.

        Args:
            price_data: OHLCV DataFrame for backtesting fitness.
            n_generations: Override config generations.
            verbose: Log progress each generation.

        Returns:
            EvolutionResult with best individual, history, and stats.
        """
        generations = n_generations or self.config.n_generations

        # Initialize population
        population = [
            self._random_individual(generation=0)
            for _ in range(self.config.population_size)
        ]

        # Evaluate initial population
        population = [
            self._evaluate_individual(ind, price_data)
            for ind in population
        ]

        best_fitness_history = []
        mean_fitness_history = []
        all_individuals = list(population)
        total_evals = len(population)
        global_best = max(population, key=lambda x: x.fitness)

        for gen in range(generations):
            # Elitism: keep top N%
            n_elite = max(1, int(self.config.population_size * self.config.elite_pct))
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            new_population = sorted_pop[:n_elite]

            # Generate offspring
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                if random.random() < self.config.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1 = Individual(params=parent1.params.copy())
                    child2 = Individual(params=parent2.params.copy())

                if random.random() < self.config.mutation_prob:
                    child1 = self._mutate(child1)
                if random.random() < self.config.mutation_prob:
                    child2 = self._mutate(child2)

                child1.generation = gen + 1
                child2.generation = gen + 1
                new_population.extend([child1, child2])

            # Trim to population size
            new_population = new_population[:self.config.population_size]

            # Evaluate new individuals
            for i, ind in enumerate(new_population):
                if ind.fitness == 0.0:  # Not yet evaluated
                    new_population[i] = self._evaluate_individual(ind, price_data)
                    total_evals += 1

            population = new_population
            all_individuals.extend(population)

            # Track stats
            gen_best = max(population, key=lambda x: x.fitness)
            gen_mean = np.mean([ind.fitness for ind in population])

            if gen_best.fitness > global_best.fitness:
                global_best = gen_best

            best_fitness_history.append(global_best.fitness)
            mean_fitness_history.append(float(gen_mean))

            if verbose and (gen % 10 == 0 or gen == generations - 1):
                logger.info(
                    "ga_generation",
                    gen=gen + 1,
                    best_fitness=f"{global_best.fitness:.4f}",
                    mean_fitness=f"{gen_mean:.4f}",
                    best_sharpe=f"{global_best.sharpe:.3f}",
                    best_trades=global_best.total_trades,
                )

        logger.info(
            "ga_evolution_complete",
            generations=generations,
            total_evaluations=total_evals,
            best_fitness=f"{global_best.fitness:.4f}",
            best_params=global_best.params,
        )

        return EvolutionResult(
            best_individual=global_best,
            best_fitness_history=best_fitness_history,
            mean_fitness_history=mean_fitness_history,
            all_individuals=all_individuals,
            generations=generations,
            total_evaluations=total_evals,
        )
