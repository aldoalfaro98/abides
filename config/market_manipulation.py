# FINTECH D1: Market Manipulation Experiment Configuration
# - 1 Exchange Agent
# - 2 Adaptive POV Market Maker Agents
# - 50 Value Agents
# - 25 Momentum Agents
# - 5000 Noise Agents
# - 100 HBL Agents
# - 10 Simple Spoofing Agent

import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from model.LatencyModel import LatencyModel

# AGENTS
from agent.ExchangeAgent import ExchangeAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.ValueAgent import ValueAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.NoiseAgent import NoiseAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent

# MARKET MANIPULATOR AGENT
from agent.market_manipulators.SimpleSpoofingAgent import SimpleSpoofingAgent

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for market manipulation experiment config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('--start-time',
                    default='09:30:00',
                    type=parse,
                    help='Starting time of simulation.'
                    )
parser.add_argument('--end-time',
                    default='11:30:00',
                    type=parse,
                    help='Ending time of simulation.'
                    )
parser.add_argument('-l',
                    '--log_dir',
                    default='market_manipulation',
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-8,
                    help='Volatility of fundamental time series.'
                    )

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed        # Random seed specification on the command line.

if not seed:
    seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

exchange_log_orders = True
log_orders = None
book_freq = 0

print("FINTECH D1: MARKET MANIPULATION EXPERIMENT\n")

simulation_start_time = dt.datetime(2020, 1, 1, 9, 30, 0)
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta(args.start_time.strftime('%H:%M:%S'))
mkt_close = historical_date + pd.to_timedelta(args.end_time.strftime('%H:%M:%S'))
agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

r_bar = 1e5
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11

symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'sigma_s': 0,
                    'fund_vol': args.fund_vol,
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# EXCHANGE AGENT ---------------------------------------------------------------

stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=exchange_log_orders,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# VALUE AGENTS -----------------------------------------------------------------

num_value = 50
agents.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=sigma_n,
                          r_bar=r_bar,
                          kappa=kappa,
                          lambda_a=lambda_a,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])

# MOMENTUM AGENTS --------------------------------------------------------------

num_momentum_agents = 25
agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=log_orders,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")

# NOISE AGENTS -----------------------------------------------------------------

num_noise = 5000
noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")
noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])


# ADAPTIVE MARKET MAKER AGENT --------------------------------------------------

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

# each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
mm_params = [
    ('adaptive', 0.025, 10, '10S', 1),
    ('adaptive', 0.025, 10, '10S', 1)
]

num_mm_agents = len(mm_params)
mm_cancel_limit_delay = 50  # 50 nanoseconds

agents.extend([AdaptiveMarketMakerAgent(id=j,
                                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                type='AdaptivePOVMarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                pov=mm_params[idx][1],
                                min_order_size=mm_params[idx][4],
                                window_size=mm_params[idx][0],
                                num_ticks=mm_params[idx][2],
                                wake_up_freq=mm_params[idx][3],
                                cancel_limit_delay=mm_cancel_limit_delay,
                                skew_beta=0,
                                level_spacing=5,
                                spread_alpha=0.75,
                                backstop_quantity=50000,
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend('AdaptivePOVMarketMakerAgent')

# HBL Agents -------------------------------------------------------------------

num_hbl_agents = 100
agents.extend([HeuristicBeliefLearningAgent(id=j,
                                            name="HBL_AGENT_{}".format(j),
                                            type="HeuristicBeliefLearningAgent",
                                            symbol=symbol,
                                            starting_cash=starting_cash,
                                            sigma_n=10000,
                                            sigma_s=symbols[symbol]['fund_vol'],
                                            kappa=1.67e-15,
                                            r_bar=symbols[symbol]['r_bar'],
                                            q_max=10,
                                            sigma_pv=5e4,
                                            R_min=0,
                                            R_max=100,
                                            eta=1,
                                            lambda_a=1e-12,
                                            L=2,
                                            log_orders=log_orders,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                      dtype='uint64')))
               for j in range(agent_count, agent_count + num_hbl_agents)])
agent_count += num_hbl_agents
agent_types.extend("HeuristicBeliefLearningAgent")

# MARKET MANIPULATOR AGENT -----------------------------------------------------

num_market_manipulators = 1
agents.extend([
    SimpleSpoofingAgent(
        id=j,
        name="Spoofing Agent {}".format(j),
        type="SpoofingAgent",
        symbol=symbol,
        starting_cash=starting_cash,
        random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))
    )
        for j in range(agent_count, agent_count + num_market_manipulators)
])
agent_count += num_market_manipulators
agent_types.extend("SimpleSpoofingAgent")

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel(
    "Market Manipulation Experiment Kernel",
    random_state = np.random.RandomState(
        seed = np.random.randint(low=0, high=2 ** 32, dtype='uint64')
    )
)

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY

latency_rstate = np.random.RandomState(
    seed = np.random.randint(low=0, high=2**32)
)
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(
    0.0, nyc_to_seattle_meters, agent_count, random_state = latency_rstate
)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(
    latency_model = 'deterministic',
    random_state = latency_rstate,
    kwargs = model_args
)

# KERNEL

kernel.runner(
    agents=agents,
    startTime=kernelStartTime,
    stopTime=kernelStopTime,
    agentLatencyModel=latency_model,
    defaultComputationDelay=defaultComputationDelay,
    oracle=oracle,
    log_dir=args.log_dir
)


simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
