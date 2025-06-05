# Minimum Weighted Vertex Cover (MWVC) for PLC Networks

This repository contains implementations and experiments for solving Minimum Weighted Vertex Cover (MWVC) problems, with a special focus on Power Line Communication (PLC) networks.

## Repository Structure

### Code/
Contains the main implementation of MWVC algorithms for PLC networks (trees):
- `perform_MWVC.py`: Implementation of MWVC and MVC algorithms specifically designed for PLC network structures.

### Experiment/
Contains various experimental implementations and tests for different graph structures and heuristics:

#### PLC Network Analysis
- `plc_network_generator.py`: Generator for PLC network structures
- `solve_mwvc_network.py`: Solver for MWVC on PLC networks
- `BA_att_calc.py`: Attenuation calculations for PLC networks

#### Algorithm Implementations
- `min_weighted_vertex_cover.py`: Base implementation of MWVC
- `approximate_mwvc.py`: Approximate solution for MWVC
- `advanced_approximate_mwvc.py`: Enhanced approximate solution
- `fast_approximate_mwvc.py`: Optimized approximate solution
- `greedy_exact.py`: Greedy algorithm implementation
- `pricing_greedy.py`: Pricing-based greedy approach
- `quantum.py`: Quantum computing approach for MWVC

#### Testing and Benchmarking
- `benchmark.py`: Benchmarking tools
- `graph_MWVC.py`: Testing on general graphs
- `comp_algo_rapport.py`: Algorithm comparison
- `test_rapport.py`: Test suite
- Various CSV files containing test results and network data

## Features

- Implementation of MWVC and MVC algorithms for PLC networks
- Multiple heuristic approaches for solving MWVC problems
- Comprehensive testing suite for different graph structures
- Benchmarking tools for algorithm comparison
- Support for both general graphs and specialized PLC network structures
- Quantum computing approach for MWVC problems

## Data Files

The repository includes several data files for testing and analysis:
- Network data in Excel format (`nodes.xlsx`, `edges.xlsx`)
- Attenuation data for PLC networks
- Benchmark results and algorithm comparison summaries
- Various CSV files containing test results and network parameters

## Usage

The code is organized to support both theoretical research and practical applications in PLC network optimization. The main implementations can be found in the `Code` directory, while the `Experiment` directory contains various experimental approaches and testing tools.

For specific usage instructions, please refer to the documentation within each file. 