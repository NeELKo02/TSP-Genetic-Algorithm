# Traveling Salesman Problem - Genetic Algorithm Solution

A Python implementation of a Genetic Algorithm to solve the Traveling Salesman Problem (TSP) using 3D coordinates.

## Problem Description
The TSP is an optimization problem where we need to find the shortest possible route that visits each city exactly once and returns to the starting point.

## Algorithm Features
- **Genetic Algorithm** with hybrid initialization
- **K-Nearest Neighbor (KNN)** + Random initialization
- **Two-point crossover** for reproduction
- **Swap and inversion mutations**
- **Elitism** to preserve best solutions
- **Early stopping** and time limits
- **3D Euclidean distance** calculations

## Files
- `AI-HW1.py` - Main genetic algorithm implementation
- `input.txt` - 500 cities with 3D coordinates
- `output.txt` - Optimal tour solution
- `Assignment Description.pdf` - Problem requirements

## Usage
```bash
python AI-HW1.py
```

## Results
- **Problem Size**: 500 cities
- **Total Distance**: 51,916.101 units
- **Performance**: 6.32x better than random tour
- **Quality**: Better than nearest neighbor approach

## Algorithm Parameters
- Population size: 1000 (adaptive)
- Generations: 500 (max)
- Mutation rate: 20%
- Elitism: 10%
- Early stopping: 50 generations without improvement
- Time limit: 290 seconds

## Performance Analysis
- Average edge length: 103.8 units
- Quality ratio: 16.1% of theoretical minimum
- Significant improvement over random and greedy approaches
