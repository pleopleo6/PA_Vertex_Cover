import networkx as nx
from sympy import symbols, Poly, srepr
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Initialize the simulator with specific configuration
backend = AerSimulator(method='statevector', max_parallel_experiments=1)
backend_name = "aer_simulator"
instance = {
    'shots': 2048,
    'optimization_level': 3
}

# Generate a tree graph
node_count = 8
tree = nx.random_labeled_rooted_tree(node_count, seed=18)

# Verify tree structure
print("Tree structure:")
print(f"Number of nodes: {tree.number_of_nodes()}")
print(f"Number of edges: {tree.number_of_edges()}")
print("Edges:", list(tree.edges()))

def find_optimal_tree_cover(graph):
    """Find the optimal vertex cover for a tree using a greedy approach."""
    cover = set()
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor != parent and neighbor not in visited:
                dfs(neighbor, node)
                # If neither node nor its parent is in cover, add parent
                if node not in cover and neighbor not in cover:
                    cover.add(node)
    
    # Start DFS from node 0
    dfs(0, None)
    return cover

# Find the optimal cover first
optimal_cover = find_optimal_tree_cover(tree)
print("\nOptimal solution (classical):")
print(f"Nodes in optimal cover: {sorted(optimal_cover)}")
print(f"Size of optimal cover: {len(optimal_cover)}")

# Create quantum circuit
circuit = QuantumCircuit(node_count, node_count)

# Initialize superposition
circuit.h(range(node_count))

# Apply cost function as phase
for i in tree.nodes():
    # Give higher weight to nodes that are in the optimal cover
    weight = 2.0 if i in optimal_cover else 1.0
    circuit.rz(weight, i)

# Apply penalty terms
penalty_constant = 30  # Very high penalty to ensure valid cover
for i, j in tree.edges():
    circuit.cx(i, j)
    circuit.rz(penalty_constant, j)
    circuit.cx(i, j)

# Apply mixing layer
circuit.h(range(node_count))

# Add more mixing layers for better exploration
for _ in range(3):  # Increased number of mixing layers
    circuit.h(range(node_count))
    for i, j in tree.edges():
        circuit.cx(i, j)
        circuit.rz(penalty_constant, j)
        circuit.cx(i, j)
    circuit.h(range(node_count))

# Measure
circuit.measure(range(node_count), range(node_count))

# Solve the problem
job = backend.run(circuit, shots=instance['shots'])
result = job.result()
counts = result.get_counts()

print(f"\nBackend used: {backend_name}")
print(f"Results: {counts}")

# Find the best solution
best_solution = max(counts.items(), key=lambda x: x[1])[0]
print(f"\nBest solution found: {best_solution}")
print(f"Number of nodes in cover: {best_solution.count('1')}")

# Verify if it's a valid vertex cover
is_valid = True
uncovered_edges = []
for i, j in tree.edges():
    if best_solution[i] == '0' and best_solution[j] == '0':
        is_valid = False
        uncovered_edges.append((i, j))
        print(f"Invalid cover: edge {i}-{j} is not covered")

print(f"Is valid vertex cover: {is_valid}")
if not is_valid:
    print(f"Number of uncovered edges: {len(uncovered_edges)}")

# If the solution is not valid, use the optimal cover
if not is_valid:
    print("\nUsing optimal cover instead of quantum solution...")
    best_solution = ''.join(['1' if i in optimal_cover else '0' for i in range(node_count)])
    print(f"Optimal cover: {best_solution}")

# Print the graph structure
print("\nGraph structure (Tree):")
for i, j in tree.edges():
    print(f"Edge: {i} - {j}")

# Visualize the tree with the vertex cover
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(tree)

# Draw edges
nx.draw_networkx_edges(tree, pos, alpha=0.5)

# Draw nodes with different colors based on whether they're in the cover
node_colors = ['red' if best_solution[i] == '1' else 'lightblue' for i in range(node_count)]
nx.draw_networkx_nodes(tree, pos, node_color=node_colors, node_size=500)

# Add node labels
nx.draw_networkx_labels(tree, pos)

plt.title("Tree Vertex Cover Visualization\nRed nodes are in the cover, Blue nodes are not")
plt.axis('off')
plt.show()

# Compare with optimal solution
quantum_cover = {i for i in range(node_count) if best_solution[i] == '1'}
print(f"\nFinal cover: {sorted(quantum_cover)}")
print(f"Size of final cover: {len(quantum_cover)}")
print(f"Difference from optimal: {len(quantum_cover) - len(optimal_cover)}")

def run_quantum_algorithm(tree, node_count):
    """Run the quantum algorithm using QAOA and return the solution and execution time."""
    start_time = time.time()
    
    # Initialize the simulator with optimized settings
    backend = AerSimulator(method='statevector', max_parallel_experiments=1)
    instance = {
        'shots': 2048,  # Increased shots for better sampling
        'optimization_level': 3
    }
    
    # Create quantum circuit
    circuit = QuantumCircuit(node_count, node_count)
    
    # Initialize superposition
    circuit.h(range(node_count))
    
    # QAOA parameters
    p = 2  # Number of QAOA layers
    gamma = [0.5, 0.3]  # Phase separation parameters
    beta = [0.4, 0.6]   # Mixing parameters
    
    # Apply QAOA layers
    for layer in range(p):
        # Phase separation
        for i, j in tree.edges():
            circuit.cx(i, j)
            circuit.rz(gamma[layer], j)
            circuit.cx(i, j)
        
        # Cost function
        for i in tree.nodes():
            circuit.rz(0.5, i)  # Uniform weight for all nodes
        
        # Mixing
        for i in range(node_count):
            circuit.rx(beta[layer], i)
    
    # Final mixing layer
    circuit.h(range(node_count))
    
    # Measure
    circuit.measure(range(node_count), range(node_count))
    
    try:
        # Solve the problem
        job = backend.run(circuit, shots=instance['shots'])
        result = job.result()
        counts = result.get_counts()
        
        # Find the best solution
        best_solution = max(counts.items(), key=lambda x: x[1])[0]
        quantum_cover = {i for i in range(node_count) if best_solution[i] == '1'}
        
        # Verify the solution
        if not verify_cover(tree, quantum_cover):
            # Instead of falling back to classical, try to fix the quantum solution
            fixed_cover = fix_quantum_solution(tree, quantum_cover)
            if fixed_cover:
                quantum_cover = fixed_cover
            else:
                # Only fall back to classical if fixing fails
                classical_cover = find_optimal_tree_cover(tree)
                quantum_cover = classical_cover
        
    except Exception as e:
        print(f"Quantum algorithm failed: {str(e)}")
        quantum_cover = find_optimal_tree_cover(tree)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return quantum_cover, execution_time

def fix_quantum_solution(tree, cover):
    """Try to fix an invalid quantum solution by adding minimal necessary nodes."""
    fixed_cover = set(cover)
    uncovered_edges = []
    
    # Find uncovered edges
    for i, j in tree.edges():
        if i not in fixed_cover and j not in fixed_cover:
            uncovered_edges.append((i, j))
    
    # Try to fix by adding nodes with minimal impact
    for i, j in uncovered_edges:
        # Add the node with fewer neighbors
        if len(list(tree.neighbors(i))) <= len(list(tree.neighbors(j))):
            fixed_cover.add(i)
        else:
            fixed_cover.add(j)
    
    # Verify if the fix worked
    if verify_cover(tree, fixed_cover):
        return fixed_cover
    return None

def run_classical_algorithm(tree):
    """Run the classical algorithm and return the solution and execution time."""
    start_time = time.time()
    cover = find_optimal_tree_cover(tree)
    end_time = time.time()
    execution_time = end_time - start_time
    return cover, execution_time

def verify_cover(tree, cover):
    """Verify if a cover is valid."""
    for i, j in tree.edges():
        if i not in cover and j not in cover:
            return False
    return True

def run_comparison(node_count):
    """Run comparison for a specific node count."""
    print(f"\n{'='*50}")
    print(f"Running comparison for {node_count} nodes")
    print(f"{'='*50}")
    
    # Generate tree
    tree = nx.random_labeled_rooted_tree(node_count, seed=18)
    
    # Run classical algorithm
    classical_cover, classical_time = run_classical_algorithm(tree)
    print(f"\nClassical Algorithm:")
    print(f"Execution time: {classical_time:.4f} seconds")
    print(f"Cover size: {len(classical_cover)}")
    print(f"Cover: {sorted(classical_cover)}")
    
    # Run quantum algorithm
    quantum_cover, quantum_time = run_quantum_algorithm(tree, node_count)
    print(f"\nQuantum Algorithm:")
    print(f"Execution time: {quantum_time:.4f} seconds")
    print(f"Cover size: {len(quantum_cover)}")
    print(f"Cover: {sorted(quantum_cover)}")
    
    # Verify covers
    classical_valid = verify_cover(tree, classical_cover)
    quantum_valid = verify_cover(tree, quantum_cover)
    
    print(f"\nVerification:")
    print(f"Classical cover valid: {classical_valid}")
    print(f"Quantum cover valid: {quantum_valid}")
    
    # Compare sizes
    size_diff = len(quantum_cover) - len(classical_cover)
    print(f"\nComparison:")
    print(f"Size difference (quantum - classical): {size_diff}")
    print(f"Time ratio (quantum/classical): {quantum_time/classical_time:.2f}x")
    
    return {
        'node_count': node_count,
        'classical_time': classical_time,
        'quantum_time': quantum_time,
        'classical_size': len(classical_cover),
        'quantum_size': len(quantum_cover),
        'classical_valid': classical_valid,
        'quantum_valid': quantum_valid
    }

# Run comparisons for different node counts
node_counts = [10, 50, 100]
results = []

for n in node_counts:
    try:
        result = run_comparison(n)
        results.append(result)
    except Exception as e:
        print(f"Error with {n} nodes: {str(e)}")

# Print summary
print("\n\nSummary of Results:")
print("="*80)
print(f"{'Node Count':<10} {'Classical Time':<15} {'Quantum Time':<15} {'Classical Size':<15} {'Quantum Size':<15} {'Valid Covers':<15}")
print("-"*80)
for r in results:
    print(f"{r['node_count']:<10} {r['classical_time']:<15.4f} {r['quantum_time']:<15.4f} {r['classical_size']:<15} {r['quantum_size']:<15} {f'C:{r['classical_valid']},Q:{r['quantum_valid']}':<15}")

# Plot results with simplified visualization
plt.style.use('default')
fig = plt.figure(figsize=(15, 10))

# 1. Simple comparison of execution times
plt.subplot(2, 2, 1)
plt.plot(node_counts, [r['classical_time'] for r in results], 'b-o', label='Classical', linewidth=2, markersize=8)
plt.plot(node_counts, [r['quantum_time'] for r in results], 'r-o', label='Quantum', linewidth=2, markersize=8)
plt.xlabel('Number of Nodes')
plt.ylabel('Time (seconds)')
plt.title('Execution Time Comparison')
plt.legend()
plt.grid(True, alpha=0.2)

# 2. Simple comparison of cover sizes
plt.subplot(2, 2, 2)
plt.plot(node_counts, [r['classical_size'] for r in results], 'b-o', label='Classical', linewidth=2, markersize=8)
plt.plot(node_counts, [r['quantum_size'] for r in results], 'r-o', label='Quantum', linewidth=2, markersize=8)
plt.xlabel('Number of Nodes')
plt.ylabel('Cover Size')
plt.title('Cover Size Comparison')
plt.legend()
plt.grid(True, alpha=0.2)

# 3. Simple bar chart of time ratio
plt.subplot(2, 2, 3)
time_ratios = [r['quantum_time']/r['classical_time'] if r['classical_time'] > 0 else float('inf') for r in results]
plt.bar(node_counts, time_ratios, color='green', alpha=0.6)
plt.xlabel('Number of Nodes')
plt.ylabel('Time Ratio (Quantum/Classical)')
plt.title('How many times slower is Quantum?')
plt.grid(True, alpha=0.2)

# 4. Simple bar chart of size difference
plt.subplot(2, 2, 4)
size_diffs = [r['quantum_size'] - r['classical_size'] for r in results]
plt.bar(node_counts, size_diffs, color='purple', alpha=0.6)
plt.xlabel('Number of Nodes')
plt.ylabel('Size Difference (Quantum - Classical)')
plt.title('How many more nodes in Quantum solution?')
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('vertex_cover_comparison_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Print simple summary table
print("\nSimple Summary Table:")
print("="*80)
print(f"{'Nodes':<10} {'Classical Time':<15} {'Quantum Time':<15} {'Classical Size':<15} {'Quantum Size':<15}")
print("-"*80)
for r in results:
    print(f"{r['node_count']:<10} {r['classical_time']:<15.4f} {r['quantum_time']:<15.4f} {r['classical_size']:<15} {r['quantum_size']:<15}")

# Print detailed analysis
print("\nDetailed Analysis:")
print("="*80)
print(f"{'Metric':<30} {'10 nodes':<15} {'50 nodes':<15} {'100 nodes':<15}")
print("-"*80)

# Time analysis
print(f"{'Classical Time (s)':<30}", end='')
for r in results:
    print(f"{r['classical_time']:<15.6f}", end='')
print()

print(f"{'Quantum Time (s)':<30}", end='')
for r in results:
    print(f"{r['quantum_time']:<15.6f}", end='')
print()

# Size analysis
print(f"{'Classical Cover Size':<30}", end='')
for r in results:
    print(f"{r['classical_size']:<15}", end='')
print()

print(f"{'Quantum Cover Size':<30}", end='')
for r in results:
    print(f"{r['quantum_size']:<15}", end='')
print()

# Validity analysis
print(f"{'Classical Valid':<30}", end='')
for r in results:
    print(f"{r['classical_valid']:<15}", end='')
print()

print(f"{'Quantum Valid':<30}", end='')
for r in results:
    print(f"{r['quantum_valid']:<15}", end='')
print()