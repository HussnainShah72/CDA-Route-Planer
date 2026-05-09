import pm4py
import os
from pathlib import Path

def verify_xes():
    # Define paths
    data_dir = Path(__file__).parent
    xes_path = data_dir / "cda_bus_routes.xes"
    output_inductive = data_dir / "task2_inductive_miner.png"
    output_heuristic = data_dir / "task2_heuristic_miner.png"
    
    print(f"--- XES Validation Tool ---")
    
    if not xes_path.exists():
        print(f"Error: {xes_path} not found!")
        return

    print(f"1. Importing XES log: {xes_path.name}...")
    log = pm4py.read_xes(str(xes_path))
    
    # Basic statistics
    num_traces = len(log)
    num_events = sum(len(trace) for trace in log)
    print(f"   Success: Imported {num_traces} traces and {num_events} events.")

    # Inductive Miner
    print(f"2. Discovering Process Tree (Inductive Miner)...")
    tree = pm4py.discover_process_tree_inductive(log)
    print(f"   Exporting to: {output_inductive.name}...")
    pm4py.save_vis_process_tree(tree, str(output_inductive))
    
    # Heuristic Miner
    print(f"3. Discovering Heuristics Net (Heuristic Miner)...")
    heu_net = pm4py.discover_heuristics_net(log)
    print(f"   Exporting to: {output_heuristic.name}...")
    pm4py.save_vis_heuristics_net(heu_net, str(output_heuristic))
    
    print(f"\n✅ VALIDATION COMPLETE.")
    print(f"Inductive Miner screenshot: {output_inductive.absolute()}")
    print(f"Heuristic Miner screenshot: {output_heuristic.absolute()}")

if __name__ == "__main__":
    try:
        verify_xes()
    except Exception as e:
        print(f"\n❌ Error during validation: {str(e)}")
        print("Note: Ensure 'graphviz' is installed on your system (sudo apt install graphviz).")
