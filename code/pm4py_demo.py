import pm4py

# 1. Load the XES file
print("Loading XES file...")
log = pm4py.read_xes('data/cda_bus_routes.xes')

# 2. Print Trace Statistics to the terminal
print("\n--- Trace Statistics ---")
print(f"Total number of events: {len(log)}")
print("------------------------\n")

# 3. Discover the Process Model using Heuristic Miner
print("Discovering Process Model (this might take a few seconds)...")
heu_net = pm4py.discover_heuristics_net(log)

# 4. View the Process Map (pops up a window!)
pm4py.view_heuristics_net(heu_net)
