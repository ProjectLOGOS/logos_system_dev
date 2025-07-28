# initialize_system.py
"""
Full system initialization script for the Logos‑AGI:
1. Seed raw data nodes into the Divine Plane.
2. Run one Divine Plane cycle (with introspection, teleology hooks).
3. Execute the branching smoke test via FullBranchExecutor.
"""
def main():
    # 1. Seed raw data nodes
    print("\n=== [1] Seeding Raw Data Nodes ===")
    import seed_data_nodes  # populates and prints DataSeedNode summary

    # 2. Run Divine Plane cycle
    print("\n=== [2] Running Divine Plane Cycle ===")
    from divine_plane_cycle import run_divine_cycle
    summary = run_divine_cycle()
    print("Divine Plane Cycle Summary:\n", summary)

    # 3. Execute Branch Executor smoke test
    print("\n=== [3] Executing Branch Executor Smoke Test ===")
    from logos_full_integration import smoke_test
    smoke_test()

    print("\n=== System Initialization Complete ===")

if __name__ == '__main__':
    main()
