#!/usr/bin/env python3
"""Test script to verify the flow logic"""

# Simulate the scenario
def test_flow(epochs_confirmed, modify_base, enable_benchmark_rl):
    """Test the flow logic"""
    print(f"\nTest: epochs_confirmed={epochs_confirmed}, modify_base={modify_base}, enable_benchmark_rl={enable_benchmark_rl}")
    print("-" * 60)

    manual_reinforced_epochs = None

    if not epochs_confirmed:
        print("  User said 'n' to 'Continue with these epoch settings?'")
        print("  Asking what to configure...")

        # Ask if user wants to modify base epochs
        print(f"  modify_base = {modify_base}")

        if modify_base:
            print("    → User modifying base epochs...")
            print("    → Base epochs modified!")
        else:
            print("    → User skipped base epochs modification")

        # This should ALWAYS execute if enable_benchmark_rl is True
        if enable_benchmark_rl:
            print("  Checking enable_benchmark_rl...")
            configure_rl_epochs = True  # Simulate user saying yes

            if configure_rl_epochs:
                print("    → User configuring RL epochs manually...")
                manual_reinforced_epochs = 10
                print(f"    → Manual RL epochs set to: {manual_reinforced_epochs}")
            else:
                print("    → RL epochs will be auto-calculated")
        else:
            print("  RL is disabled, skipping RL epoch configuration")
    else:
        print("  User confirmed epochs, no configuration needed")

    print(f"\nResult: manual_reinforced_epochs = {manual_reinforced_epochs}")
    print("=" * 60)


# Test all scenarios
print("\n" + "=" * 60)
print("SCENARIO 1: User says 'n' to epochs, 'y' to modify_base")
print("=" * 60)
test_flow(epochs_confirmed=False, modify_base=True, enable_benchmark_rl=True)

print("\n" + "=" * 60)
print("SCENARIO 2: User says 'n' to epochs, 'n' to modify_base")
print("=" * 60)
test_flow(epochs_confirmed=False, modify_base=False, enable_benchmark_rl=True)

print("\n" + "=" * 60)
print("SCENARIO 3: User says 'y' to epochs (should skip all)")
print("=" * 60)
test_flow(epochs_confirmed=True, modify_base=False, enable_benchmark_rl=True)
