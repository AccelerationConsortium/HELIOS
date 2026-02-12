import argparse
import sys
import random
from ..orchestrator.supervisor import Supervisor

def main():
    parser = argparse.ArgumentParser(description="Run the Experiment Agent Simulation")
    parser.add_argument("--fault-mode", type=str, default="none", 
                        choices=["none", "random", "timeout", "overshoot", "sensor_fail"],
                        help="Type of fault to simulate")
    parser.add_argument("--target-temp", type=float, default=120.0,
                        help="Target temperature to reach")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random Seed: {args.seed}")
    
    # 2. Log Config
    print(f"Configuration: Fault Mode={args.fault_mode}, Target Temp={args.target_temp}")
    
    supervisor = Supervisor(target_temp=args.target_temp, fault_mode=args.fault_mode)
    try:
        supervisor.run()
    finally:
        supervisor.shutdown()

if __name__ == "__main__":
    main()
