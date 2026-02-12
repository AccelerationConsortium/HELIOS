"""
Enhanced CLI for running the Experiment Agent with real or simulated devices.
"""

import argparse
import sys
import random
from pathlib import Path

from ..orchestrator.supervisor import Supervisor
from ..devices.factory import DeviceFactory
from ..devices.real.config import ConfigManager, LabConfig


def main():
    parser = argparse.ArgumentParser(description="Run the Experiment Agent")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to lab configuration file"
    )
    parser.add_argument(
        "--simulation", action="store_true", help="Run in simulation mode (default)"
    )
    parser.add_argument(
        "--real-hardware", action="store_true", help="Run with real hardware devices"
    )
    parser.add_argument(
        "--device-name", type=str, default="heater_1", help="Name of the device to use"
    )
    parser.add_argument(
        "--target-temp", type=float, default=120.0, help="Target temperature to reach"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for simulation"
    )

    # Simulation-specific options
    sim_group = parser.add_argument_group("Simulation options")
    sim_group.add_argument(
        "--fault-mode",
        type=str,
        default="none",
        choices=["none", "random", "timeout", "overshoot", "sensor_fail"],
        help="Type of fault to simulate",
    )

    # Real hardware options
    hw_group = parser.add_argument_group("Real hardware options")
    hw_group.add_argument("--port", type=str, help="Serial port for device connection")
    hw_group.add_argument("--host", type=str, help="Network host for device connection")
    hw_group.add_argument(
        "--connection-type",
        type=str,
        default="serial",
        choices=["serial", "network"],
        help="Connection type for real devices",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random Seed: {args.seed}")

    # Determine mode
    simulation_mode = not args.real_hardware
    if args.real_hardware and args.simulation:
        print("Error: Cannot specify both --simulation and --real-hardware")
        sys.exit(1)

    print(f"Mode: {'Simulation' if simulation_mode else 'Real Hardware'}")
    print(f"Target Temperature: {args.target_temp}°C")

    try:
        if simulation_mode:
            # Create simulated device
            device = DeviceFactory.create_simulated_device(
                device_type="heater", name=args.device_name, fault_mode=args.fault_mode
            )
            print(
                f"Created simulated device: {args.device_name} (fault mode: {args.fault_mode})"
            )

        else:
            # Load configuration for real hardware
            config_manager = ConfigManager()

            if args.config:
                config_path = Path(args.config)
                lab_config = config_manager.load_config(config_path.name)
                config_manager.config_dir = config_path.parent
            else:
                lab_config = config_manager.load_config()

            # Find or create device configuration
            device_config = None
            for dev_config in lab_config.devices:
                if dev_config.name == args.device_name:
                    device_config = dev_config
                    break

            if not device_config:
                # Create device config from command line args
                if args.connection_type == "serial":
                    if not args.port:
                        print("Error: --port required for serial connection")
                        sys.exit(1)
                    from ..devices.real.config import create_serial_heater_config

                    device_config = create_serial_heater_config(
                        args.device_name, args.port
                    )
                elif args.connection_type == "network":
                    if not args.host:
                        print("Error: --host required for network connection")
                        sys.exit(1)
                    from ..devices.real.config import create_network_heater_config

                    device_config = create_network_heater_config(
                        args.device_name, args.host
                    )
                else:
                    print(f"Error: Unsupported connection type: {args.connection_type}")
                    sys.exit(1)

                # Add to config and save
                lab_config.devices.append(device_config)
                config_manager.save_config(lab_config)

            # Create real device
            device = DeviceFactory.create_device(device_config, simulation_mode=False)
            print(
                f"Created real device: {args.device_name} ({device_config.connection_type})"
            )

        # Create and run supervisor
        supervisor = Supervisor(target_temp=args.target_temp, device=device)
        try:
            supervisor.run()
        finally:
            supervisor.shutdown()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
