import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_methodology_plots():
    csv_path = 'data/robustness_evaluation_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Run python3 scripts/run_domain_randomization.py first.")
        return

    df = pd.read_csv(csv_path)
    summary = df.groupby('noise_level').mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Methodology Telemetry Profile: Sim-to-Real Robustness Mapping', fontsize=14, fontweight='bold')

    # Plot 1: Throughput Metrics
    axes[0].plot(summary['noise_level'], summary['throughput_rate'], marker='o', color='#2C3E50', linewidth=2)
    axes[0].set_title('Operational Throughput vs. Sensor Noise')
    axes[0].set_xlabel('Sensor Noise Variance ($\sigma$)')
    axes[0].set_ylabel('Mean Task Completion Rate')
    axes[0].grid(True, linestyle='--')

    # Plot 2: Efficiency Index
    axes[1].bar(summary['noise_level'].astype(str), summary['efficiency_score'], color='#34495E', width=0.4)
    axes[1].set_title('Policy Efficiency vs. Sensor Noise')
    axes[1].set_xlabel('Sensor Noise Variance ($\sigma$)')
    axes[1].set_ylabel('Reward/Step Efficiency Index')
    axes[1].grid(True, linestyle='--')

    # Plot 3: Collision Mitigation Metrics
    axes[2].plot(summary['noise_level'], summary['collision_rate'], marker='s', color='#E74C3C', linewidth=2, linestyle=':')
    axes[2].set_title('Safety Violation Metrics')
    axes[2].set_xlabel('Sensor Noise Variance ($\sigma$)')
    axes[2].set_ylabel('Collision Frequency Per Episode')
    axes[2].grid(True, linestyle='--')

    plt.tight_layout()
    plot_output_path = 'data/methodology_comparative_telemetry.png'
    plt.savefig(plot_output_path, dpi=300)
    print(f"[SUCCESS] High-resolution figure saved to: {plot_output_path}")

if __name__ == "__main__":
    generate_methodology_plots()
