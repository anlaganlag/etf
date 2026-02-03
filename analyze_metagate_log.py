import re
import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "output_log_metagate.txt"
OUTPUT_CSV = "metagate_analysis.csv"

def parse_metagate_log():
    data = []
    
    # Regex to parse NOT the METAGATE_LOG line (it's csv) but to robustly handle it
    # Format: METAGATE_LOG,2022-01-01 10:00:00,0.05,0.05,SAFE,1.0
    
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found.")
        return

    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("METAGATE_LOG"):
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    date_str = parts[1]
                    br_raw = float(parts[2])
                    br_smooth = float(parts[3])
                    state = parts[4]
                    scaler = float(parts[5])
                    data.append({
                        'date': pd.to_datetime(date_str),
                        'br_raw': br_raw,
                        'br_smooth': br_smooth,
                        'state': state,
                        'risk_scaler': scaler
                    })

    if not data:
        print("No METAGATE_LOG entries found.")
        return

    df = pd.DataFrame(data).sort_values('date').set_index('date')
    df.to_csv(OUTPUT_CSV)
    print(f"Saved parsed data to {OUTPUT_CSV}")

    # Basic Analysis
    print("\n--- Meta-Gate Diagnostics ---")
    print(f"Total Trading Days: {len(df)}")
    
    state_counts = df['state'].value_counts()
    print("\nState Distribution:")
    print(state_counts)
    print((state_counts / len(df)).apply(lambda x: f"{x:.1%}"))
    
    # Correlation Check (Logic):
    # When BR > 20% (Caution Threshold), what happened?
    caution_days = df[df['br_smooth'] > 0.20]
    print(f"\nDays above 20% Broken Ratio: {len(caution_days)} ({len(caution_days)/len(df):.1%})")
    
    danger_days = df[df['br_smooth'] > 0.40]
    print(f"Days above 40% Broken Ratio: {len(danger_days)} ({len(danger_days)/len(df):.1%})")
    
    # Plotting (Optional if running locally, but useful to generate PNG)
    # plt.figure(figsize=(12, 6))
    # plt.plot(df.index, df['br_smooth'], label='Broken Ratio (Smooth)')
    # plt.axhline(0.20, color='orange', linestyle='--', label='Caution (20%)')
    # plt.axhline(0.40, color='red', linestyle='--', label='Danger (40%)')
    # plt.fill_between(df.index, 0, 1, where=(df['state']=='CAUTION'), color='orange', alpha=0.3)
    # plt.fill_between(df.index, 0, 1, where=(df['state']=='DANGER'), color='red', alpha=0.3)
    # plt.title('Meta-Gate Broken Ratio & State Machine')
    # plt.legend()
    # plt.savefig('metagate_plot.png')
    # print("Saved plot to metagate_plot.png")

if __name__ == "__main__":
    parse_metagate_log()
