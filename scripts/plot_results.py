from pathlib import Path
import csv
import matplotlib.pyplot as plt

Path("figures").mkdir(exist_ok=True)

settings = []
wiki = []
c4 = []

with open("results/ppl_results.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] != "success":
            continue
        settings.append(row["setting"])
        wiki.append(float(row["wikitext2_ppl"]))
        c4.append(float(row["c4_ppl"]))

plt.figure()
plt.plot(settings, wiki, marker="o", label="WikiText2")
plt.plot(settings, c4, marker="o", label="C4")
plt.xlabel("Quantization Setting")
plt.ylabel("Perplexity")
plt.title("PPL under Different Weight Bit-widths")
plt.legend()
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/ppl_vs_bitwidth.png", dpi=300)

loss_settings = ["W4-baseline", "W3-baseline", "W2-safe"]
loss_values = [0.1408, 0.5435, 3.4950]

plt.figure()
plt.bar(loss_settings, loss_values)
plt.xlabel("Run")
plt.ylabel("Final Calibration Loss")
plt.title("Calibration Loss Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/calibration_loss_comparison.png", dpi=300)

time_settings = ["W4-baseline", "W3-baseline", "W2-safe"]
time_values = [1.69, 3.30, 8.76]

plt.figure()
plt.bar(time_settings, time_values)
plt.xlabel("Run")
plt.ylabel("Time Cost (hours)")
plt.title("Runtime Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/runtime_comparison.png", dpi=300)

print("Saved figures to figures/")
