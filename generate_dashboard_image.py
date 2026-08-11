import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("images", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor="#0F172A")
fig.suptitle("🚦 Traffic Volume Prediction System - Dashboard Preview\nSwapna V | ML Engineer | IPEC Solutions", color="white", fontsize=14, fontweight="bold")

# Plot 1: Hourly curve
hours = np.arange(24)
vol = 1000 + 3500 * np.exp(-((hours - 8)/2)**2) + 4000 * np.exp(-((hours - 17)/2.5)**2) + np.random.normal(0, 100, 24)
axes[0, 0].set_facecolor("#1E293B")
axes[0, 0].plot(hours, vol, color="#3B82F6", linewidth=2.5, marker="o")
axes[0, 0].set_title("Hourly Traffic Volume", color="white", fontsize=10)
axes[0, 0].tick_params(colors="white")
axes[0, 0].grid(True, color="#334155")

# Plot 2: Weather Bar Chart
weathers = ["Clear", "Clouds", "Rain", "Mist", "Snow"]
w_vol = [3800, 3500, 2900, 3100, 2200]
axes[0, 1].set_facecolor("#1E293B")
axes[0, 1].bar(weathers, w_vol, color="#10B981")
axes[0, 1].set_title("Traffic by Weather Condition", color="white", fontsize=10)
axes[0, 1].tick_params(colors="white")
axes[0, 1].grid(True, color="#334155")

# Plot 3: Model Accuracy Bar Chart
models = ["Linear Reg", "Decision Tree", "Random Forest", "Grad Boost"]
r2_scores = [0.45, 0.969, 0.979, 0.982]
axes[1, 0].set_facecolor("#1E293B")
axes[1, 0].barh(models, r2_scores, color="#8B5CF6")
axes[1, 0].set_title("Model R² Score Comparison", color="white", fontsize=10)
axes[1, 0].tick_params(colors="white")
axes[1, 0].grid(True, color="#334155")

# Plot 4: Prediction Gauge metric card
axes[1, 1].set_facecolor("#1E293B")
axes[1, 1].text(0.5, 0.6, "PREDICTED TRAFFIC", color="#94A3B8", fontsize=11, ha="center", weight="bold")
axes[1, 1].text(0.5, 0.35, "4,820 veh/h", color="#F59E0B", fontsize=18, ha="center", weight="bold")
axes[1, 1].text(0.5, 0.15, "🟠 HIGH TRAFFIC", color="#F97316", fontsize=12, ha="center", weight="bold")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig("images/dashboard.png", dpi=150, bbox_inches="tight")
print("Dashboard preview image generated at images/dashboard.png")
