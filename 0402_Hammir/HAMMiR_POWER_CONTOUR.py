import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Constants
# -----------------------------
P_fus = 350.0
P_alpha = 0.2 * P_fus
P_n = 0.8 * P_fus

P_NBI = 20.0
eta_NBI = 0.60

# -----------------------------
# 2. Parameter ranges
# -----------------------------
eta_ele = np.linspace(0.45, 0.55, 101)
C_mult = np.linspace(1.05, 1.15, 101)

EE, CC = np.meshgrid(eta_ele, C_mult)

# -----------------------------
# 3. Net electric power (Eq. 3.14)
# -----------------------------
P_net = (P_alpha + P_NBI + CC * P_n) * EE - (P_NBI / eta_NBI)

# -----------------------------
# 4. Specific point of interest
# -----------------------------
eta_point = 0.50
C_point = 1.10

P_net_point = (P_alpha + P_NBI + C_point * P_n) * eta_point - (P_NBI / eta_NBI)

print("====================================")
print(f"At η_ele = {eta_point:.2f},  C_mult = {C_point:.2f}")
print(f"Net electric power P_net = {P_net_point:.3f} MW")
print("====================================")

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(7,5))
cont = plt.contourf(EE, CC, P_net, levels=30)

# ---- COLORBAR with fontsize ----
cbar = plt.colorbar(cont)
cbar.ax.tick_params(labelsize=14)  # tick fontsize
cbar.set_label("P_net,electric (MW)", fontsize=14)  # label fontsize; underscore OK

# ---- main plot ----
plt.scatter(eta_point, C_point, color='red', s=60)

plt.xlabel(r"$\eta_{\rm electric}$", fontsize=16)
plt.ylabel(r"$C_{\rm mult}$", fontsize=16)
plt.title("Net Electric Power Contour", fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()
