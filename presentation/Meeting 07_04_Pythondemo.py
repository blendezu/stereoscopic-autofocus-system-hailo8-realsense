import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.signal
from scipy.interpolate import interp1d

from scipy.interpolate import CubicSpline


dataStereo = {
    "Rail_distance": [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, 3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4, 4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5, 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75, 5.8, 5.85, 5.9, 5.95, 6, 6.05, 6.1, 6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55, 6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7, 7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35, 7.4, 7.45, 7.5, 7.55, 7.6, 7.65, 7.7, 7.75, 7.8, 7.85, 7.9, 7.95, 8, 8.05, 8.1, 8.15, 8.2, 8.25, 8.3, 8.35, 8.4, 8.45, 8.5, 8.55, 8.6, 8.65, 8.7, 8.75, 8.8, 8.85, 8.9, 8.95, 9, 9.05, 9.1, 9.15, 9.2, 9.25, 9.3, 9.35, 9.4, 9.45, 9.5, 9.55, 9.6, 9.65, 9.7, 9.75, 9.8, 9.85, 9.9, 9.95, 10],
     "D435_Abweichung": [0.80, 1.06, 1.00, 1.14, 1.20, 1.28, 1.30, 1.56, 1.50, 1.52, 1.56, 1.80, 2.00, 2.20, 2.00, 1.70, 2.44, 2.20, 2.52, 1.70, 2.60, 3.10, 3.20, 3.20, 2.64, 2.70, 3.40, 3.90, 4.18, 4.80, 4.80, 3.80, 3.32, 3.50, 5.08, 5.30, 5.86, 6.36, 6.20, 6.44, 5.06, 4.70, 4.50, 5.30, 5.30, 7.40, 7.10, 8.10, 9.10, 9.20, 10.00, 9.14, 9.40, 9.12, 7.30, 7.10, 7.00, 6.72, 6.86, 6.98, 7.10, 8.30, 10.60, 11.60, 12.00, 11.70, 12.54, 14.66, 15.74, 16.50, 14.62, 14.10, 14.70, 16.10, 15.20, 14.10, 13.00, 11.70, 9.90, 10.40, 12.80, 12.60, 13.12, 10.28, 8.70, 9.06, 11.10, 13.20, 13.10, 15.30, 15.10, 17.40, 16.80, 19.24, 21.78, 24.00, 24.76, 24.70, 23.00, 26.08, 29.34, 31.46, 30.70, 32.20, 30.90, 31.26, 29.36, 25.32, 26.72, 28.12, 31.60, 28.98, 30.40, 30.68, 27.10, 30.50, 27.00, 28.00, 29.00, 25.30, 26.10, 21.90, 23.38, 23.56, 27.70, 28.30, 28.50, 28.60, 28.80, 29.00, 26.78, 22.12, 19.84, 18.70, 19.22, 22.18, 22.60, 22.10, 25.98, 26.60, 31.50, 25.50, 30.10, 29.00, 34.10, 29.18, 30.40, 31.80, 35.70, 38.24, 32.90, 37.90, 38.82, 41.00, 44.62, 44.50, 49.50, 46.72, 47.00, 51.80, 49.70, 51.92, 53.74, 50.30, 55.00, 53.00, 58.10, 61.46, 59.20, 65.80, 70.80, 69.30, 72.80, 76.26, 75.50, 74.20, 84.10, 77.90, 81.40, 84.70, 85.00],
     "D455_Abweichung": [0.22, 0.40, 0.50, 0.60, 0.40, 0.50, 0.70, 0.70, 0.80, 0.80, 1.20, 1.10, 1.08, 1.32, 1.64, 1.68, 1.92, 2.08, 2.00, 1.64, 2.16, 2.28, 2.16, 2.34, 2.66, 2.12, 2.80, 3.10, 2.50, 3.66, 2.94, 3.62, 3.44, 4.48, 4.12, 4.20, 4.32, 4.40, 5.60, 5.08, 5.58, 5.40, 5.20, 6.56, 6.60, 5.90, 5.40, 6.38, 6.70, 6.30, 7.80, 8.00, 8.30, 8.08, 7.36, 8.20, 8.14, 8.50, 8.20, 10.56, 10.50, 10.70, 10.20, 9.50, 11.32, 11.50, 12.20, 11.04, 11.40, 10.80, 13.04, 13.88, 14.00, 14.60, 14.50, 13.98, 13.00, 14.00, 16.60, 16.90, 15.90, 17.70, 19.50, 18.52, 18.50, 23.90, 22.96, 23.32, 23.12, 23.24, 23.36, 22.66, 22.30, 20.80, 21.00, 25.86, 25.30, 23.80, 24.40, 25.10, 24.40, 24.50, 24.50, 25.20, 25.10, 23.60, 31.00, 32.90, 32.60, 34.70, 32.00, 32.94, 33.80, 33.90, 33.80, 31.20, 33.14, 31.70, 31.50, 29.20, 32.20, 40.34, 39.16, 38.08, 37.50, 38.10, 37.52, 37.44, 36.40, 38.12, 37.20, 39.98, 40.74, 40.66, 38.22, 38.10, 36.92, 37.36, 42.02, 50.30, 53.30, 51.18, 47.90, 49.78, 49.50, 48.70, 48.50, 48.90, 53.60, 55.80, 58.00, 58.60, 55.36, 57.50, 56.30, 53.32, 52.00, 52.00, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
     }

dfStereo = pd.DataFrame(dataStereo)


# Daten als NumPy-Arrays sicherstellen
X = np.array(dataStereo["Rail_distance"])
y = np.array(dataStereo["D435_Abweichung"])

spline = CubicSpline(X, y)

# Berechne die korrigierte Abweichung
y_corrected = y - spline(X)






plt.figure(figsize=(14, 8))

plt.plot(dfStereo["Rail_distance"], y_corrected, 
         label="D435 (300€ - 5,0cm)", color="blue")

plt.plot(dfStereo["Rail_distance"], dfStereo["D435_Abweichung"], 
         label="D435 (300€ - 5,0cm)", color="blue")

plt.plot(dfStereo["Rail_distance"], -dfStereo["D455_Abweichung"], 
         label="D455 (400€ - 9,5cm)", color="black")


plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("D455 und D435", fontsize=14)
plt.xlabel("Echte Abstände [m]", fontsize=12)
plt.ylabel("Abweichung [cm]", fontsize=12)
plt.yticks(np.arange(-100, 100, 10), [f"{abs(y)}" for y in np.arange(-100, 100, 10)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()


# Daten aus der Excel-Tabelle (Beispielwerte für Abweichungen - ersetzen Sie diese!)
data16 = {
    "Fokusabstand [m]": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0],
    "Blende2_vorne [cm]": [2.2, 4.8, 8.4, 12.8, 18, 24, 30.6, 37.8, 45.7, 54.1, 63, 72.3, 82.2, 92.4, 103, 114, 125, 137, 149, 161, 174, 187, 200, 213, 227, 241, 254, 269, 283],
    "Blende2_hinten [cm]": [2.4, 5.7, 10.6, 17.2, 25.7, 36.4, 49.5, 65.3, 84.1, 106, 132, 163, 199, 241, 289, 346, 413, 492, 586, 697, 831, 994, 1196, 1448, 1773, 2201, 2790, 3644, 4982],
    "Blende4_vorne [cm]": [4.1, 8.9, 15.2, 22.7, 31.3, 40.9, 51.3, 62.5, 74.4, 86.8, 99.7, 113, 127, 141, 156, 171, 186, 201, 217, 233, 249, 266, 282, 299, 316, 333, 350, 367, 385],
    "Blende4_hinten [cm]": [5.2, 12.7, 24.4, 41.5, 65.5, 98.4, 143, 205, 290, 411, 591, 874, 1372, 2430, 6052, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    "D435_Abweichung [cm]": [None, None, None, 0.80, 1.20, 1.50, 2.00, 2.44, 2.60, 2.64, 4.18, 3.32, 5.86, 5.06, 5.30, 9.10, 9.40, 7.00, 7.10, 12.00, 15.74, 14.70, 13.00, 12.80, 8.70, 13.10, 16.80, 24.76, 29.34],
    "D455_Abweichung [cm]": [None, None, None, 0.22, 0.40, 0.80, 1.08, 1.92, 2.16, 2.66, 2.50, 3.44, 4.32, 5.58, 6.60, 6.70, 8.30, 8.14, 10.50, 11.32, 11.40, 14.00, 13.00, 15.90, 18.50, 23.12, 22.30, 25.30, 24.40]
}

df = pd.DataFrame(data16)

# Diagramm erstellen
plt.figure(figsize=(14, 8))

# --- Schärfentiefen ---
# Blende 2 (durchgezogene Linien)
plt.plot(df["Fokusabstand [m]"], df["Blende2_vorne [cm]"], 
         label="Blende 2: Vordere Ebene", color="blue", marker="^")
plt.plot(df["Fokusabstand [m]"], -df["Blende2_hinten [cm]"], 
         label="Blende 2: Hintere Ebene", color="blue", marker="v")

# Blende 4 (gestrichelte Linien)
plt.plot(df["Fokusabstand [m]"], df["Blende4_vorne [cm]"], 
         label="Blende 4: Vordere Ebene", color="red", marker="^")
plt.plot(df["Fokusabstand [m]"], -df["Blende4_hinten [cm]"], 
         label="Blende 4: Hintere Ebene", color="red", marker="v")

# --- Abweichungen als separate Linien ---
# D435-Abweichung (grüne Linie)
plt.plot(df["Fokusabstand [m]"], df["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")

# D455-Abweichung (lila Linie)
plt.plot(df["Fokusabstand [m]"], -df["D455_Abweichung [cm]"], 
         label="D455 Abweichung", color="purple", linestyle="-.")

# X-Achse (y=0) und Layout
plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("APSC - Cropfaktor 1.5 - Brennweite 16mm", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Vordere Ebene: + / Hintere Ebene: -)", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

data25 = {
    "Fokusabstand [m]": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0],
    "Blende2_vorne [cm]": [0.9, 2, 3.6, 5.6, 8, 10.8, 14, 17.6, 21.5, 25.8, 30.4, 35.3, 40.6, 46.1, 52, 58.1, 64.5, 71.2, 78.2, 85.4, 92.8, 100, 108, 117, 125, 133, 142, 151, 160],
    "Blende2_hinten [cm]": [0.9, 2.2, 4, 6.3, 9.3, 12.8, 17, 21.9, 27.4, 33.7, 40.7, 48.5, 57.2, 66.6, 77, 88.3, 101, 114, 128, 144, 160, 178, 198, 218, 240, 264, 289, 316, 345],
    "Blende4_vorne [cm]": [1.8, 3.9, 6.9, 10.6, 15.1, 20.1, 25.8, 32.1, 38.9, 46.2, 54, 62.2, 70.9, 80, 89.4, 99.3, 109, 120, 131, 142, 153, 165, 177, 189, 201, 214, 227, 240, 253],
    "Blende4_hinten [cm]": [1.9, 4.5, 8.4, 13.5, 20.1, 28.3, 38.1, 49.8, 63.6, 79.6, 98.1, 119, 144, 171, 203, 239, 279, 325, 378, 437, 505, 583, 672, 775, 894, 1033, 1196, 1390, 1623],
    "D435_Abweichung [cm]": [None, None, None, 0.80, 1.20, 1.50, 2.00, 2.44, 2.60, 2.64, 4.18, 3.32, 5.86, 5.06, 5.30, 9.10, 9.40, 7.00, 7.10, 12.00, 15.74, 14.70, 13.00, 12.80, 8.70, 13.10, 16.80, 24.76, 29.34],
    "D455_Abweichung [cm]": [None, None, None, 0.22, 0.40, 0.80, 1.08, 1.92, 2.16, 2.66, 2.50, 3.44, 4.32, 5.58, 6.60, 6.70, 8.30, 8.14, 10.50, 11.32, 11.40, 14.00, 13.00, 15.90, 18.50, 23.12, 22.30, 25.30, 24.40]

}

df25 = pd.DataFrame(data25)
# Diagramm erstellen
plt.figure(figsize=(14, 8))

# --- Schärfentiefen ---
# Blende 2 (durchgezogene Linien)
plt.plot(df25["Fokusabstand [m]"], df25["Blende2_vorne [cm]"], 
         label="Blende 2: Vordere Ebene", color="blue", marker="^")
plt.plot(df25["Fokusabstand [m]"], -df25["Blende2_hinten [cm]"], 
         label="Blende 2: Hintere Ebene", color="blue", marker="v")

# Blende 4 (gestrichelte Linien)
plt.plot(df25["Fokusabstand [m]"], df25["Blende4_vorne [cm]"], 
         label="Blende 4: Vordere Ebene", color="red", marker="^")
plt.plot(df25["Fokusabstand [m]"], -df25["Blende4_hinten [cm]"], 
         label="Blende 4: Hintere Ebene", color="red", marker="v")

# --- Abweichungen als separate Linien ---
# D435-Abweichung (grüne Linie)
plt.plot(df25["Fokusabstand [m]"], df25["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")

# D455-Abweichung (lila Linie)
plt.plot(df25["Fokusabstand [m]"], -df25["D455_Abweichung [cm]"], 
         label="D455 Abweichung", color="purple", linestyle="-.")

# X-Achse (y=0) und Layout
plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("APSC - Cropfaktor 1.5 - Brennweite 25mm", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Vordere Ebene: + / Hintere Ebene: -)", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

data50 = {
    "Fokusabstand [m]": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0],
    "Blende2_vorne [cm]": [0.2, 0.5, 0.9, 1.4, 2.1, 2.8, 3.7, 4.7, 5.8, 7.0, 8.3, 9.7, 11.3, 12.9, 14.7, 16.5, 18.5, 20.6, 22.8, 25.0, 27.4, 29.9, 32.5, 35.1, 37.9, 40.8, 43.8, 46.8, 50.0, 53.2, 56.6, 60.0, 63.6, 67.2, 70.9, 74.7, 78.6, 82.6, 86.6, 90.8, 95.0, 99.3, 104.0, 108.0, 113.0, 117.0, 122.0, 127.0, 134.0],
    "Blende2_hinten [cm]": [0.2, 0.5, 0.9, 1.5, 2.1, 2.9, 3.9, 4.9, 6.1, 7.5, 8.9, 10.5, 12.3, 14.2, 16.2, 18.3, 20.6, 23.1, 25.7, 28.4, 31.3, 34.4, 37.5, 40.9, 44.4, 48.0, 51.9, 55.8, 60.0, 64.3, 68.7, 73.4, 78.2, 83.1, 88.3, 93.6, 99.1, 105.0, 111.0, 117.0, 123.0, 129.0, 136.0, 143.0, 149.0, 157.0, 164.0, 171.0, 183.0],
    "Blende4_vorne [cm]": [0.4, 1.0, 1.0, 2.8, 4.1, 5.5, 7.2, 9.1, 11.2, 13.6, 16.1, 18.8, 21.7, 24.8, 28.1, 31.6, 35.2, 39.1, 43.1, 47.3, 51.6, 56.1, 60.8, 65.7, 70.7, 75.9, 81.2, 86.7, 92.3, 98.1, 104.0, 110.0, 116.0, 123.0, 129.0, 136.0, 142.0, 149.0, 156.0, 163.0, 171.0, 178.0, 186.0, 193.0, 201.0, 209.0, 217.0, 225.0, 237.0],
    "Blende4_hinten [cm]": [0.4, 1.0, 1.0, 3.0, 4.4, 6.0, 8.0, 10.2, 12.7, 15.5, 18.6, 22.0, 25.7, 29.7, 34.1, 38.8, 43.8, 49.2, 54.9, 61.0, 67.4, 74.3, 81.5, 89.1, 97.1, 105.0, 114.0, 124.0, 133.0, 143.0, 154.0, 165.0, 177.0, 189.0, 201.0, 214.0, 228.0, 242.0, 257.0, 272.0, 288.0, 304.0, 321.0, 339.0, 357.0, 376.0, 395.0, 416.0, 447.0],
    "D435_Abweichung [cm]": [None, None, None, 0.80, 1.20, 1.50, 2.00, 2.44, 2.60, 2.64, 4.18, 3.32, 5.86, 5.06, 5.30, 9.10, 9.40, 7.00, 7.10, 12.00, 15.74, 14.70, 13.00, 12.80, 8.70, 13.10, 16.80, 24.76, 29.34, 30.90, 26.72, 30.40, 27.00, 26.10, 27.70, 28.80, 19.84, 22.60, 31.50, 34.10, 35.70, 38.82, 49.50, 49.70, 55.00, 59.20, 72.80, 84.10, 85.00],
    "D455_Abweichung [cm]": [None, None, None, 0.22, 0.40, 0.80, 1.08, 1.92, 2.16, 2.66, 2.50, 3.44, 4.32, 5.58, 6.60, 6.70, 8.30, 8.14, 10.50, 11.32, 11.40, 14.00, 13.00, 15.90, 18.50, 23.12, 22.30, 25.30, 24.40, 25.10, 32.60, 33.80, 33.14, 32.20, 37.50, 36.40, 40.74, 36.92, 53.30, 49.50, 53.60, 55.36, 52.00, None, None, None, None, None, None]   
}

df50 = pd.DataFrame(data50)

# Diagramm erstellen
plt.figure(figsize=(14, 8))

# --- Schärfentiefen ---
# Blende 2 (durchgezogene Linien)
plt.plot(df50["Fokusabstand [m]"], df50["Blende2_vorne [cm]"], 
         label="Blende 2: Vordere Ebene", color="blue", marker="^")
plt.plot(df50["Fokusabstand [m]"], -df50["Blende2_hinten [cm]"], 
         label="Blende 2: Hintere Ebene", color="blue", marker="v")

# Blende 4 (gestrichelte Linien)
plt.plot(df50["Fokusabstand [m]"], df50["Blende4_vorne [cm]"], 
         label="Blende 4: Vordere Ebene", color="red", marker="^")
plt.plot(df50["Fokusabstand [m]"], -df50["Blende4_hinten [cm]"], 
         label="Blende 4: Hintere Ebene", color="red", marker="v")

# --- Abweichungen als separate Linien ---
# D435-Abweichung (grüne Linie)
plt.plot(df50["Fokusabstand [m]"], df50["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")

# D455-Abweichung (lila Linie)
plt.plot(df50["Fokusabstand [m]"], -df50["D455_Abweichung [cm]"], 
         label="D455 Abweichung", color="purple", linestyle="-.")

# X-Achse (y=0) und Layout
plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("APSC - Cropfaktor 1.5 - Brennweite 50mm", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Vordere Ebene: + / Hintere Ebene: -)", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()


plt.figure(figsize=(14, 8))


plt.plot(df50["Fokusabstand [m]"], df50["Blende2_vorne [cm]"], 
         label="Blende 2: Vordere Ebene", color="blue", marker="^")
plt.plot(df50["Fokusabstand [m]"], -df50["Blende2_hinten [cm]"], 
         label="Blende 2: Hintere Ebene", color="blue", marker="v")

plt.plot(df50["Fokusabstand [m]"], df50["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")

# Blende 4 (gestrichelte Linien)
plt.plot(df50["Fokusabstand [m]"], df50["Blende2_vorne [cm]"]+df50["D435_Abweichung [cm]"], 
         label="Blende 2: Verschobene vordere Ebene", color="black", marker="^", linestyle="-.")
plt.plot(df50["Fokusabstand [m]"], -df50["Blende2_hinten [cm]"]+df50["D435_Abweichung [cm]"], 
         label="Blende 2: Verschobene hintere Ebene", color="black", marker="v", linestyle="-.")


plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("Verschobene Schärfentiefe - 50mm - Blende 2", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Vordere Ebene: + / Hintere Ebene: -)", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()


# Daten aus der Excel-Tabelle (Beispielwerte für Abweichungen - ersetzen Sie diese!)
data75 = {
    "Fokusabstand [m]": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.8, 5.0, 5.4, 5.6, 6.0, 6.2, 6.6, 6.8, 7.2, 7.4, 7.8, 8.0, 8.4, 8.6, 9.0, 9.2, 9.6, 10.0],
    "Blende2_vorne [cm]": [0.1, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.1, 2.6, 3.1, 3.7, 4.4, 5.1, 5.8, 6.6, 7.5, 8.4, 9.4, 10.4, 11.4, 12.5, 14.9, 16.2, 18.6, 20.2, 23.2, 24.7, 28.0, 29.7, 33.2, 35.1, 38.9, 40.8, 44.9, 47.0, 51.4, 53.7, 58.3, 63.1],
    "Blende2_hinten [cm]": [0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.7, 2.1, 2.6, 3.2, 3.8, 4.5, 5.3, 6.1, 6.9, 7.9, 8.8, 9.9, 10.9, 12.1, 13.3, 15.9, 17.3, 20.3, 21.8, 25.1, 26.9, 30.6, 32.5, 36.6, 38.7, 43.2, 45.5, 50.3, 52.8, 58.0, 60.7, 66.4, 72.2],
    "Blende4_vorne [cm]": [0.2, 0.4, 0.8, 1.2, 1.8, 2.5, 3.2, 4.1, 5.1, 6.2, 7.3, 8.6, 10.0, 11.5, 13.0, 14.7, 16.4, 18.3, 20.2, 22.3, 24.4, 28.9, 31.3, 36.4, 39.1, 44.7, 47.6, 53.7, 56.9, 63.5, 66.9, 74.0, 77.7, 85.3, 89.2, 97.3, 101.0, 110.0, 119.0],
    "Blende4_hinten [cm]": [0.2, 0.4, 0.8, 1.3, 1.9, 2.6, 3.4, 4.3, 5.4, 6.5, 7.8, 9.2, 10.8, 12.4, 14.2, 16.1, 18.1, 20.2, 22.5, 24.9, 27.4, 32.9, 35.8, 42.1, 45.4, 52.5, 56.2, 64.1, 68.3, 77.1, 81.7, 91.4, 96.4, 107.0, 113.0, 124.0, 130.0, 143.0, 156.0],
    "Blende8_vorne [cm]": [0.3, 0.8, 1.5, 2.5, 3.6, 4.9, 6.4, 8.1, 9.9, 12.0, 14.3, 16.7, 19.3, 22.1, 25.0, 28.2, 31.4, 34.9, 38.5, 42.3, 46.2, 54.6, 59.0, 68.2, 73.0, 83.1, 88.4, 99.3, 105.0, 117.0, 123.0, 135.0, 142.0, 155.0, 162.0, 176.0, 183.0, 197.0, 212.0],
    "Blende8_hinten [cm]": [0.3, 0.9, 1.6, 2.6, 3.8, 5.2, 6.9, 8.8, 11.0, 13.5, 16.2, 19.1, 22.4, 25.9, 29.7, 33.7, 38.1, 42.8, 47.7, 53.0, 58.5, 70.6, 77.2, 91.3, 98.8, 115.0, 124.0, 142.0, 152.0, 173.0, 184.0, 207.0, 219.0, 245.0, 259.0, 288.0, 303.0, 335.0, 369.0],
    "D435_Abweichung [cm]": [None, None, None, 0.80, 1.20, 1.50, 2.00, 2.44, 2.60, 2.64, 4.18, 3.32, 5.86, 5.06, 5.30, 9.10, 9.40, 7.00, 7.10, 12.00, 15.74, 13.00, 12.80, 13.10, 16.80, 29.34, 30.90, 30.40, 27.00, 27.70, 28.80, 22.60, 31.50, 35.70, 38.82, 49.70, 55.00, 72.80, 85.00],
    "D455_Abweichung [cm]": [None, None, None, 0.22, 0.40, 0.80, 1.08, 1.92, 2.16, 2.66, 2.50, 3.44, 4.32, 5.58, 6.60, 6.70, 8.30, 8.14, 10.50, 11.32, 11.40, 13.00, 15.90, 23.12, 22.30, 24.40, 25.10, 33.80, 33.14, 37.50, 36.40, 36.92, 53.30, 53.60, 55.36, None, None, None, None]   
}

df75 = pd.DataFrame(data75)

# Diagramm erstellen
plt.figure(figsize=(14, 8))

# --- Schärfentiefen ---
# Blende 2 (durchgezogene Linien)
plt.plot(df75["Fokusabstand [m]"], df75["Blende2_vorne [cm]"], 
         label="Blende 2: Vordere Ebene", color="blue", marker="^")
plt.plot(df75["Fokusabstand [m]"], -df75["Blende2_hinten [cm]"], 
         label="Blende 2: Hintere Ebene", color="blue", marker="v")

# Blende 4 (gestrichelte Linien)
plt.plot(df75["Fokusabstand [m]"], df75["Blende4_vorne [cm]"], 
         label="Blende 4: Vordere Ebene", color="red", marker="^")
plt.plot(df75["Fokusabstand [m]"], -df75["Blende4_hinten [cm]"], 
         label="Blende 4: Hintere Ebene", color="red", marker="v")


# Blende 8 (gestrichelte Linien)
plt.plot(df75["Fokusabstand [m]"], df75["Blende8_vorne [cm]"], 
         label="Blende 8: Vordere Ebene", color="black", marker="^")
plt.plot(df75["Fokusabstand [m]"], -df75["Blende8_hinten [cm]"], 
         label="Blende 8: Hintere Ebene", color="black", marker="v")

# --- Abweichungen als separate Linien ---
# D435-Abweichung (grüne Linie)
plt.plot(df75["Fokusabstand [m]"], df75["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")

# D455-Abweichung (lila Linie)
plt.plot(df75["Fokusabstand [m]"], -df75["D455_Abweichung [cm]"], 
         label="D455 Abweichung", color="purple", linestyle="-.")

# X-Achse (y=0) und Layout
plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("APSC - Cropfaktor 1.5 - Brennweite 75mm", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Hintere Ebene: - / Vordere Ebene: +", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()



plt.figure(figsize=(14, 8))

# --- Schärfentiefen ---
# Blende 2 (durchgezogene Linien)
plt.plot(df75["Fokusabstand [m]"], df75["Blende2_vorne [cm]"]+df75["D435_Abweichung [cm]"], 
         label="Blende 2: Vordere Ebene", color="blue", marker="^")
plt.plot(df75["Fokusabstand [m]"], -df75["Blende2_hinten [cm]"]+df75["D435_Abweichung [cm]"], 
         label="Blende 2: Hintere Ebene", color="blue", marker="v")

# Blende 4 (gestrichelte Linien)
plt.plot(df75["Fokusabstand [m]"], df75["Blende4_vorne [cm]"]+df75["D435_Abweichung [cm]"], 
         label="Blende 4: Vordere Ebene", color="red", marker="^")
plt.plot(df75["Fokusabstand [m]"], -df75["Blende4_hinten [cm]"]+df75["D435_Abweichung [cm]"], 
         label="Blende 4: Hintere Ebene", color="red", marker="v")


# Blende 8 (gestrichelte Linien)
plt.plot(df75["Fokusabstand [m]"], df75["Blende8_vorne [cm]"]+df75["D435_Abweichung [cm]"], 
         label="Blende 8: Vordere Ebene", color="black", marker="^")
plt.plot(df75["Fokusabstand [m]"], -df75["Blende8_hinten [cm]"]+df75["D435_Abweichung [cm]"], 
         label="Blende 8: Hintere Ebene", color="black", marker="v")

# --- Abweichungen als separate Linien ---
# D435-Abweichung (grüne Linie)
plt.plot(df75["Fokusabstand [m]"], df75["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")


# X-Achse (y=0) und Layout
plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("Verschobene Schärfentiefe 75mm", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Hintere Ebene: - / Vordere Ebene: +", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()




data100 = {
    "Fokusabstand [m]": [1, 1.2, 1.6, 2, 2.4, 2.8, 3, 3.4, 3.8, 4, 4.4, 5, 5.6, 6, 6.6, 7.2, 7.4, 7.8, 8, 8.4, 9, 9.6, 10],
    "Blende2_vorne [cm]": [0.3, 0.5, 0.9, 1.4, 2.1, 2.9, 3.3, 4.2, 5.3, 5.9, 7.1, 9.2, 11.5, 13.2, 16, 19, 20.1, 22.3, 23.4, 25.8, 29.6, 33.6, 36.4],
    "Blende2_hinten [cm]": [0.3, 0.5, 0.9, 1.5, 2.1, 2.9, 3.4, 4.3, 5.4, 6, 7.3, 9.5, 12, 13.8, 16.8, 20.1, 21.2, 23.6, 24.9, 27.5, 31.7, 36.1, 39.3],
    "Blende4_vorne [cm]": [0.7, 1, 1.8, 2.9, 4.1, 5.9, 6.5, 8.4, 10.4, 11.6, 14, 18, 22.6, 25.9, 31.2, 37, 39.1, 43.3, 45.5, 50.1, 57.3, 64.9, 70.3],
    "Blende4_hinten [cm]": [0.7, 1, 1.9, 2.9, 4.3, 6, 6.8, 8.8, 11, 12.3, 14.9, 19.4, 24.6, 28.3, 34.5, 41.3, 43.7, 48.7, 51.4, 56.8, 65.6, 75.1, 81.8],
    "Blende8_vorne [cm]": [1.4, 2, 3.6, 5.6, 8.1, 11.1, 12.7, 16.3, 20.3, 22.5, 27.1, 34.8, 43.4, 49.6, 59.6, 70.4, 74.2, 82.1, 86.1, 94.5, 108, 122, 131],
    "Blende8_hinten [cm]": [1.4, 2.1, 3.8, 6, 8.7, 12, 13.9, 18, 22.8, 25.3, 30.9, 40.4, 51.4, 59.4, 72.7, 87.6, 92.9, 104, 110, 122, 142, 163, 178],
    "D435_Abweichung [cm]": [0.80, 1.20, 2.00, 2.60, 4.18, 5.86, 5.06, 9.10, 7.00, 7.10, 15.74, 12.80, 16.80, 29.34, 30.40, 27.70, 28.80, 22.60, 31.50, 35.70, 49.70, 72.80, 85.00],
    "D455_Abweichung [cm]": [0.22, 0.40, 1.08, 2.16, 2.50, 4.32, 5.58, 6.70, 8.14, 10.50, 11.40, 15.90, 22.30, 24.40, 33.80, 37.50, 36.40, 36.92, 53.30, 53.60, None, None, None]   
}

df100 = pd.DataFrame(data100)

# Diagramm erstellen
plt.figure(figsize=(14, 8))

# --- Schärfentiefen ---
# Blende 2 (durchgezogene Linien)
plt.plot(df100["Fokusabstand [m]"], df100["Blende2_vorne [cm]"], 
         label="Blende 2: Vordere Ebene", color="blue", marker="^")
plt.plot(df100["Fokusabstand [m]"], -df100["Blende2_hinten [cm]"], 
         label="Blende 2: Hintere Ebene", color="blue", marker="v")

# Blende 4 (gestrichelte Linien)
plt.plot(df100["Fokusabstand [m]"], df100["Blende4_vorne [cm]"], 
         label="Blende 4: Vordere Ebene", color="red", marker="^")
plt.plot(df100["Fokusabstand [m]"], -df100["Blende4_hinten [cm]"], 
         label="Blende 4: Hintere Ebene", color="red", marker="v")


# Blende 8 (gestrichelte Linien)
plt.plot(df100["Fokusabstand [m]"], df100["Blende8_vorne [cm]"], 
         label="Blende 8: Vordere Ebene", color="black", marker="^")
plt.plot(df100["Fokusabstand [m]"], -df100["Blende8_hinten [cm]"], 
         label="Blende 8: Hintere Ebene", color="black", marker="v")

# --- Abweichungen als separate Linien ---
# D435-Abweichung (grüne Linie)
plt.plot(df100["Fokusabstand [m]"], df100["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")

# D455-Abweichung (lila Linie)
plt.plot(df100["Fokusabstand [m]"], -df100["D455_Abweichung [cm]"], 
         label="D455 Abweichung", color="purple", linestyle="-.")

# X-Achse (y=0) und Layout
plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("APSC - Cropfaktor 1.5 - Brennweite 100mm", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Hintere Ebene: - / Vordere Ebene: +", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()


plt.figure(figsize=(14, 8))



plt.plot(df100["Fokusabstand [m]"], df100["Blende2_vorne [cm]"]+df100["D435_Abweichung [cm]"], 
         label="Blende 2: Verschobene vordere Ebene", color="blue", marker="^", linestyle="-.")
plt.plot(df100["Fokusabstand [m]"], -df100["Blende2_hinten [cm]"]+df100["D435_Abweichung [cm]"], 
         label="Blende 2: Verschobene hintere Ebene", color="blue", marker="v", linestyle="-.")

plt.plot(df100["Fokusabstand [m]"], df100["D435_Abweichung [cm]"], 
         label="D435 Abweichung", color="green", linestyle="-.")

# Blende 4 (gestrichelte Linien)
plt.plot(df100["Fokusabstand [m]"], df100["Blende4_vorne [cm]"]+df100["D435_Abweichung [cm]"], 
         label="Blende 4: Verschobene vordere Ebene", color="black", marker="^", linestyle="-.")
plt.plot(df100["Fokusabstand [m]"], -df100["Blende4_hinten [cm]"]+df100["D435_Abweichung [cm]"], 
         label="Blende 4: Verschobene hintere Ebene", color="black", marker="v", linestyle="-.")


plt.plot(df100["Fokusabstand [m]"], df100["Blende8_vorne [cm]"]+df100["D435_Abweichung [cm]"], 
         label="Blende 8: Verschobene vordere Ebene", color="red", marker="^", linestyle="-.")
plt.plot(df100["Fokusabstand [m]"], -df100["Blende8_hinten [cm]"]+df100["D435_Abweichung [cm]"], 
         label="Blende 8: Verschobene hintere Ebene", color="red", marker="v", linestyle="-.")


plt.axhline(y=0, color="black", linewidth=0.8)
plt.title("Verschobene Schärfentiefe - 100mm", fontsize=14)
plt.xlabel("Fokusabstand [m]", fontsize=12)
plt.ylabel("Vordere Ebene: + / Hintere Ebene: -)", fontsize=12)
plt.yticks(np.arange(-150, 151, 20), [f"{abs(y)}" for y in np.arange(-150, 151, 20)])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()




