import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, median, multimode, pstdev, pvariance

# ----------------------------------------------------------------------
# DATOS REALES DE NOTAS
# ----------------------------------------------------------------------
notas = [
    30,50,20,40,30,20,30,40,40,50,
    50,20,40,30,20,40,30,30,30,20,20,
    40,30,40,50,20,50,30,30,40,30,20,
    30,40,20,30,50,50,20,30,40,20,20,
    40,50,30,20,30,30,50,50,20,10,20,
    30,30,40,40,30,50,30,20,20,10
]

df = pd.DataFrame({"nota": notas})

# ----------------------------------------------------------------------
# MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN
# ----------------------------------------------------------------------

# Frecuencia absoluta (para gráficos de barras)
abs_freq = df["nota"].value_counts().sort_index()

# Medidas de tendencia central
media = mean(notas)
mediana = median(notas)
moda = multimode(notas)

# Medidas de dispersión (p: poblacional)
varianza = pvariance(notas) # varianza poblacional
desviacion_std = pstdev(notas) # desviación estándar poblacional

# ----------------------------------------------------------------------
# IMPRESIÓN DE RESULTADOS
# ----------------------------------------------------------------------

print("\nMedidas de tendencia central:")
print(f"Media: {media}")
print(f"Mediana: {mediana}")
print(f"Moda: {moda}")

print("\nMedidas de dispersión:")
print(f"Varianza: {varianza}")
print(f"Desviación estándar: {desviacion_std}")

# ----------------------------------------------------------------------
# GRÁFICOS
# ----------------------------------------------------------------------

# 1. Diagrama de barras de frecuencia
plt.figure(figsize=(8,5))
plt.bar(abs_freq.index, abs_freq.values)
plt.title("Cantidad de personas por calificación")
plt.xlabel("Calificación")
plt.ylabel("Número de personas")
plt.xticks(abs_freq.index)
plt.show()

# 2. Gráfico de la media (sobre histograma)
plt.figure(figsize=(8,5))
plt.hist(notas, bins=10)
plt.axvline(media, linestyle="--", color='red')
plt.title("Media de las calificaciones")
plt.xlabel("Notas")
plt.ylabel("Frecuencia")
plt.show()

# 3. Gráfico de la mediana (sobre histograma)
plt.figure(figsize=(8,5))
plt.hist(notas, bins=10)
plt.axvline(mediana, linestyle="--", color='green')
plt.title("Mediana de las calificaciones")
plt.xlabel("Notas")
plt.ylabel("Frecuencia")
plt.show()

# 4. Gráfico de la moda (sobre histograma)
plt.figure(figsize=(8,5))
plt.hist(notas, bins=10)
for m in moda:
    plt.axvline(m, linestyle="--", color='orange')
plt.title("Moda(s) de las calificaciones")
plt.xlabel("Notas")
plt.ylabel("Frecuencia")
plt.show()

# 5. Gráfico de la desviación estándar (sobre histograma)
plt.figure(figsize=(8,5))
plt.hist(notas, bins=10)
plt.axvline(media, linestyle="--", color='red', label="Media")
plt.axvline(media + desviacion_std, linestyle="-.", color='blue', label="+1 Desv.Est")
plt.axvline(media - desviacion_std, linestyle="-.", color='blue', label="-1 Desv.Est")
plt.title("Desviación estándar sobre la distribución")
plt.xlabel("Notas")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()

# 6. GRÁFICO DE LÍNEA con Cuartiles, Media, Moda y Mediana
# (Se asume que deseas usar el DataFrame de frecuencias para la línea)
# Generamos los cuartiles Q1, Q2, Q3
Q1 = df['nota'].quantile(0.25)
Q2 = df['nota'].quantile(0.50) # Mediana
Q3 = df['nota'].quantile(0.75)

plt.figure(figsize=(10,5))

# Línea de frecuencias
plt.plot(abs_freq.index, abs_freq.values, marker="o", linewidth=2, label="Frecuencia Absoluta")

# Marcadores de medidas de tendencia central y cuartiles
plt.axvline(Q1, color='purple', linestyle="--", label="Q1")
plt.axvline(Q2, color='green', linestyle="--", label="Q2 (Mediana)")
plt.axvline(Q3, color='blue', linestyle="--", label="Q3")
plt.axvline(media, color='red', linestyle="--", label="Media")

# Modas
for m in moda:
    plt.axvline(m, color="orange", linestyle="-.", label=f"Moda: {m}")

plt.title("Diagrama de Línea con Cuartiles, Media, Moda y Mediana")
plt.xlabel("Calificación")
plt.ylabel("Frecuencia")
plt.xticks(abs_freq.index)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()