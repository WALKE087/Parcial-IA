import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------------ Implementación Perceptrón ------------------------------
class PerceptronSimple:
    def __init__(self, entradas, salidas=1, pesos=None, umbral=0.0, con_sesgo=True):
        self.entradas = entradas
        self.salidas = salidas
        self.con_sesgo = con_sesgo
        if pesos is None:
            forma = (salidas, entradas + (1 if con_sesgo else 0))
            self.pesos = np.random.uniform(-0.5, 0.5, size=forma)
        else:
            self.pesos = np.array(pesos, dtype=float)
        self.umbral = float(umbral)

    def funcion_escalon(self, x):
        return np.where(x >= 0.0, 1.0, 0.0)

    def calcular_entrada(self, X):
        if self.con_sesgo:
            Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        else:
            Xb = X
        return Xb @ self.pesos.T

    def predecir(self, X):
        neto = self.calcular_entrada(np.atleast_2d(X))
        return self.funcion_escalon(neto)

    def entrenar_regla_perceptron(self, X, Y, tasa=0.1, max_iter=100, error_min=1e-3, mostrar=None):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))]) if self.con_sesgo else X
        historial_errores = []
        for epoca in range(1, max_iter + 1):
            error_total = 0.0
            for patron, salida_esperada in zip(Xb, Y):
                neto = self.pesos @ patron
                salida_predicha = self.funcion_escalon(neto)
                error = salida_esperada - salida_predicha
                self.pesos += tasa * np.outer(error, patron)
                error_total += np.mean(np.abs(error))
            rms = np.sqrt(error_total**2 / X.shape[0])
            historial_errores.append(rms)
            if mostrar:
                mostrar(epoca, rms)
            if rms <= error_min:
                break
        return historial_errores

    def entrenar_regla_delta(self, X, Y, tasa=0.01, max_iter=100, error_min=1e-4, mostrar=None):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))]) if self.con_sesgo else X
        historial_errores = []
        for epoca in range(1, max_iter + 1):
            neto = Xb @ self.pesos.T
            error = Y - neto
            ajuste_pesos = tasa * (error.T @ Xb) / X.shape[0]
            self.pesos += ajuste_pesos
            mse = np.mean(error**2)
            rms = np.sqrt(mse)
            historial_errores.append(rms)
            if mostrar:
                mostrar(epoca, rms)
            if rms <= error_min:
                break
        return historial_errores

# ------------------------------ Funciones de utilidad ------------------------------

def detectar_datos(tabla):
    numericas = tabla.select_dtypes(include=[np.number])
    if numericas.shape[1] == 0:
        raise ValueError('No se encontraron columnas numéricas en los datos')
    posibles = [c for c in numericas.columns if c.lower() in ('target', 'label', 'class', 'y')]
    if posibles:
        col_salida = [posibles[-1]]
    else:
        col_salida = [numericas.columns[-1]]
    col_entrada = [c for c in numericas.columns if c not in col_salida]
    return col_entrada, col_salida, len(col_entrada), len(col_salida)

def normalizar_datos(X):
    minimo = X.min(axis=0)
    maximo = X.max(axis=0)
    rango = (maximo - minimo)
    rango[rango == 0] = 1.0
    Xn = (X - minimo) / rango
    return Xn, minimo, maximo

# ------------------------------ Interfaz gráfica ------------------------------
class AplicacionPerceptron(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Taller: Perceptrón Simple')
        self.geometry('1000x700')

        self.tabla = None
        self.col_entrada = []
        self.col_salida = []
        self.X = None
        self.Y = None
        self.perceptron = None
        self.errores = []

        self.construir_interfaz()

    def construir_interfaz(self):
        izquierda = ttk.Frame(self)
        izquierda.pack(side='left', fill='y', padx=8, pady=8)

        ttk.Label(izquierda, text='Datos').pack(anchor='w')
        ttk.Button(izquierda, text='Cargar datos', command=self.cargar_datos).pack(fill='x')
        self.info_datos = tk.Text(izquierda, height=6, width=40)
        self.info_datos.pack()

        ttk.Separator(izquierda, orient='horizontal').pack(fill='x', pady=6)

        ttk.Label(izquierda, text='Parámetros').pack(anchor='w')
        marco_param = ttk.Frame(izquierda)
        marco_param.pack(fill='x')

        ttk.Label(marco_param, text='Tasa (η):').grid(row=0, column=0, sticky='w')
        self.tasa_var = tk.DoubleVar(value=0.1)
        ttk.Entry(marco_param, textvariable=self.tasa_var).grid(row=0, column=1)

        ttk.Label(marco_param, text='Máx iter:').grid(row=1, column=0, sticky='w')
        self.max_iter_var = tk.IntVar(value=200)
        ttk.Entry(marco_param, textvariable=self.max_iter_var).grid(row=1, column=1)

        ttk.Label(marco_param, text='Error ε:').grid(row=2, column=0, sticky='w')
        self.error_min_var = tk.DoubleVar(value=1e-3)
        ttk.Entry(marco_param, textvariable=self.error_min_var).grid(row=2, column=1)

        ttk.Label(marco_param, text='Umbral:').grid(row=3, column=0, sticky='w')
        self.umbral_var = tk.DoubleVar(value=0.0)
        ttk.Entry(marco_param, textvariable=self.umbral_var).grid(row=3, column=1)

        ttk.Label(marco_param, text='Pesos iniciales:').grid(row=4, column=0, sticky='w')
        self.pesos_iniciales_var = tk.StringVar(value='aleatorio')
        ttk.Combobox(marco_param, textvariable=self.pesos_iniciales_var, values=['aleatorio', 'manual']).grid(row=4, column=1)

        ttk.Label(marco_param, text='Algoritmo:').grid(row=5, column=0, sticky='w')
        self.algoritmo_var = tk.StringVar(value='delta')
        ttk.Combobox(marco_param, textvariable=self.algoritmo_var, values=['delta', 'perceptron']).grid(row=5, column=1)

        ttk.Button(izquierda, text='Configurar perceptrón', command=self.configurar_perceptron).pack(fill='x', pady=6)
        ttk.Button(izquierda, text='Entrenar', command=self.iniciar_entrenamiento).pack(fill='x')

        ttk.Separator(izquierda, orient='horizontal').pack(fill='x', pady=6)

        ttk.Label(izquierda, text='Pruebas').pack(anchor='w')
        ttk.Button(izquierda, text='Probar con dataset', command=self.probar_datos).pack(fill='x')
        ttk.Button(izquierda, text='Probar manual', command=self.abrir_prueba_manual).pack(fill='x')

        derecha = ttk.Frame(self)
        derecha.pack(side='right', fill='both', expand=True)

        self.figura = Figure(figsize=(6,4))
        self.grafica = self.figura.add_subplot(111)
        self.grafica.set_xlabel('Iteración')
        self.grafica.set_ylabel('Error RMS')
        self.canvas = FigureCanvasTkAgg(self.figura, master=derecha)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.registro = tk.Text(derecha, height=10)
        self.registro.pack(fill='x')

    def mostrar_registro(self, mensaje):
        self.registro.insert('end', mensaje + '\n')
        self.registro.see('end')

    def cargar_datos(self):
        ruta = filedialog.askopenfilename(filetypes=[('CSV','*.csv'),('Excel','*.xlsx;*.xls'),('JSON','*.json'),('Todos','*.*')])
        if not ruta:
            return
        try:
            if ruta.lower().endswith('.csv'):
                tabla = pd.read_csv(ruta)
            elif ruta.lower().endswith(('.xls','.xlsx')):
                tabla = pd.read_excel(ruta)
            elif ruta.lower().endswith('.json'):
                tabla = pd.read_json(ruta)
            else:
                tabla = pd.read_csv(ruta)
        except Exception as e:
            messagebox.showerror('Error', f'No se pudo leer el archivo: {e}')
            return
        self.tabla = tabla
        try:
            self.col_entrada, self.col_salida, n_in, n_out = detectar_datos(tabla)
        except Exception as e:
            messagebox.showerror('Error', f'Detección automática falló: {e}')
            return
        patrones = tabla.shape[0]
        info = f"Filas: {patrones}\nEntradas: {n_in} -> {self.col_entrada}\nSalidas: {n_out} -> {self.col_salida}"
        self.info_datos.delete('1.0','end')
        self.info_datos.insert('1.0', info)
        self.mostrar_registro('Datos cargados correctamente')
        X = tabla[self.col_entrada].to_numpy(dtype=float)
        Y = tabla[self.col_salida].to_numpy(dtype=float)
        Xn, self.min_x, self.max_x = normalizar_datos(X)
        if Y.ndim == 1:
            Y = Y.reshape(-1,1)
        self.X = Xn
        if not np.all(np.isin(np.unique(Y), [0,1])):
            th = np.median(Y, axis=0)
            Yb = (Y >= th).astype(float)
            self.mostrar_registro('Salida no binaria detectada: binarizando por mediana')
            Y = Yb
        self.Y = Y
        self.mostrar_registro(info)

    def configurar_perceptron(self):
        if self.X is None:
            messagebox.showwarning('Atención', 'Cargue un dataset primero')
            return
        n_in = self.X.shape[1]
        n_out = self.Y.shape[1]
        modo = self.pesos_iniciales_var.get()
        if modo == 'aleatorio':
            p = PerceptronSimple(n_in, n_out, umbral=self.umbral_var.get())
        else:
            w_str = tk.simpledialog.askstring('Pesos manuales', f'Ingrese {n_out} grupos de {n_in+1} valores (incluye sesgo). Ejemplo: "0.1,0.2,0.3"')
            if not w_str:
                return
            try:
                valores = [float(x.strip()) for x in w_str.split(',')]
                esperado = (n_in + 1) * n_out
                if len(valores) != esperado:
                    messagebox.showerror('Error', f'Se esperaban {esperado} valores')
                    return
                pesos = np.array(valores).reshape((n_out, n_in + 1))
                p = PerceptronSimple(n_in, n_out, pesos=pesos, umbral=self.umbral_var.get())
            except Exception as e:
                messagebox.showerror('Error', f'Error en pesos: {e}')
                return
        self.perceptron = p
        self.mostrar_registro(f'Perceptrón configurado: entradas={n_in}, salidas={n_out}')

    def iniciar_entrenamiento(self):
        if self.perceptron is None or self.X is None:
            messagebox.showwarning('Atención', 'Configure perceptrón y cargue los datos')
            return
        tasa = float(self.tasa_var.get())
        max_iter = int(self.max_iter_var.get())
        error_min = float(self.error_min_var.get())
        algoritmo = self.algoritmo_var.get()
        self.grafica.clear()
        self.grafica.set_xlabel('Iteración')
        self.grafica.set_ylabel('Error RMS')
        self.canvas.draw()
        self.errores = []

        def retroalimentacion(epoca, rms):
            self.errores.append((epoca, rms))
            if epoca % max(1, max_iter//100) == 0 or epoca == 1:
                iteraciones, errores = zip(*self.errores)
                self.grafica.clear()
                self.grafica.plot(iteraciones, errores)
                self.grafica.set_xlabel('Iteración')
                self.grafica.set_ylabel('Error RMS')
                self.canvas.draw()
                self.mostrar_registro(f'Iter {epoca}: RMS={rms:.6f}')
                self.update()

        if algoritmo == 'delta':
            self.perceptron.entrenar_regla_delta(self.X, self.Y, tasa=tasa, max_iter=max_iter, error_min=error_min, mostrar=retroalimentacion)
        else:
            self.perceptron.entrenar_regla_perceptron(self.X, self.Y, tasa=tasa, max_iter=max_iter, error_min=error_min, mostrar=retroalimentacion)
        self.mostrar_registro('Entrenamiento finalizado')

    def probar_datos(self):
        if self.perceptron is None or self.tabla is None:
            messagebox.showwarning('Atención', 'Configure perceptrón y cargue datos')
            return
        Xraw = self.tabla[self.col_entrada].to_numpy(dtype=float)
        rango = (self.max_x - self.min_x)
        rango[rango == 0] = 1.0
        Xn = (Xraw - self.min_x) / rango
        predicciones = self.perceptron.predecir(Xn)
        esperado = self.tabla[self.col_salida].to_numpy()
        if esperado.ndim == 1:
            esperado = esperado.reshape(-1,1)
        if not np.all(np.isin(np.unique(esperado), [0,1])):
            th = np.median(esperado, axis=0)
            esperado = (esperado >= th).astype(float)
        win = tk.Toplevel(self)
        win.title('Resultados del dataset')
        texto = tk.Text(win, width=80, height=20)
        texto.pack()
        texto.insert('end', 'Fila\tPredicho\tEsperado\n')
        for i, (p, e) in enumerate(zip(predicciones, esperado)):
            texto.insert('end', f'{i}\t{p} \t{e}\n')

    def abrir_prueba_manual(self):
        if self.perceptron is None or self.X is None:
            messagebox.showwarning('Atención', 'Configure perceptrón y cargue datos')
            return
        n = self.X.shape[1]
        win = tk.Toplevel(self)
        win.title('Prueba manual')
        entradas = []
        for i in range(n):
            ttk.Label(win, text=f'X{i}').grid(row=i, column=0)
            v = tk.DoubleVar(value=0.0)
            e = ttk.Entry(win, textvariable=v)
            e.grid(row=i, column=1)
            entradas.append(v)

        def hacer_prueba():
            valores = np.array([v.get() for v in entradas], dtype=float)
            valores_n = (valores - self.min_x) / (self.max_x - self.min_x)
            valores_n = np.nan_to_num(valores_n)
            pred = self.perceptron.predecir(valores_n)
            messagebox.showinfo('Resultado', f'Predicción: {pred.flatten().tolist()}')

        ttk.Button(win, text='Probar', command=hacer_prueba).grid(row=n, column=0, columnspan=2)

# ------------------------------ Main ------------------------------
if __name__ == '__main__':
    app = AplicacionPerceptron()
    app.mainloop()