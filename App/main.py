import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import segmentation_functions as s
import denoising as d

class ImageProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing App")

        self.image_data = None
        self.segmentation_data = None

        self.create_menu()
        self.create_image_display()
        self.create_algorithm_form()
        self.create_navigation_bar()

    def create_menu(self):
        menu_bar = tk.Menu(self)

        # Menú Archivo
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Abrir", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Reset", command=self.reset_image)
        file_menu.add_command(label="Salir", command=self.quit)
        menu_bar.add_cascade(label="Archivo", menu=file_menu)

        # Menú segnebtation
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="Umbralización", command=self.show_umbralization_form)
        algorithm_menu.add_command(label="Isodata", command=self.show_isodata_form)
        algorithm_menu.add_command(label="K-Means", command=self.show_kmeans_form)
        algorithm_menu.add_command(label="GMM", command=self.show_gmm_form)
        menu_bar.add_cascade(label="Segmentacion", menu=algorithm_menu)

        # Menú denoising
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="Mean Filter", command=self.show_mean_form)
        algorithm_menu.add_command(label="Median Filter", command=self.show_median_form)
        menu_bar.add_cascade(label="Denoising", menu=algorithm_menu)

        #
        # Menú Algoritmo
        algorithm_menu = tk.Menu(menu_bar, tearoff=0)
        algorithm_menu.add_command(label="Edge detection", command=self.show_edge_form)
        menu_bar.add_cascade(label="Edge detection", menu=algorithm_menu)

        self.config(menu=menu_bar)

    def create_image_display(self):
        self.right_frame = tk.Frame(self)
        self.right_frame.pack(pady=10)

        self.figure = plt.Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def create_algorithm_form(self):
        self.algorithm_frame = tk.Frame(self)
        self.algorithm_frame.pack(pady=10)

        self.algorithm_label = tk.Label(self.algorithm_frame, text="Parámetros de Algoritmo:")
        self.algorithm_label.grid(row=0, column=0, columnspan=2, pady=5)

        self.algorithm_entry_label = tk.Label(self.algorithm_frame, text="Tau:")
        self.algorithm_entry_label.grid(row=1, column=0, padx=5)

        self.algorithm_entry = tk.Entry(self.algorithm_frame)
        self.algorithm_entry.grid(row=1, column=1, padx=5)

        self.run_algorithm_button = tk.Button(self.algorithm_frame, text="Ejecutar", command=self.run_algorithm)
        self.run_algorithm_button.grid(row=2, column=0, columnspan=2, pady=5)

    def create_navigation_bar(self):
        self.navigation_frame = tk.Frame(self)
        self.navigation_frame.pack(pady=10)

        self.navigation_label = tk.Label(self.navigation_frame, text="Seleccionar eje:")
        self.navigation_label.grid(row=0, column=0, padx=5)

        self.navigation_var = tk.StringVar(self)
        self.navigation_var.set("Z")
        self.navigation_menu = tk.OptionMenu(self.navigation_frame, self.navigation_var, "X", "Y", "Z")
        self.navigation_menu.grid(row=0, column=1, padx=5)

        
        self.navigation_scale = tk.Scale(self.navigation_frame, from_=0, to=100, orient="horizontal", command=self.update_image_display)
        self.navigation_scale.grid(row=0, column=2, padx=5)


    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii.gz"), ("All files", "*.*")])
        if file_path:
            image = nib.load(file_path)
            self.image_data = image.get_fdata()
            self.navigation_scale.config(to=self.image_data.shape[2]-1)
            self.update_image_display()

    def update_image_display(self, event=None):
        if self.image_data is not None:
            current_slice = int(self.navigation_scale.get())
            axis = self.navigation_var.get()

            if axis == "X":
                image_slice = self.image_data[current_slice, :, :]
            elif axis == "Y":
                image_slice = self.image_data[:, current_slice, :]
            else:  # axis == "Z"
                image_slice = self.image_data[:, :, current_slice]

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.imshow(image_slice, cmap="gray")
            self.canvas.draw()

    def show_umbralization_form(self):
        self.algorithm_label.config(text="Parámetros de Umbralización:")
        self.algorithm_entry_label.config(text="Tau:")

    def show_isodata_form(self):
        self.algorithm_label.config(text="Parámetros de Isodata:")
        self.algorithm_entry_label.config(text="Tau:")
    
    def show_kmeans_form(self):
        self.algorithm_label.config(text="Parámetros de K-Means:")
        self.algorithm_entry_label.config(text="K:")

    def show_gmm_form(self):
        self.algorithm_label.config(text="Parámetros de GMM:")
        self.algorithm_entry_label.config(text="n clusters:")
    
    def show_mean_form(self):
        self.algorithm_label.config(text="Parámetros de Mean Filter:")
        self.algorithm_entry_label.config(text="size:")

    def show_median_form(self):
        self.algorithm_label.config(text="Parámetros de Median Filter:")
        self.algorithm_entry_label.config(text="size:")
    
    def show_edge_form(self):
        self.algorithm_label.config(text="Parámetros de Edge detction:")
        self.algorithm_entry_label.config(text="No requiere parametro")

    def run_algorithm(self):
        if self.algorithm_label.cget("text") == "Parámetros de Umbralización:":
            tau = float(self.algorithm_entry.get())
            if self.image_data is not None:
                self.image_data = s.umbralizacion(self.image_data, tau)
        elif self.algorithm_label.cget("text") == "Parámetros de Isodata:":
            tau = float(self.algorithm_entry.get())
            if self.image_data is not None:
                self.image_data = s.isodata(self.image_data, tau)
        elif self.algorithm_label.cget("text") == "Parámetros de K-Means:":
            k = int(self.algorithm_entry.get())
            if self.image_data is not None:
                self.image_data = s.kmeans(self.image_data, k).astype(np.uint8)
        elif self.algorithm_label.cget("text") == "Parámetros de GMM:":
            k = int(self.algorithm_entry.get())
            if self.image_data is not None:
                self.image_data = s.gmm(self.image_data, k).astype(np.uint8)
        elif self.algorithm_label.cget("text") == "Parámetros de Mean Filter:":
            size = int(self.algorithm_entry.get())
            if self.image_data is not None:
                self.image_data = d.meanFilter(self.image_data, size)
        elif self.algorithm_label.cget("text") == "Parámetros de Median Filter:":
            size = int(self.algorithm_entry.get())
            if self.image_data is not None:
                self.image_data = d.medianFilter(self.image_data, size)
        elif self.image_data is not None:
                self.image_data = d.edgeDetection(self.image_data)
        
        self.update_image_display()

    def reset_image(self):
            self.image_data = None
            self.figure.clear()
            self.canvas.draw() 
            self.open_image()

app = ImageProcessingApp()
app.mainloop()
