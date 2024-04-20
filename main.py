import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np 
import spacy
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
"""""
Example
dog, cat, mouse, happy, sad, love, sun, moon, sand, river

"""""
def generate_plot():
    words_to_visualize = entry.get().split(", ")
    if not words_to_visualize:
        messagebox.showerror("Error", "Please insert a list of words in english. Be careful with spaces. For example:dog, cat, mouse, happy, sad, love, sun, moon, sand, river .")
        return
    plot_word_vectors(words_to_visualize)

def plot_word_vectors(words, model_name="en_core_web_md", fig_size=(12, 10), point_size=50, background_color="white"):
    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading spaCy model: {e}")
        return

    word_vectors = []
    missing_words = []
    for w in words:
        if not nlp.vocab.has_vector(w):
            missing_words.append(w)
            continue
        word_vector = nlp.vocab.vectors[nlp.vocab.strings[w]]
        word_vectors.append(word_vector)

    if missing_words:
        messagebox.showerror("Error", f"The following words are not include in the model: {', '.join(missing_words)}")
        return

    word_vectors = np.vstack(word_vectors)

    pca = PCA(n_components=2)
    try:
        word_vectors_transformed = pca.fit_transform(word_vectors)
    except Exception as e:
        messagebox.showerror("Error", f"Error executing PCA: {e}")
        return

    plt.figure(figsize=fig_size)
    plt.scatter(word_vectors_transformed[:, 0], word_vectors_transformed[:, 1], s=point_size)
    plt.gca().set_facecolor(background_color)
    for word, coord in zip(words, word_vectors_transformed):
        x, y = coord
        plt.text(x, y, word, size=10)
    plt.xlabel('Principal Component  1')
    plt.ylabel('Principal Component  2')
    plt.title('Visualization of vector of words with PCA')
    plt.savefig("Visualization of vector of words with PCA.jpg")
    plt.grid(True)
    try:
        save_file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if save_file_path:
            plt.savefig(save_file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error saving the file: {e}")
    plt.show()

# Crear la ventana principal
root = tk.Tk()
root.title("Word Vector Visualization")

# Crear etiqueta y campo de entrada
label = ttk.Label(root, text="Insert a list of words in english. For example: dog, cat...")
label.grid(row=0, column=0, padx=10, pady=10)
entry = ttk.Entry(root)
entry.grid(row=0, column=1, padx=10, pady=10)

# Crear botón de generar visualización
generate_button = ttk.Button(root, text="Genarate visualization", command=generate_plot)
generate_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Crear botón de salir
exit_button = ttk.Button(root, text="Exit", command=root.destroy)
exit_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
