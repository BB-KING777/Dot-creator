import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL, OptionMenu, StringVar, IntVar
import cv2
from PIL import Image, ImageTk
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import json
import os

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dot Creator")

        # Create a frame for buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.process_button = tk.Button(self.button_frame, text="Process Image", command=self.process_image)
        self.process_button.pack(side=tk.LEFT)
        
        self.bg_remove_button = tk.Button(self.button_frame, text="Remove Background", command=self.remove_background)
        self.bg_remove_button.pack(side=tk.LEFT)

        self.dot_art_button = tk.Button(self.button_frame, text="Create Dot Art", command=self.create_dot_art)
        self.dot_art_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.button_frame, text="Reset Image", command=self.reset_image)
        self.reset_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.button_frame, text="Save Image", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)

        # Create a frame for scales and option menu
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.edge_intensity_scale = Scale(self.control_frame, from_=1, to=100, orient=HORIZONTAL, label="Edge Detection Intensity")
        self.edge_intensity_scale.set(50)
        self.edge_intensity_scale.pack(side=tk.LEFT)

        self.edge_thickness_scale = Scale(self.control_frame, from_=1, to=200, orient=HORIZONTAL, label="Edge Thickness (0.1 units)")
        self.edge_thickness_scale.set(10)
        self.edge_thickness_scale.pack(side=tk.LEFT)

        self.palette_name = StringVar(root)
        self.palette_name.set("15bit")
        self.palette_menu = OptionMenu(self.control_frame, self.palette_name, "15bit", "8bit", "4bit", "4-color Grayscale")
        self.palette_menu.pack(side=tk.LEFT)

        self.dot_size_var = IntVar(root)
        self.dot_size_var.set(8)
        self.dot_size_scale = Scale(self.control_frame, from_=1, to=32, orient=HORIZONTAL, label="Dot Size", variable=self.dot_size_var)
        self.dot_size_scale.pack(side=tk.LEFT)

        self.color_count_var = IntVar(root)
        self.color_count_var.set(8)
        self.color_count_scale = Scale(self.control_frame, from_=2, to=32, orient=HORIZONTAL, label="Color Count", variable=self.color_count_var)
        self.color_count_scale.pack(side=tk.LEFT)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.original_image = None
        self.processed_image = None
        self.edge_image = None
        self.displayed_image = None
        self.palette = None
        self.palette_file = "palette.json"

        if os.path.exists(self.palette_file):
            with open(self.palette_file, 'r') as f:
                self.palette = np.array(json.load(f), dtype=np.uint8)

    def generate_palette(self, bit_depth):
        if bit_depth == "15bit":
            return self.generate_15bit_palette()
        elif bit_depth == "8bit":
            return self.generate_8bit_palette()
        elif bit_depth == "4bit":
            return self.generate_4bit_palette()
        elif bit_depth == "4-color Grayscale":
            return self.generate_4color_grayscale_palette()

    def generate_15bit_palette(self):
        """Generate a 15-bit color palette (32,768 colors)."""
        palette = []
        for r in range(32):
            for g in range(32):
                for b in range(32):
                    r_scaled = (r << 3) | (r >> 2)
                    g_scaled = (g << 3) | (g >> 2)
                    b_scaled = (b << 3) | (b >> 2)
                    palette.append((r_scaled, g_scaled, b_scaled))
        return np.array(palette, dtype=np.uint8)

    def generate_8bit_palette(self):
        """Generate an 8-bit color palette (256 colors)."""
        palette = []
        for r in range(8):
            for g in range(8):
                for b in range(4):
                    r_scaled = (r * 255 // 7)
                    g_scaled = (g * 255 // 7)
                    b_scaled = (b * 255 // 3)
                    palette.append((r_scaled, g_scaled, b_scaled))
        return np.array(palette, dtype=np.uint8)

    def generate_4bit_palette(self):
        """Generate a 4-bit color palette (16 colors)."""
        palette = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
            (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
            (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255)
        ]
        return np.array(palette, dtype=np.uint8)

    def generate_4color_grayscale_palette(self):
        """Generate a 4-color grayscale palette."""
        palette = [
            (0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)
        ]
        return np.array(palette, dtype=np.uint8)

    def save_palette_to_json(self, palette):
        with open(self.palette_file, 'w') as f:
            json.dump(palette.tolist(), f)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = self.resize_image(self.original_image, 800)
            self.display_image(self.original_image)
            self.palette = self.generate_palette(self.palette_name.get())
            self.save_palette_to_json(self.palette)

    def process_image(self):
        if self.original_image is not None:
            intensity = self.edge_intensity_scale.get()
            thickness = self.edge_thickness_scale.get() / 10.0  # Convert scale value to 0.1 units
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, intensity, intensity * 2)
            
            # Create a kernel for dilation based on the thickness
            kernel_size = max(1, int(thickness))  # Ensure the kernel size is at least 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Create an edge mask where edges are white and background is black
            edge_mask = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR)
            edge_mask[np.where((edge_mask == [255, 255, 255]).all(axis=2))] = [0, 0, 0]  # Convert white edges to black

            # Invert the edge mask to overlay on the original image
            inverted_mask = cv2.bitwise_not(edges_dilated)
            inverted_mask = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)

            # Combine the original image and the inverted edge mask
            combined_image = cv2.bitwise_and(self.original_image, inverted_mask)

            self.processed_image = combined_image
            self.edge_image = edges_dilated
            self.display_image(self.processed_image)

    def remove_background(self):
        if self.original_image is not None:
            mask = np.zeros(self.original_image.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (50, 50, self.original_image.shape[1]-50, self.original_image.shape[0]-50)
            cv2.grabCut(self.original_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            self.processed_image = self.original_image * mask2[:, :, np.newaxis]
            self.display_image(self.processed_image)

    def create_dot_art(self):
        if self.edge_image is not None and self.processed_image is not None:
            # Convert the processed image to RGBA (to support transparency)
            rgba_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGBA)
            rgba_image[np.all(rgba_image[:, :, :3] == [0, 0, 0], axis=2)] = [0, 0, 0, 0]  # Set black background to transparent
    
            # Reshape the image to a 2D array of pixels
            pixels = rgba_image.reshape(-1, 4)
    
            # Perform k-means clustering to find dominant colors
            color_count = self.color_count_var.get()
            kmeans = KMeans(n_clusters=color_count, n_init=10, random_state=0)
            kmeans.fit(pixels[:, :3])  # Use only RGB for clustering
            dominant_colors = kmeans.cluster_centers_.astype(int)
    
            # Map the dominant colors to the nearest palette colors
            new_palette = []
            for color in dominant_colors:
                index = pairwise_distances_argmin([color], self.palette)
                new_palette.append(self.palette[index][0])
            new_palette = np.array(new_palette)
    
            # Create a new image with the mapped colors
            new_pixels = np.array([new_palette[label] for label in kmeans.labels_])
            new_pixels = np.concatenate((new_pixels, pixels[:, 3].reshape(-1, 1)), axis=1)  # Add alpha channel back
            
            dot_art = new_pixels.reshape(rgba_image.shape).astype(np.uint8)
    
            # Reduce the size of the image to create pixelation effect
            dot_size = self.dot_size_var.get()
            small = cv2.resize(dot_art, (dot_art.shape[1] // dot_size, dot_art.shape[0] // dot_size), interpolation=cv2.INTER_NEAREST)
            dot_art = cv2.resize(small, (dot_art.shape[1], dot_art.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Convert edge image to RGBA
            edge_overlay = cv2.cvtColor(self.edge_image, cv2.COLOR_GRAY2RGBA)
            
            # Set edge pixels to black
            edge_overlay[np.where((edge_overlay == [255, 255, 255, 255]).all(axis=2))] = [0, 0, 0, 255]
            
            # Resize edge overlay to match dot art size
            edge_overlay = cv2.resize(edge_overlay, (dot_art.shape[1], dot_art.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Combine dot art and edge overlay
            result = cv2.bitwise_and(dot_art, cv2.bitwise_not(edge_overlay))
            result = cv2.add(result, edge_overlay)
            
            self.display_image(result, True)

    def reset_image(self):
        if self.original_image is not None:
            self.display_image(self.original_image)

    def save_image(self):
        if self.displayed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                self.displayed_image.save(file_path)

    def display_image(self, image, is_rgba=False):
        if is_rgba:
            image_pil = Image.fromarray(image)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        self.displayed_image = image_pil
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def resize_image(self, image, width):
        height = int((width / image.shape[1]) * image.shape[0])
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()
