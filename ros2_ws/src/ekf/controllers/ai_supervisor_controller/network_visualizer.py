import tkinter as tk
import torch

infos = [['ax', 'ay', 'az', 'vroll', 'vpitch', 'vyaw'], [], [], ['x', 'y', 'z', 'roll', 'pitch', 'yaw']]  # Список для хранения текстовой информации для каждого слоя

class NetworkVisualizer:
    def __init__(self, root, layer_sizes, layer_map):
        self.root = root
        self.canvas = tk.Canvas(root, width=800, height=600, bg='white')
        self.canvas.pack()
        self.layer_sizes = layer_sizes
        # Карта, связывающая имя слоя с его индексом в визуализации
        # Пример: {'input_current': 0, 'current_mlp_relu': 1, ...}
        self.layer_map = layer_map
        self.neuron_ovals = []
        self.neuron_texts = []
        self.neuron_info_texts = []

        self._draw_network()

    def _get_color(self, value):
        """Преобразует нормализованное значение (0..1) в цвет."""
        value = max(0, min(1, value))
        red = int(255 * value)
        green = int(255 * value)
        blue = int(255 * (1 - value))
        return f'#{red:02x}{green:02x}{blue:02x}'

    def _draw_network(self):
        x_spacing = 800 / (len(self.layer_sizes) + 1)
        
        for i, size in enumerate(self.layer_sizes):
            layer_ovals = []
            layer_texts = [] # Временный список для текстов в слое
            layer_info_texts = []
            y_spacing = 600 / (size + 1)
            for j in range(size):
                x = x_spacing * (i + 1)
                y = y_spacing * (j + 1)
                
                # Создаем овал
                oval = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill='blue', outline='white')
                
                # Создаем текст поверх овала
                text = self.canvas.create_text(x, y, text='0.0', fill='black', font=('Arial', 7))
                if len(infos[i]) > 0:
                    info_text = self.canvas.create_text(x, y - 30, text=infos[i][j], fill='gray', font=('Arial', 8))
                    layer_info_texts.append(info_text)

                layer_ovals.append(oval)
                layer_texts.append(text) # Добавляем ID текста

            self.neuron_ovals.append(layer_ovals)
            self.neuron_texts.append(layer_texts) # Сохраняем ID текстов для всего слоя
            self.neuron_info_texts.append(layer_info_texts)


    def update_visuals(self, activations_dict):
        for name, layer_idx in self.layer_map.items():
            if name not in activations_dict:
                continue

            activations_tensor = activations_dict[name].flatten()
            
            # ... (логика нормализации для цвета остается прежней) ...
            if name == 'input_current':
                normalized_values = (activations_tensor + 5) / 10
            elif 'relu' in name:
                max_val = torch.max(activations_tensor)
                normalized_values = activations_tensor / max_val if max_val > 0 else activations_tensor
            elif name == 'output_mu':
                normalized_values = (activations_tensor + 1) / 2
            else:
                normalized_values = torch.clamp(activations_tensor, 0, 1)

            # Обновляем цвета и ТЕКСТ нейронов
            for i, norm_value in enumerate(normalized_values):
                if i < len(self.neuron_ovals[layer_idx]):
                    # Обновляем цвет
                    color = self._get_color(norm_value.item())
                    self.canvas.itemconfig(self.neuron_ovals[layer_idx][i], fill=color)
                    
                    # Обновляем текст, используя исходное значение
                    original_value = activations_tensor[i].item()
                    formatted_text = f"{original_value:.2f}" # Форматируем до 2 знаков после запятой
                    self.canvas.itemconfig(self.neuron_texts[layer_idx][i], text=formatted_text)
