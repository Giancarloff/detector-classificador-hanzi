import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scatter import Scatter
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from fastai.vision.all import *

from PIL import Image as PILImage
import io

MODEL_PATH = 'Models/EfficientNET/EfficientnetB0-hanzi_0(1).pkl'
learn = load_learner(MODEL_PATH)

def predict_image(img_path, _learn=learn):
    pred_class, pred_idx, probs = _learn.predict(img_path)
    print(f"Predicted class: {pred_class}")
    print(f"Probability: {probs[pred_idx]:.4f}")
    return pred_class, probs[pred_idx].item()

kivy.require('2.0.0')

class CameraApp(App):

    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.state = "CAMERA"
        self.camera = None
        self.captured_image_widget = None
        self.crop_scatter = None
        self.control_buttons_layout = None
        self.result_label = None
        self.setup_camera_view()
        return self.layout

    def setup_camera_view(self, *args):
        self.layout.clear_widgets()
        self.state = "CAMERA"
        try:
            self.camera = Camera(play=True, resolution=(640, 480))
        except Exception as e:
            error_label = Label(text=f'Erro ao acessar a câmera.\n{e}')
            self.layout.add_widget(error_label)
            return

        self.control_buttons_layout = BoxLayout(size_hint_y=None, height=50)
        capture_button = Button(text='Capturar Foto')
        capture_button.bind(on_press=self.capture_photo)
        
        self.layout.add_widget(self.camera)
        self.control_buttons_layout.add_widget(capture_button)
        self.layout.add_widget(self.control_buttons_layout)

    def capture_photo(self, *args):
        if self.camera and self.camera.texture:
            self.camera.play = False
            self.setup_crop_view(self.camera.texture)

    def setup_crop_view(self, texture):
        self.layout.clear_widgets()
        self.state = "CROP"
        image_layout = BoxLayout()
        self.captured_image_widget = Image(texture=texture, size_hint=(1, 1))
        self.crop_scatter = Scatter(do_rotation=False, size_hint=(None, None), size=(200, 200), pos_hint={'center_x': 0.5, 'center_y': 0.5})

        crop_box = Label(text='Selecione', outline_color=(0,0,0), outline_width=2)
        crop_box_bg = crop_box.canvas.before
        from kivy.graphics import Color, Rectangle
        crop_box_bg.add(Color(1, 1, 0, 0.3))
        crop_box_bg.add(Rectangle(size=self.crop_scatter.size, pos=(0,0)))

        def update_rect(instance, value):
            instance.canvas.before.clear()
            with instance.canvas.before:
                Color(1, 1, 0, 0.3)
                Rectangle(size=instance.size, pos=(0,0))
                
        self.crop_scatter.bind(size=update_rect)
        self.crop_scatter.add_widget(crop_box)
        
        image_layout.add_widget(self.captured_image_widget)
        image_layout.add_widget(self.crop_scatter)
        
        self.control_buttons_layout = BoxLayout(size_hint_y=None, height=50)
        process_button = Button(text='Processar Seleção')
        process_button.bind(on_press=self.process_selection)
        
        reset_button = Button(text='Reiniciar')
        reset_button.bind(on_press=self.setup_camera_view)

        self.control_buttons_layout.add_widget(process_button)
        self.control_buttons_layout.add_widget(reset_button)
        
        self.layout.add_widget(image_layout)
        self.layout.add_widget(self.control_buttons_layout)
        
        def recenter_scatter(*args):
            self.crop_scatter.center = self.captured_image_widget.center
        Clock.schedule_once(recenter_scatter)

    def process_selection(self, *args):
        if self.state != "CROP":
            return

        img_texture = self.captured_image_widget.texture
        img_pil = PILImage.frombytes('RGBA', img_texture.size, img_texture.pixels).transpose(PILImage.FLIP_TOP_BOTTOM)

        scatter_pos = self.crop_scatter.to_window(*self.crop_scatter.pos)
        scatter_size = self.crop_scatter.size

        scale_x = img_texture.width / self.captured_image_widget.width
        scale_y = img_texture.height / self.captured_image_widget.height

        crop_x = (self.crop_scatter.x - self.captured_image_widget.x) * scale_x
        crop_y = (self.captured_image_widget.y + self.captured_image_widget.height - self.crop_scatter.y - self.crop_scatter.height) * scale_y

        crop_width = self.crop_scatter.width * scale_x
        crop_height = self.crop_scatter.height * scale_y
        
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)
        
        box = (int(crop_x), int(crop_y), int(crop_x + crop_width), int(crop_y + crop_height))
        cropped_img_pil = img_pil.crop(box)

        predicted_char, prob = predict_image(cropped_img_pil)  
        self.show_result(cropped_img_pil, f"{predicted_char} ({prob:.2%})")

    def show_result(self, cropped_image, prediction):
        self.layout.clear_widgets()
        self.state = "RESULT"
        
        data = io.BytesIO()
        cropped_image.save(data, format='png')
        data.seek(0)
        img_texture = kivy.core.image.Image(data, ext='png').texture

        result_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        result_layout.add_widget(Label(text="Área Selecionada:", size_hint_y=None, height=40))
        result_layout.add_widget(Image(texture=img_texture))
        
        self.result_label = Label(text=f'Resultado da Rede: [b]{prediction}[/b]', font_size='24sp', markup=True, size_hint_y=None, height=60)
        result_layout.add_widget(self.result_label)
        
        reset_button = Button(text='Tirar Outra Foto', size_hint_y=None, height=50)
        reset_button.bind(on_press=self.setup_camera_view)

        self.layout.add_widget(result_layout)
        self.layout.add_widget(reset_button)

if __name__ == '__main__':
    CameraApp().run()