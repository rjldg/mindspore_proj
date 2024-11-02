import os
import numpy as np
import mindspore as ms

from mindspore import Tensor
from mindspore.train import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import Softmax
from PIL import Image
import flet as ft
from flet import AppBar, CupertinoFilledButton, Page, Container, Text, View, FontWeight, colors, TextButton, padding, ThemeMode, border_radius, Image as FletImage, FilePicker, FilePickerResultEvent, icons

from image_classifier.resnet50_archi import resnet50
from main import predict

cfg = {
    'HEIGHT': 224,
    'WIDTH': 224,
    '_R_MEAN': 123.68,
    '_G_MEAN': 116.78,
    '_B_MEAN': 103.94,
    '_R_STD': 1,
    '_G_STD': 1,
    '_B_STD': 1,
    'num_class': 2,
    'model_path': '../../best_model.ckpt'
}

class_names = {0:'algal_leaf',1:'anthracnose',2:'bird_eye_spot',3:'brown_blight',4:'gray_light',5:'healthy',6:'red_leaf_spot',7:'white_spot'}

def main(page: Page):
    page.title = "SpotumAI"
    page.theme_mode = ThemeMode.LIGHT

    page.fonts = {
        "RobotoFlex": f"fonts/RobotoFlex-VariableFont.ttf",
        "RobotoMono": f"fonts/RobotoMono-VariableFont.ttf",
        "RobotoMonoItalic": f"fonts/RobotoMono-Italic-VariableFont.ttf"
    }

    def pick_files_result(e: FilePickerResultEvent):
        selected_files.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        selected_files.update()

    def process_image(e):
        if file_picker.result and file_picker.result.files:
            image_path = file_picker.result.files[0].path
            print(image_path)

            if os.path.exists(image_path):
                predicted_class, confidence = predict(image_path)
                conf_converted = confidence * 100
                
                result_pred.value = f"Prediction: {predicted_class}"
                result_conf.value = f"[{conf_converted:.2f} Confident]"
                result_image.src = image_path

                if predicted_class == "Normal":
                    result_pred.color = "green"
                else:
                    result_pred.color = "red"
                
                if confidence > 0.8:
                    result_conf.color = "green"
                elif confidence > 0.5:
                    result_conf.color = "orange"
                else:
                    result_conf.color = "red"
                
                result.disabled = True
                restart_button.visible = True
                result_pred.update()
                result_conf.update()
                restart_button.update()
                result.update()

            else:
                result_pred.value = f"Error: File '{image_path}' does not exist."
                result_pred.update()
                result_conf.update()

    def restart_process(e):
        result_pred.value = ""
        result_conf.value = ""
        result_image.src = "assets/result_initial.png"
        result.disabled = False
        restart_button.visible = False
        result_pred.update()
        result_conf.update()
        restart_button.update()
        result.update()
        # print("RESTARTED")

    file_picker = FilePicker(on_result=process_image)
    selected_files = Text()
    
    page.overlay.append(file_picker)

    restart_button = TextButton(content=Text("Start over", font_family="RobotoMono", size=14, weight=FontWeight.W_300, color=colors.BLACK), visible=False, on_click=restart_process)
    
    result_pred = Text(size=40, font_family="RobotoFlex", weight=FontWeight.W_700)
    result_conf = Text(size=24, font_family="RobotoMono", weight=FontWeight.W_500)
    result_image = ft.Image(src="assets/result_initial.png", width=350, height=350, border_radius=border_radius.all(10))

    result = TextButton(content=result_image, on_click=lambda _: file_picker.pick_files(allow_multiple=False))

    def route_change(e):
        page.views.clear()
        page.views.append(
            View(
                "/",
                [
                    AppBar(title_spacing=50, title=Text("Botani", font_family="RobotoFlex", size=40, weight=FontWeight.W_700, color=colors.BLACK), bgcolor="#f8f9ff", toolbar_height=120,
                           actions=[
                               TextButton(content=Container(Text("GitHub", font_family="RobotoFlex", size=18, weight=FontWeight.W_400, color=colors.BLACK), padding=padding.only(right=25, left=25)), url="https://github.com/rjldg/mindspore_proj"),
                               TextButton(content=Container(Text("About Us", font_family="RobotoFlex", size=18, weight=FontWeight.W_400, color=colors.BLACK), padding=padding.only(right=25, left=25)), on_click=lambda _: page.go("/aboutus"))
                           ]
                    ),
                    Container(
                        content=ft.Column(
                            [
                                Text("DETECT TEA LEAF SICKNESS\nUSING ARTIFICIAL INTELLIGENCE", font_family="RobotoFlex", size=40, weight=FontWeight.W_800, color=colors.BLACK),
                                Text("Your AI-powered ally in detecting\ntea leaf sickness from images\nwith impressively satisfactory accuracy!", font_family="RobotoMono", size=20, weight=FontWeight.W_300, color=colors.BLACK),
                            ]
                        ),
                        padding=padding.only(left=100, top=70)
                    ),
                    Container(
                        CupertinoFilledButton(content=Text("Get Started", font_family="RobotoFlex", size=18, weight=FontWeight.W_500, color=colors.WHITE), border_radius=border_radius.all(8), on_click=lambda _: page.go("/spotumapp")),
                        padding=padding.only(left=100, top=70)
                    )
                ],
            )
        )
        if page.route == "/spotumapp":
            page.views.append(
                View(
                    "/spotumapp",
                    [
                        AppBar(color=colors.BLACK, bgcolor="#f8f9ff"),
                        Container(
                            content=ft.Row(
                                [
                                    result,
                                    Container(
                                        content=ft.Column(
                                            [
                                                result_pred,
                                                result_conf,
                                                restart_button,
                                            ]
                                        )
                                    )
                                ],
                                alignment=ft.MainAxisAlignment.CENTER,
                                spacing=10
                            ),
                            padding=padding.only(top=100)
                        ),
                    ],
                )
            )
        if page.route == "/aboutus":
            page.views.append(
                View(
                    "/aboutus",
                    [
                        AppBar(color=colors.BLACK, bgcolor="#f8f9ff"),
                        Container(
                            Text("MEET THE TEAM", font_family="RobotoFlex", size=40, weight=FontWeight.W_700, color=colors.BLACK),
                            padding=padding.only(left=100)
                        ),
                        Container(
                            content=ft.Column(
                                [
                                    ft.Row(
                                        [
                                            ft.Image(src="assets/rance.png"),
                                            Container(content=ft.Column(
                                                [
                                                    Text("Aaron Abadiano", font_family="RobotoFlex", size=20, weight=FontWeight.W_600, color=colors.BLACK),
                                                    Text("ababadiano@mymail.mapua.edu.ph", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color=colors.BLACK),
                                                ]
                                            )
                                            ),
                                        ]
                                    ),
                                    ft.Row(
                                        [
                                            ft.Image(src="assets/rance.png"),
                                            Container(content=ft.Column(
                                                [
                                                    Text("Clara Capistrano", font_family="RobotoFlex", size=20, weight=FontWeight.W_600, color=colors.BLACK),
                                                    Text("cascapistrano@mymail.mapua.edu.ph", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color=colors.BLACK),
                                                ]
                                            )
                                            ),
                                        ]
                                    ),
                                    ft.Row(
                                        [
                                            ft.Image(src="assets/rance.png"),
                                            Container(content=ft.Column(
                                                [
                                                    Text("Rance De Guzman", font_family="RobotoFlex", size=20, weight=FontWeight.W_600, color=colors.BLACK),
                                                    Text("rjldeguzman@mymail.mapua.edu.ph", font_family="RobotoMono", size=16, weight=FontWeight.W_300, color=colors.BLACK),
                                                ]
                                            )
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            padding=padding.only(left=200),
                        ),
                    ],
                )
            )
        page.update()

    def view_pop(e):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    page.go(page.route)

ft.app(main)
