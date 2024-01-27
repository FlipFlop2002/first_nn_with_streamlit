import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import AnimalClassificationModel


def predict(image, model):
    classes = ['Cheetah', 'Jaguar', 'Leopard', 'Lion', 'Tiger']
    # Przetwarzanie obrazu
    preprocess = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    input_tensor = preprocess(image)

    # Przewidywanie
    model.eval()
    with torch.inference_mode():
        y_logits_2 = model(input_tensor.unsqueeze(dim=0))

    y_pred_label_2 = torch.argmax(y_logits_2)

    result = classes[y_pred_label_2]

    return result


def main():
    model_0 = AnimalClassificationModel(3, 10, 5)
    model_0.load_state_dict(torch.load("models/model_0.pth"))
    st.title("Wild cats classification")

    # Wybierz plik obrazu
    uploaded_file = st.file_uploader(
        "Wybierz plik obrazu", type=["jpg", "jpeg", "png"]
        )

    if uploaded_file is not None:
        # Wyświetl obraz
        image = Image.open(uploaded_file)
        st.image(image, caption="Wgrany obraz", use_column_width=True)

        # Wykonaj predykcję
        prediction = predict(image, model_0)
        st.subheader(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
