from app import generate_caricature
from PIL import Image
import urllib.request
import os

def download_sample(path="test_in.jpg"):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print("Falha ao baixar imagem de teste:", e)
        return None


def main():
    inp = download_sample()
    if not inp:
        print("Nenhuma imagem de teste disponível. Coloque uma imagem chamada 'test_in.jpg' na pasta do projeto e rode novamente.")
        return

    img = Image.open(inp).convert('RGB')
    print("Gerando caricatura... (pode demorar)")
    out = generate_caricature(img)
    out_path = "test_out.jpg"
    out.save(out_path)
    print(f"Caricatura salva em: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
