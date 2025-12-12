import requests

URL = "http://127.0.0.1:7860/generate"
TEST_IMAGE = r"C:\caricatura_test\test_in.jpg"
OUT = "test_endpoint_out.png"

files = {
    'file': open(TEST_IMAGE, 'rb')
}
data = {
    'eye_scale': '1.6',
    'nose_scale': '1.1',
    'mouth_scale': '0.95',
    'stylize_strength': '0.6'
}

print('Enviando imagem de teste para', URL)
resp = requests.post(URL, files=files, data=data)
if resp.status_code == 200:
    with open(OUT, 'wb') as f:
        f.write(resp.content)
    print('Caricatura recebida e salva em', OUT)
else:
    print('Falha:', resp.status_code, resp.text)
