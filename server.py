import io
import os
import time
import base64
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image

from app import generate_caricature

app = FastAPI(title="Caricatura API", description="SaaS-ready caricature generator")

# CORS: permitir chamadas do frontend (ajuste em produção para restringir origens)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático na raiz para facilitar hospedagem única
from fastapi.staticfiles import StaticFiles

if os.path.isdir("frontend"):
    # montar em /static para não interceptar endpoints como /generate
    app.mount("/static", StaticFiles(directory="frontend"), name="frontend")
    # servir index na raiz
    from fastapi.responses import FileResponse

    @app.get("/")
    async def root():
        return FileResponse(os.path.join("frontend", "index.html"))


def read_image_from_upload(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao ler imagem: {e}")


def call_replicate(img: Image.Image, model_version: str, token: str, inputs: dict) -> Image.Image:
    """Enviar imagem ao Replicate API (prediction) e retornar imagem resultante.

    Requer as variáveis de ambiente `REPLICATE_API_TOKEN` e `REPLICATE_MODEL_VERSION` ou passar explicitamente.
    Observação: o formato de `inputs` depende do modelo usado em Replicate — o exemplo supõe um campo `image` que aceita data URL.
    """
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64}"

    payload = {
        "version": model_version,
        "input": {**inputs, "image": data_url},
    }

    resp = requests.post("https://api.replicate.com/v1/predictions", headers=headers, json=payload)
    resp.raise_for_status()
    pr = resp.json()

    # Poll until finished
    pred_url = pr.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{pr['id']}"
    for _ in range(120):
        r2 = requests.get(pred_url, headers=headers)
        r2.raise_for_status()
        status = r2.json()
        if status.get("status") == "succeeded":
            output = status.get("output")
            if isinstance(output, list):
                out_url = output[0]
            else:
                out_url = output
            # fetch image
            img_data = requests.get(out_url).content
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        if status.get("status") == "failed":
            raise HTTPException(status_code=500, detail="Replicate job failed")
        time.sleep(1)

    raise HTTPException(status_code=504, detail="Replicate timed out")


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    eye_scale: float = 1.5,
    nose_scale: float = 1.15,
    mouth_scale: float = 0.95,
    stylize_strength: float = 0.6,
    provider: Optional[str] = None,
):
    img = read_image_from_upload(file)

    # Se configurado para usar Replicate (via parâmetro ou variável de ambiente), envie para lá
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")
    replicate_model = os.environ.get("REPLICATE_MODEL_VERSION")

    use_replicate = False
    if provider == "replicate":
        use_replicate = True
    if replicate_token and replicate_model:
        use_replicate = True

    if use_replicate and replicate_token and replicate_model:
        # Mapear parâmetros simples para inputs do modelo — ajuste conforme o modelo escolhido
        inputs = {
            "eye_scale": eye_scale,
            "nose_scale": nose_scale,
            "mouth_scale": mouth_scale,
            "stylize_strength": stylize_strength,
        }
        try:
            out_img = call_replicate(img, replicate_model, replicate_token, inputs)
        except HTTPException:
            raise
        except Exception as e:
            # fallback local em caso de erro no provedor remoto
            out_img = generate_caricature(
                img,
                eye_scale=eye_scale,
                nose_scale=nose_scale,
                mouth_scale=mouth_scale,
                stylize_strength=stylize_strength,
            )
    else:
        # fallback local heurístico
        out_img = generate_caricature(
            img,
            eye_scale=eye_scale,
            nose_scale=nose_scale,
            mouth_scale=mouth_scale,
            stylize_strength=stylize_strength,
        )

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok"}
