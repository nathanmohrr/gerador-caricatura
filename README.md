# Gerador de Caricatura (Protótipo)

Protótipo local de geração de caricaturas no estilo do anexo usando uma abordagem heurística.

Características:
- Detecta face com `MediaPipe`.
- Exagera regiões (olhos, nariz, boca) usando recorte e `seamlessClone` do OpenCV.
- Aplica filtro tipo "cartoon" (bilateral + detecção de bordas).
- Interface web local com `Gradio` para upload e ajustes em tempo real.

Instalação (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Executar:

```powershell
python app.py
```

Observações e próximos passos:
- Esta é uma solução heurística (não usa modelos de difusão). Para resultados mais artísticos e próximos do estilo do anexo, recomendo integrar um modelo de difusão condicional (ex.: Fine-tuned Stable Diffusion, ControlNet para face-conditioning). Isso exigirá GPU e mais trabalho de fine-tuning.
- Se quiser, eu implemento a versão com Diffusers/Hugging Face (posso criar scripts de download e exemplos de prompts estilizados).
