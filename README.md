# FLUX.1-Kontext Multi-Image Texture Transfer

A Gradio-based application for multi-image composition and texture transfer using FLUX.1-Kontext-dev model.

## Features

- **Multi-image input**: Upload and process multiple images simultaneously
- **Intelligent composition**: Automatically concatenates input images and creates unified compositions
- **Texture transfer**: Apply textures and characteristics from one image to elements in another
- **Adaptive sizing**: Uses dimensions from the second image when available for optimal results
- **Interactive web interface**: Built with Gradio for easy use

## Requirements

- Python 3.8+
- CUDA-compatible GPU
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

Then open your browser to the displayed local URL. Upload multiple images and describe the desired composition in the prompt field.

## Example Prompts

- "Apply the texture from the first image to the sofa in the second image"
- "Combine these elements into a single harmonious scene"
- "Transfer the material properties from image 1 to the object in image 2"

## Model

Uses the FLUX.1-Kontext-dev model from Black Forest Labs for advanced multi-image understanding and generation.