# Adobe Hackathon Challenge - Challenge 1B

This project is a fully offline NLP pipeline that helps to analyze PDF documents and generate refined sections relevant to a task.

It uses a multilingual SentenceTransformer model i.e. distiluse-base-multilingual-cased-v1 and adheres to all constraints:
- Model size: **under 1GB**
- Runs completely **offline**
- Packaged inside a Docker container

---
## Requirements
- Docker Desktopinstalled and running
- OS: Windows/macOS/Linux

---

## Project Structure
ADOBE_HACK/
├── Dockerfile
├── README.md
├── main.py
├── requirements.txt
├── model/ # Pre-downloaded transformer model
├── input/ # Contains PDFs and input JSON
├── output/ # Will contain the generated output
└── .dockerignore

---
## How to Build and Run

1. Ensure your input files (`challenge1b_input.json` and PDFs) are inside the `input/` folder.
2. Ensure your pre-saved model is in the `model/` folder.
3. Build the Docker image: (First navigate to the folder containing the project)

   - Now run the following command to generate a docker image: docker build -t adobe-hack .

   - After the image is generated, execute the following command to run the image and generate output:

           docker run --rm -v "%cd%/input:/app/input" -v "%cd%/output:/app/output" adobe-hack
