from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from PIL import Image
import open_clip

# Initialize Flask app
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageSearcher:
    def __init__(self, embedding_path="image_embeddings.pickle", image_folder="coco_images_resized"):
        self._validate_paths(embedding_path, image_folder)

        self.image_folder = image_folder
        self.df = pd.read_pickle(embedding_path)
        self.filenames = self.df["file_name"].values
        self.embeddings = np.vstack(self.df["embedding"].values)

        self.model, self.preprocess, self.tokenizer = self._load_clip_model()
        self.pca = PCA(n_components=50)
        self.embeddings_pca = self.pca.fit_transform(self.embeddings)

    def _validate_paths(self, embedding_path, image_folder):
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file missing: {embedding_path}")
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder missing: {image_folder}")

    def _load_clip_model(self):
        model_name, pretrained = "ViT-B-32", "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        return model, preprocess, tokenizer

    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(device)
        with torch.no_grad():
            return self.model.encode_text(tokens).cpu().numpy().flatten()

    def encode_image(self, image_file):
        image = Image.open(image_file).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model.encode_image(tensor).cpu().numpy().flatten()

    def combine_embeddings(self, text_emb, image_emb, weight=0.5):
        combined = weight * text_emb + (1 - weight) * image_emb
        return combined / np.linalg.norm(combined) if np.linalg.norm(combined) > 0 else combined

    def search(self, query_emb, top_k=5, use_pca=False):
        query_emb = query_emb / np.linalg.norm(query_emb)
        embeddings = self.embeddings_pca if use_pca else self.embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if use_pca:
            query_emb = self.pca.transform(query_emb.reshape(1, -1))[0]
            query_emb = query_emb / np.linalg.norm(query_emb)

        similarities = embeddings @ query_emb
        indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.filenames[i], similarities[i]) for i in indices]

searcher = ImageSearcher()

@app.route("/", methods=["GET", "POST"])
def index():
    results, error = [], None

    if request.method == "POST":
        query_type = request.form.get("query_type", "text")
        text_query = request.form.get("text_query", "").strip()
        use_pca = request.form.get("use_pca") == "on"
        image_file = request.files.get("image_query")
        weight = float(request.form.get("hybrid_weight", 0.5))

        try:
            if query_type == "text" and text_query:
                query_emb = searcher.encode_text(text_query)
            elif query_type == "image":
                if image_file and image_file.filename != "":
                    query_emb = searcher.encode_image(image_file)
                else:
                    raise ValueError("No image file uploaded. Please provide a valid image file.")
            elif query_type == "hybrid":
                if text_query and image_file and image_file.filename != "":
                    text_emb = searcher.encode_text(text_query)
                    image_emb = searcher.encode_image(image_file)
                    query_emb = searcher.combine_embeddings(text_emb, image_emb, weight=weight)
                else:
                    raise ValueError("Provide both text and a valid image file for hybrid queries.")
            else:
                raise ValueError("Provide valid input for the chosen query type.")

            results = searcher.search(query_emb, use_pca=use_pca)
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", results=results, error_message=error)

@app.route("/coco_images_resized/<path:filename>")
def serve_image(filename):
    image_path = os.path.join(searcher.image_folder, filename)
    if not os.path.exists(image_path):
        return f"Image {filename} not found.", 404
    return send_from_directory(searcher.image_folder, filename)

if __name__ == "__main__":
    app.run(debug=True)
