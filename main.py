from src.preprocessing.pipeline import preprocess_invoice_pipeline
from src.classification.inference import classify_document
import os

def run():
    # 1. Paths
    raw_image = "data/raw/invoice_original.png"
    preprocessed_image = "data/processed/invoice_preprocessed.png"
    model_path = "models/document_classifier.pt"  

    os.makedirs(os.path.dirname(preprocessed_image), exist_ok=True)

    # 1. Preprocessing
    processed = preprocess_invoice_pipeline(
        image_path=raw_image,
        output_path=preprocessed_image
    )

    # 2. Classification
    result = classify_document(
        image_path=preprocessed_image,
        model_path=model_path
    )

    # 3. Result
    print("Prediction result:", result)


if __name__ == "__main__":
    run()
