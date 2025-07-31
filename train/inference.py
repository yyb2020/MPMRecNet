# train/inference.py

import argparse
import torch
from models.model import MultiModalClassifier

def load_patch_tensor(file_path, device):
    patch_data = torch.load(file_path, map_location=device)

    a_list = []
    b_list = []

    for patch in patch_data.values():
        a_list.append(patch["a"].unsqueeze(0).to(device))
        b_list.append(patch["b"].unsqueeze(0).to(device))

    return a_list, b_list

def predict(pt_path, model_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = MultiModalClassifier(freeze_a=False, freeze_b=False, patch_batch_size=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    a_list, b_list = load_patch_tensor(pt_path, device)

    with torch.no_grad():
        logits = model(a_list, b_list)
        probs = torch.softmax(logits, dim=1)[0]
        prediction = probs.argmax().item()
        confidence = probs[prediction].item()

    print(f"✅ Prediction: {prediction} ({'Recurrence' if prediction == 1 else 'Non-recurrence'})")
    print(f"📊 Confidence: {confidence:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained MPMRecNet model")
    parser.add_argument("--file_path", required=True, help="Path to input .pt file (preprocessed sample)")
    parser.add_argument("--model_path", default="model/final_model.pt", help="Path to trained model file")
    parser.add_argument("--device", default="cuda", help="Device to run inference (cuda or cpu)")

    args = parser.parse_args()
    predict(args.pt_path, args.model_path, args.device)
