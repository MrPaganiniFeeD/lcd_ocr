import os
from typing import List, Tuple, Optional, Dict
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

import concurrent.futures
from typing import Iterable

def build_resnet18_single_logit(pretrained: bool = False) -> nn.Module:
    net = models.resnet18(pretrained=pretrained)
    in_feats = net.fc.in_features
    net.fc = nn.Linear(in_feats, 1)
    return net

def _load_state_dict_flex(model: nn.Module, ckpt_path: str, map_location=None) -> nn.Module:
    ck = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ck, dict):
        if 'model_state_dict' in ck:
            state = ck['model_state_dict']
        elif 'state_dict' in ck:
            state = ck['state_dict']
        elif 'model_state' in ck:
            state = ck['model_state']
        else:
            state = ck
    else:
        raise RuntimeError("Unexpected checkpoint format (expected dict/state_dict).")

    new_state = {}
    for k, v in state.items():
        nk = k[len('module.'):] if k.startswith('module.') else k
        new_state[nk] = v

    try:
        model.load_state_dict(new_state)
    except RuntimeError as e:
        missing = unexpected = None
        try:
            missing, unexpected = model.load_state_dict(new_state, strict=False)
        except Exception:
            pass
        raise RuntimeError(
            f"Ошибка загрузки state_dict: {e}\n"
            f"(Попытка загрузки с strict=False вернула missing={missing} unexpected={unexpected})"
        )
    return model

class ResNetPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        img_size: int = 224,
        threshold: float = 0.5,
        pretrained_backbone: bool = False,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.img_size = img_size
        self.threshold = float(threshold)

        self.model = build_resnet18_single_logit(pretrained=pretrained_backbone)
        self.model = _load_state_dict_flex(self.model, checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize(int(self.img_size / 1.2)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def predict_pil(self, img: Image.Image) -> Dict[str, object]:
        img = img.convert("RGB")
        x = self.tf(img).unsqueeze(0).to(self.device)  # (1,C,H,W)
        with torch.no_grad():
            logits = self.model(x)
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > self.threshold else 0
        return {"probability": float(prob), "pred": int(pred)}

    def predict(self, image_path: str) -> Dict[str, object]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path)
        return self.predict_pil(img)

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        ) -> List[Dict[str, object]]:
        """
        Пакетная предсказательная функция.

        Args:
            image_paths: список путей к изображениям.
            batch_size: количество изображений, прогоняемых через модель за раз.
            num_workers: число потоков для параллельного чтения/предобработки.

        Returns:
            список словарей (в том же порядке, что и image_paths). Для успешного
            предсказания: {"path": p, "probability": float, "pred": int}.
            Для ошибки при чтении/предобработке: {"path": p, "error": str}.
        """
        # вспомогательная функция для загрузки + предобработки одного изображения
        def _load_and_preprocess(path: str):
            try:
                img = Image.open(path).convert("RGB")
                tensor = self.tf(img)  # (C,H,W), CPU tensor
                return (path, tensor, None)
            except Exception as e:
                return (path, None, str(e))

        n = len(image_paths)
        results: List[Optional[Dict[str, object]]] = [None] * n

        preproc_success: List[tuple] = []  # список (original_index, path, tensor)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_load_and_preprocess, p): idx for idx, p in enumerate(image_paths)}
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                path, tensor, err = fut.result()
                if err is not None:
                    results[idx] = {"path": path, "error": err}
                else:
                    preproc_success.append((idx, path, tensor))

        if len(preproc_success) == 0:
            return results 

        preproc_success.sort(key=lambda x: x[0]) 

        def _chunks(iterable: Iterable, size: int):
            it = iter(iterable)
            while True:
                chunk = []
                try:
                    for _ in range(size):
                        chunk.append(next(it))
                except StopIteration:
                    if chunk:
                        yield chunk
                    break
                else:
                    yield chunk

        device = self.device
        self.model.eval()
        with torch.no_grad():
            for chunk in _chunks(preproc_success, batch_size):
                idxs = [c[0] for c in chunk]
                paths = [c[1] for c in chunk]
                tensors = [c[2] for c in chunk]  # CPU tensors
                batch = torch.stack(tensors, dim=0).to(device)  # (B,C,H,W) -> на device

                logits = self.model(batch)
                # нормализуем форму logits -> (B,)
                if logits.dim() == 2 and logits.size(1) == 1:
                    logits = logits.view(-1)
                elif logits.dim() > 1:
                    logits = logits.squeeze()
                probs = torch.sigmoid(logits).cpu().tolist()  # обратно на CPU

                for i, path, prob in zip(idxs, paths, probs):
                    pred = 1 if prob > self.threshold else 0
                    results[i] = {"path": path, "probability": float(prob), "pred": int(pred)}

        # 3) на всякий случай заполнить любые None (если такие остались) ошибкой
        for i, r in enumerate(results):
            if r is None:
                results[i] = {"path": image_paths[i], "error": "Unknown error during processing."}

        return results


# ========== helper loader ==========
def load_predictor(checkpoint_path: str, device: Optional[str]=None, img_size: int=224, threshold: float=0.5) -> ResNetPredictor:
    return ResNetPredictor(checkpoint_path=checkpoint_path, device=device, img_size=img_size, threshold=threshold)

# ========== optional CLI usage ==========
if __name__ == "__main__":
    import argparse, csv
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    pred = load_predictor(args.model, threshold=args.threshold)
    print(pred.predict(args.image))
