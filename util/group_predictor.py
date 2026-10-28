"""One detector/group/mask/classifier implementation for both inference policies."""
import numpy as np
import torch

from util.mask_input import build_mask_input
from util.spectrogram_frontend import FrontendConfig, render_rgb


class GroupPredictor:
    def __init__(self, classifier, detector, preprocessor, config, device="cpu", batch_size=32, return_features=False):
        if batch_size < 1:
            raise ValueError("classifier_batch_size must be >= 1")
        self.classifier = classifier.to(device).eval()
        self.detector, self.preprocessor = detector, preprocessor
        self.config, self.device = config, device
        self.batch_size, self.return_features = batch_size, return_features
        self.frontend = FrontendConfig.from_config(config)

    @torch.inference_mode()
    def detect_groups(self, spectrum, domain):
        rgb = render_rgb(spectrum, self.frontend, domain)
        boxes = self.detector.detect_rgb(rgb)
        result = {"status": "no_detection", "detected_boxes": len(boxes), "group_count": 0}
        if not len(boxes):
            return result, []
        groups = self.preprocessor.select_signal_groups(boxes, spectrogram=spectrum) or []
        result.update(group_count=len(groups), status="ok" if groups else "no_group")
        return result, groups

    @torch.inference_mode()
    def classify_groups(self, spectrum, groups):
        predictions = []
        names = {idx: name for name, idx in self.config.classes.items()}
        for start in range(0, len(groups), self.batch_size):
            chunk = groups[start:start+self.batch_size]
            tensors = [build_mask_input(spectrum, group["boxes"], self.config)[0] for group in chunk]
            batch = torch.stack(tensors).to(self.device)
            if self.return_features:
                from util.open_set import normalize_features
                logits, features = self.classifier(batch, return_features=True)
                features = normalize_features(features.cpu().numpy())
            else:
                logits = self.classifier(batch)
            if tuple(logits.shape) != (len(chunk), self.config.num_classes) or not torch.isfinite(logits).all():
                raise ValueError("Classifier must return finite [group_count, num_classes] logits")
            for index, (group, probabilities) in enumerate(zip(chunk, logits.softmax(dim=1).cpu().numpy())):
                label = int(np.argmax(probabilities))
                prediction = {"group_id": start+index, "status": "ok", "class_id": label,
                    "class_name": names[label], "confidence": float(probabilities[label]),
                    "probabilities": probabilities.astype(float).tolist(),
                    "selected_boxes": np.asarray(group["boxes"]).tolist(),
                    "selected_group_score": float(group.get("score", 0.)),
                    "group_type": group.get("group_type", "frequency")}
                if self.return_features:
                    prediction["embedding"] = features[index].tolist()
                predictions.append(prediction)
        return predictions
