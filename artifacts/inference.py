"""P5 Fine-Grained Vehicle Recognition — inference module for the frontend.

    from inference import VehicleRecognizer
    rec = VehicleRecognizer('artifacts')          # backend='onnx' -> CPU INT8 classifier
    out = rec.predict('car.jpg')

Needs in the artifacts folder: config.json, class_meta.json, yolo_car_best.pt, classifier_best.pth
(or classifier_int8.onnx), classical_lr.joblib, and classical_cv.py next to this file.
"""
import os, json, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from classical_cv import to_bgr, adaptive_enhance, crop_box, classical_roi, extract_classical_features, processing_stages


def _log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


STATUS_COLORS = {'confident': (0, 200, 0), 'uncertain_model': (255, 165, 0), 'low_confidence': (230, 0, 0)}


class VehicleRecognizer:
    def __init__(self, art_dir='artifacts', device=None, backend='torch', onnx_file='classifier_int8.onnx', use_fusion=True):
        from ultralytics import YOLO
        self.art = art_dir
        self.cfg = json.load(open(os.path.join(art_dir, 'config.json')))
        self.classes = json.load(open(os.path.join(art_dir, 'class_meta.json')))
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.backend = backend
        self.det = YOLO(os.path.join(art_dir, 'yolo_car_best.pt'))
        S = self.cfg['img_size']
        self.tf = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize(self.cfg['mean'], self.cfg['std'])])
        self.clf = None
        if backend == 'onnx':
            import onnxruntime as ort
            self.sess = ort.InferenceSession(os.path.join(art_dir, onnx_file), providers=['CPUExecutionProvider'])
        else:
            import timm
            ck = torch.load(os.path.join(art_dir, 'classifier_best.pth'), map_location='cpu', weights_only=False)
            self.clf = timm.create_model(ck['model_name'], pretrained=False, num_classes=ck['num_classes'])
            self.clf.load_state_dict(ck['state_dict'])
            self.clf.eval().to(self.device)
        self.classical = None
        p = os.path.join(art_dir, 'classical_lr.joblib')
        if use_fusion and self.cfg.get('alpha', 0) > 0 and os.path.exists(p):
            import joblib
            self.classical = joblib.load(p)
        self.make_names = sorted({c['make'] for c in self.classes})
        self.make_idx = np.array([self.make_names.index(c['make']) for c in self.classes])
        self._cam = None

    # ---------- detection with low-confidence fallbacks ----------
    def detect(self, bgr, max_vehicles=5):
        r = self.det.predict(bgr, conf=0.05, iou=0.6, imgsz=self.cfg['yolo_imgsz'], verbose=False)[0]
        b, s = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        order = np.argsort(-s)
        keep = [i for i in order if s[i] >= self.cfg['det_conf']][:max_vehicles]
        if keep:
            return [(b[i].tolist(), float(s[i]), 'yolo') for i in keep]
        if len(b):
            i = order[0]
            return [(b[i].tolist(), float(s[i]), 'yolo_low_confidence')]
        box, score, _ = classical_roi(bgr)
        if box is not None:
            return [(list(map(float, box)), score, 'classical_roi_fallback')]
        H, W = bgr.shape[:2]
        return [([0.0, 0.0, float(W), float(H)], 0.0, 'full_image_fallback')]

    # ---------- classification ----------
    def _logits(self, crops):
        x = torch.stack([self.tf(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops])
        if self.backend == 'onnx':
            xn = x.numpy()
            z = self.sess.run(None, {'input': xn})[0]
            if self.cfg.get('tta', True):
                z = (z + self.sess.run(None, {'input': np.ascontiguousarray(xn[:, :, :, ::-1])})[0]) / 2
            return z
        x = x.to(self.device)
        with torch.no_grad():
            z = self.clf(x).float()
            if self.cfg.get('tta', True):
                z = (z + self.clf(torch.flip(x, [3])).float()) / 2
        return z.cpu().numpy()

    def classify(self, crops):
        enh = [adaptive_enhance(c, self.cfg['dark_thr'], self.cfg['bright_thr'])[0] if self.cfg.get('enhance') else c for c in crops]
        s = _log_softmax(self._logits(enh) / self.cfg['T'])
        if self.classical is not None:
            f = np.stack([extract_classical_features(c) for c in crops])
            b = self.classical
            s = s + self.cfg['alpha'] * b['lr'].predict_log_proba(b['pca'].transform(b['scaler'].transform(f)))
        return np.exp(_log_softmax(s / self.cfg['T2']))

    def decide(self, p):
        top = np.argsort(-p)[:5]
        c = self.classes[top[0]]
        mp = np.bincount(self.make_idx, weights=p, minlength=len(self.make_names))
        mk = int(mp.argmax())
        conf = float(p[top[0]])
        cands = ', '.join(self.classes[i]['name'] for i in top[:3])
        if conf >= self.cfg['tau']:
            status, msg = 'confident', f"{c['make']} {c['model']} ({c['year']}) — {conf:.1%}"
        elif mp[mk] >= self.cfg['tau_make']:
            status, msg = 'uncertain_model', f"Likely a {self.make_names[mk]} ({mp[mk]:.0%}); exact model/year uncertain. Candidates: {cands}"
        else:
            status, msg = 'low_confidence', f"Low confidence — try a clearer / closer image. Top guesses: {cands}"
        return dict(label=c['name'], make=c['make'], model=c['model'], year=c['year'], class_id=int(top[0]),
                    confidence=conf, status=status, message=msg,
                    make_guess=self.make_names[mk], make_confidence=float(mp[mk]),
                    top5=[dict(label=self.classes[i]['name'], confidence=float(p[i])) for i in top])

    # ---------- explainability ----------
    def gradcam(self, crop_bgr):
        if self.clf is None:
            return None
        if self._cam is None:
            layer = self.clf.stages[-1] if hasattr(self.clf, 'stages') else list(self.clf.children())[-3]
            def hook(mod, inp, out):
                self._act = out
                if out.requires_grad:
                    out.register_hook(lambda g: setattr(self, '_grad', g))
            layer.register_forward_hook(hook)
            self._cam = True
        x = self.tf(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))[None].to(self.device)
        self.clf.zero_grad()
        with torch.enable_grad():
            out = self.clf(x.requires_grad_(True))
            out[0, int(out.argmax(1))].backward()
        w = self._grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * self._act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=crop_bgr.shape[:2], mode='bilinear')[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.cvtColor((0.55 * crop_bgr + 0.45 * heat).astype(np.uint8), cv2.COLOR_BGR2RGB)

    # ---------- drawing ----------
    @staticmethod
    def draw(bgr, vehicles):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        th = max(2, int(max(rgb.shape[:2]) / 300))
        for v in vehicles:
            x1, y1, x2, y2 = [int(t) for t in v['box']]
            col = STATUS_COLORS[v['status']]
            cv2.rectangle(rgb, (x1, y1), (x2, y2), col, th)
            txt = f"{v['label']} {v['confidence']:.0%}" if v['status'] == 'confident' else f"{v['make_guess']}? {v['make_confidence']:.0%}"
            fs = max(0.5, th / 3)
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th // 2))
            yt = max(tht + 6, y1)
            cv2.rectangle(rgb, (x1, yt - tht - 6), (x1 + tw + 6, yt), col, -1)
            cv2.putText(rgb, txt, (x1 + 3, yt - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
        return rgb

    # ---------- full pipeline ----------
    def predict(self, image, max_vehicles=5, return_stages=True, gradcam=False):
        t0 = time.time()
        bgr = to_bgr(image)
        dets = self.detect(bgr, max_vehicles)
        t1 = time.time()
        crops = [crop_box(bgr, d[0], self.cfg['crop_pad']) for d in dets]
        probs = self.classify(crops)
        t2 = time.time()
        vehicles = []
        for (box, dconf, src), p in zip(dets, probs):
            v = self.decide(p)
            v.update(box=[float(t) for t in box], det_conf=dconf, det_source=src)
            vehicles.append(v)
        out = dict(vehicles=vehicles, annotated_image=self.draw(bgr, vehicles),
                   timings_ms=dict(detection=(t1 - t0) * 1000, classification=(t2 - t1) * 1000))
        if return_stages:
            out['stages'], out['classical_info'] = processing_stages(bgr, dets[0][0])
        if gradcam:
            out['gradcam'] = self.gradcam(crops[0])
        out['timings_ms']['total'] = (time.time() - t0) * 1000
        return out
