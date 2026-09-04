# WebGPU Deployment Guide — Embedding PlayWorld

Deploying `WorldEngine.cpp` (`PlayWorld`) into any website requires zero server-side compute infrastructure.

---

## 1. Quick Integration (4 Lines of HTML)

```html
<!-- Include viewport canvas and script -->
<canvas id="viewport" width="640" height="360" style="width: 100%; aspect-ratio: 16/9;"></canvas>
<script type="module">
    import { PlayWorldWebClient } from './player/player.js';
    const client = new PlayWorldWebClient({
        canvasElement: document.getElementById('viewport'),
        modelUrl: 'https://huggingface.co/playworld/minecraft-1.3b-q4/resolve/main/model.pwmf',
        denoisingSteps: 1,
        enableVoxelCache: true
    });
    await client.initialize();
    client.startLoop();
</script>
```

---

## 2. Browser Compatibility & Fallback Matrix

| Browser | WebGPU Acceleration | Status | Recommended Action |
|:---|:---|:---|:---|
| **Google Chrome 113+** | Native Hardware | Fully Supported | None required (works out of the box) |
| **Microsoft Edge 113+**| Native Hardware | Fully Supported | None required |
| **Safari 18+ (macOS/iPadOS)**| Native Hardware | Supported | Enable `WebGPU` feature flag in Advanced settings |
| **Firefox** | Experimental | Fallback Stream | Low-latency 60 FPS WebRTC video fallback |
| **Mobile Safari / Chrome** | Mobile WebGPU | Click-to-Load | Interactive preview gate with click-to-load consent |

---

## 3. Mobile Click-to-Load Consent Gate

To protect mobile users on cellular data plans and avoid iOS WebKit memory limits (~1.5 GB process ceiling):
1. Mobile visitors see a lightweight 60 FPS video preview loop (<1.5 MB).
2. An explicit button **"Tap to Load Neural World (~820 MB)"** is presented.
3. Once tapped by the user, the full WebGPU pipeline is initialized.
