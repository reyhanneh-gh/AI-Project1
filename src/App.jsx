import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity, Camera, CheckCircle2, CircleAlert, Cpu, Hand, ScanFace,
  Play, RefreshCw, Trash2, BrainCircuit, Sparkles, ShieldCheck
} from "lucide-react";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import {
  initializeTensorflow, cropFromVideo, GESTURES, EXPRESSIONS,
  createCnnBaseline, createAttentionModel, makeDataset, evaluatePredictions,
  predictSample
} from "./ml.js";
import "@mediapipe/face_mesh";
import "@mediapipe/hands";

function boundsFromPoints(points, padding = 0) {
  if (!points?.length) return null;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const x = Math.min(...xs), y = Math.min(...ys);
  const right = Math.max(...xs), bottom = Math.max(...ys);
  return { x: Math.max(0, x - padding), y: Math.max(0, y - padding), w: right - x + padding * 2, h: bottom - y + padding * 2 };
}

function faceBox(face) {
  const tl = face?.box?.xMin !== undefined ? face.box : null;
  if (tl) return { x: tl.xMin, y: tl.yMin, w: tl.width, h: tl.height };
  const pts = face?.keypoints?.map(p => ({ x: p.x, y: p.y })) || [];
  return boundsFromPoints(pts, 8);
}

function handBox(hand) {
  const pts = hand?.keypoints?.map(p => ({ x: p.x, y: p.y })) || [];
  return boundsFromPoints(pts, 12);
}

function pct(x) { return `${Math.round((x || 0) * 100)}%`; }

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const detectorRef = useRef({ face: null, hand: null });
  const latestRef = useRef({ faceCanvas: null, handCanvas: null, faceBox: null, handBox: null });
  const samplesRef = useRef([]);
  const modelsRef = useRef({ baseline: null, attention: null });
  const runningRef = useRef(false);
  const rafRef = useRef(null);

  const [status, setStatus] = useState("Preparing TensorFlow.js…");
  const [cameraOn, setCameraOn] = useState(false);
  const [detectorsReady, setDetectorsReady] = useState(false);
  const [faceFound, setFaceFound] = useState(false);
  const [handFound, setHandFound] = useState(false);
  const [gestureLabel, setGestureLabel] = useState(0);
  const [expressionLabel, setExpressionLabel] = useState(0);
  const [sampleCount, setSampleCount] = useState(0);
  const [samplesByGesture, setSamplesByGesture] = useState([0, 0, 0, 0]);
  const [samplesByExpression, setSamplesByExpression] = useState([0, 0, 0, 0]);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [activeModel, setActiveModel] = useState("attention");
  const [error, setError] = useState("");
  const [backend, setBackend] = useState("loading");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const b = await initializeTensorflow();
        if (!cancelled) { setBackend(b); setStatus("Load camera and models to begin."); }
      } catch (e) {
        if (!cancelled) { setError(`TensorFlow.js could not initialize: ${e.message}`); setStatus("Initialization failed."); }
      }
    })();
    return () => { cancelled = true; stopCamera(); };
  }, []);

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const video = videoRef.current;
    if (video?.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    setCameraOn(false);
  }, []);


  const loadDetectors = async () => {
    setStatus("Loading face and hand detectors…");

    try {
      const face = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: "mediapipe",
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
          refineLandmarks: false,
          maxFaces: 1,
        }
      );


      const hand = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        {
          runtime: "mediapipe",
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
          modelType: "lite",
          maxHands: 1,
        }
      );


      detectorRef.current = { face, hand };
      setDetectorsReady(true);
      setStatus("Detectors ready. Start the webcam.");
      setError("");
    } catch (e) {
      console.error("Detector loading error:", e);

      setError(
        `Could not load detectors: ${e.message}. Check your internet connection and refresh.`
      );
      setStatus("Detector loading failed.");
    }
  };


  const startCamera = async () => {
    setError("");
    try {
      if (!detectorsReady) await loadDetectors();
      if (!detectorRef.current.face || !detectorRef.current.hand) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      setCameraOn(true);
      setStatus("Camera active. Keep your face and one hand in view.");
      runningRef.current = true;
      scanLoop();
    } catch (e) {
      setError(`Camera could not start: ${e.message}. Use localhost or HTTPS and allow camera access.`);
      setStatus("Camera unavailable.");
    }
  };

  const scanLoop = async () => {
    if (!runningRef.current) return;
    const video = videoRef.current;
    if (video && video.readyState >= 2 && detectorRef.current.face && detectorRef.current.hand) {
      try {
        const [faces, hands] = await Promise.all([
          detectorRef.current.face.estimateFaces(video, { flipHorizontal: false }),
          detectorRef.current.hand.estimateHands(video, { flipHorizontal: false })
        ]);
        const fbox = faces.length ? faceBox(faces[0]) : null;
        const hbox = hands.length ? handBox(hands[0]) : null;
        const fcanvas = fbox ? cropFromVideo(video, fbox, 0.2) : null;
        const hcanvas = hbox ? cropFromVideo(video, hbox, 0.24) : null;
        latestRef.current = { faceCanvas: fcanvas, handCanvas: hcanvas, faceBox: fbox, handBox: hbox };
        setFaceFound(Boolean(fcanvas));
        setHandFound(Boolean(hcanvas));
        drawOverlay(fbox, hbox, video.videoWidth, video.videoHeight);
        const model = modelsRef.current[activeModel];
        if (model && fcanvas && hcanvas) {
          const result = predictSample(model, fcanvas, hcanvas);
          setPredictions(result);
        } else if (!model) setPredictions(null);
      } catch (e) {
        // A dropped frame should not stop the camera loop.
      }
    }
    if (runningRef.current) rafRef.current = requestAnimationFrame(scanLoop);
  };

  const drawOverlay = (face, hand, width, height) => {
    const canvas = overlayRef.current;
    if (!canvas || !width || !height) return;
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    const draw = (box, label, color) => {
      if (!box) return;
      ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, width / 240);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
      ctx.fillStyle = color; ctx.font = `bold ${Math.max(12, width / 45)}px Inter, sans-serif`;
      ctx.fillText(label, box.x + 4, Math.max(18, box.y - 6));
    };
    draw(face, "FACE", "#8df7bd");
    draw(hand, "HAND", "#d4f77b");
  };

  useEffect(() => {
    if (cameraOn) {
      runningRef.current = true;
      scanLoop();
    }
    return () => { runningRef.current = false; if (rafRef.current) cancelAnimationFrame(rafRef.current); };
    // scanLoop is intentionally started only when cameraOn changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOn, activeModel]);

  const captureSample = () => {
    const { faceCanvas, handCanvas } = latestRef.current;
    if (!faceCanvas || !handCanvas) {
      setError("Both face and hand must be detected before capturing a sample.");
      return;
    }
    // Copy the canvases so subsequent camera frames don't mutate stored samples.
    const clone = c => {
      const n = document.createElement("canvas"); n.width = c.width; n.height = c.height;
      n.getContext("2d").drawImage(c, 0, 0); return n;
    };
    const next = [...samplesRef.current, {
      face: clone(faceCanvas), hand: clone(handCanvas),
      gesture: gestureLabel, expression: expressionLabel
    }];
    samplesRef.current = next;
    setSampleCount(next.length);
    setSamplesByGesture(GESTURES.map((_, i) => next.filter(s => s.gesture === i).length));
    setSamplesByExpression(EXPRESSIONS.map((_, i) => next.filter(s => s.expression === i).length));
    setError("");
  };

  const clearSamples = () => {
    samplesRef.current = [];
    setSampleCount(0); setSamplesByGesture([0, 0, 0, 0]); setSamplesByExpression([0, 0, 0, 0]);
    setMetrics(null); setTrained(false); setPredictions(null);
    if (modelsRef.current.baseline) modelsRef.current.baseline.dispose();
    if (modelsRef.current.attention) modelsRef.current.attention.dispose();
    modelsRef.current = { baseline: null, attention: null };
  };

  const trainModels = async () => {
    const samples = samplesRef.current;
    if (samples.length < 24) {
      setError("Capture at least 24 paired examples first. For a useful comparison, aim for 10–20 examples per class and vary the lighting and position.");
      return;
    }
    if (GESTURES.some((_, i) => !samples.some(s => s.gesture === i)) ||
      EXPRESSIONS.some((_, i) => !samples.some(s => s.expression === i))) {
      setError("Capture at least one sample for every gesture and every expression before training.");
      return;
    }
    setTraining(true); setError(""); setStatus("Training models in your browser…");
    try {
      const shuffled = [...samples].sort(() => Math.random() - 0.5);
      const splitAt = Math.max(1, Math.floor(shuffled.length * 0.8));
      const trainSet = shuffled.slice(0, splitAt);
      const valSet = shuffled.slice(splitAt);

      console.log("Stage 1: Preparing dataset...");
      const trainData = makeDataset(trainSet);

      console.log("Stage 2: Creating CNN baseline...");
      const baseline = createCnnBaseline();
      console.log("Baseline created successfully.");

      console.log("Stage 3: Creating attention model...");
      const attention = createAttentionModel();
      console.log("Attention model created successfully.");

      const models = [
        ["baseline", baseline],
        ["attention", attention]
      ];

      const nextMetrics = {};
      try {
        for (const [name, model] of models) {
          setStatus(`Training ${name === "baseline" ? "CNN baseline" : "CNN + attention"}…`);
          await model.fit(trainData.xs, trainData.ys, {
            epochs: 8, batchSize: Math.min(8, trainSet.length), shuffle: true, verbose: 0,
            callbacks: { onEpochEnd: async () => { await tfYield(); } }
          });
          nextMetrics[name] = evaluatePredictions(model, valSet.length ? valSet : trainSet);
        }
      } finally {
        trainData.xs.forEach(t => t.dispose()); trainData.ys.forEach(t => t.dispose());
      }
      if (modelsRef.current.baseline) modelsRef.current.baseline.dispose();
      if (modelsRef.current.attention) modelsRef.current.attention.dispose();
      modelsRef.current = { baseline: models[0][1], attention: models[1][1] };
      setMetrics(nextMetrics); setTrained(true); setActiveModel("attention");
      setStatus("Training complete. Compare validation scores below.");
    } catch (e) {
      setError(`Training failed: ${e.message}`);
      setStatus("Training failed.");
    } finally {
      setTraining(false);
    }
  };

  const tfYield = () => new Promise(resolve => setTimeout(resolve, 0));

  const bestGesture = predictions ? predictions.gesture.indexOf(Math.max(...predictions.gesture)) : -1;
  const bestExpression = predictions ? predictions.expression.indexOf(Math.max(...predictions.expression)) : -1;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><BrainCircuit size={22} /></div>
        <div className="brand-copy"><span className="eyebrow">COMPUTER VISION LAB</span><strong>Gesture<span className="green"> & </span>Expression</strong></div>
        <div className="topbar-right"><span className="status-dot" /><span>{backend === "loading" ? "Initializing" : `TensorFlow.js · ${backend}`}</span></div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div className="hero-copy">
            <div className="pill"><Sparkles size={14} /> TWO-MODEL EXPERIMENT</div>
            <h1>See the frame.<br /><span>Understand the signals.</span></h1>
            <p>Recognize a hand gesture and facial expression from one webcam frame. Compare a CNN baseline with a CNN + attention fusion model — right in your browser.</p>
            <div className="hero-tags"><span><ShieldCheck size={14} /> Local browser processing</span><span><Cpu size={14} /> TensorFlow.js</span></div>
          </div>
          <div className="hero-orb" aria-hidden="true"><div className="orb-ring ring-one" /><div className="orb-ring ring-two" /><div className="orb-core"><ScanFace size={44} /><Hand size={32} /></div><span className="orb-label label-face">FACE TOKEN</span><span className="orb-label label-hand">HAND TOKEN</span></div>
        </section>

        {error && <div className="alert"><CircleAlert size={18} /><span>{error}</span><button className="alert-close" onClick={() => setError("")}>Dismiss</button></div>}

        <div className="workspace-grid">
          <section className="panel camera-panel">
            <div className="panel-heading"><div><span className="section-kicker">01 / LIVE INPUT</span><h2>Webcam studio</h2></div><span className={`mini-status ${cameraOn ? "live" : ""}`}>{cameraOn ? "LIVE" : "OFFLINE"}</span></div>
            <div className="video-frame">
              <video ref={videoRef} playsInline muted className={cameraOn ? "video-visible" : "video-hidden"} />
              <canvas ref={overlayRef} className="overlay-canvas" />
              {!cameraOn && <div className="camera-placeholder"><div className="placeholder-icon"><Camera size={30} /></div><strong>Your camera view appears here</strong><span>Allow camera access to detect your face and hand.</span></div>}
              {cameraOn && <div className="frame-corner top-left" />}
              <div className="video-badge"><span className={cameraOn ? "status-dot" : "status-dot muted"} />{cameraOn ? "LIVE FEED" : "CAMERA STANDBY"}</div>
            </div>
            <div className="detection-status">
              <div className="detection-item"><span className={`detect-icon ${faceFound ? "detected" : ""}`}><ScanFace size={17} /></span><div><strong>Face</strong><span>{faceFound ? "Detected" : "Not detected"}</span></div></div>
              <div className="detection-item"><span className={`detect-icon ${handFound ? "detected" : ""}`}><Hand size={17} /></span><div><strong>Hand</strong><span>{handFound ? "Detected" : "Not detected"}</span></div></div>
            </div>
            <div className="button-row">
              {!cameraOn ? <button className="btn btn-primary" onClick={startCamera}><Camera size={16} /> Start webcam</button> : <button className="btn btn-secondary" onClick={stopCamera}>Stop camera</button>}
              {!detectorsReady && <button className="btn btn-ghost" onClick={loadDetectors}><RefreshCw size={15} /> Load detectors</button>}
            </div>
            <p className="helper-text">{status}</p>
          </section>

          <section className="panel prediction-panel">
            <div className="panel-heading"><div><span className="section-kicker">02 / MODEL OUTPUT</span><h2>Live predictions</h2></div><span className="model-chip">{activeModel === "baseline" ? "CNN BASELINE" : "CNN + ATTENTION"}</span></div>
            <div className="prediction-card">
              <div className="prediction-card-top"><span className="prediction-icon"><Hand size={19} /></span><span className="prediction-type">HAND GESTURE</span><span className="confidence">{predictions && bestGesture >= 0 ? pct(predictions.gesture[bestGesture]) : "—"}</span></div>
              <h3>{predictions && bestGesture >= 0 ? GESTURES[bestGesture] : "Waiting for input"}</h3>
              <div className="probability-list">{GESTURES.map((g, i) => <div className="probability-row" key={g}><span>{g}</span><div className="prob-track"><div className="prob-fill" style={{ width: predictions ? pct(predictions.gesture[i]) : "0%" }} /></div><small>{predictions ? pct(predictions.gesture[i]) : "—"}</small></div>)}</div>
            </div>
            <div className="prediction-card">
              <div className="prediction-card-top"><span className="prediction-icon face"><ScanFace size={19} /></span><span className="prediction-type">FACIAL EXPRESSION</span><span className="confidence">{predictions && bestExpression >= 0 ? pct(predictions.expression[bestExpression]) : "—"}</span></div>
              <h3>{predictions && bestExpression >= 0 ? EXPRESSIONS[bestExpression] : "Waiting for input"}</h3>
              <div className="probability-list">{EXPRESSIONS.map((g, i) => <div className="probability-row" key={g}><span>{g}</span><div className="prob-track"><div className="prob-fill expression-fill" style={{ width: predictions ? pct(predictions.expression[i]) : "0%" }} /></div><small>{predictions ? pct(predictions.expression[i]) : "—"}</small></div>)}</div>
            </div>
            <div className="privacy-note"><ShieldCheck size={16} /><span>Frames and training samples stay in this browser session. Nothing is uploaded by this app.</span></div>
          </section>
        </div>

        <section className="panel data-panel">
          <div className="panel-heading"><div><span className="section-kicker">03 / YOUR DATA</span><h2>Collect training samples</h2></div><div className="sample-count"><strong>{sampleCount}</strong><span>samples</span></div></div>
          <p className="section-description">Keep your face and one hand visible. Choose the labels that match what you're doing, then capture a paired face/hand crop. Vary your distance, lighting, and background.</p>
          <div className="label-columns">
            <div className="label-group"><div className="label-group-title"><Hand size={16} /> Hand gesture label</div><div className="choice-grid">{GESTURES.map((g, i) => <button key={g} className={`choice ${gestureLabel === i ? "selected" : ""}`} onClick={() => setGestureLabel(i)}><span>{g}</span><small>{samplesByGesture[i]} samples</small></button>)}</div></div>
            <div className="label-group"><div className="label-group-title"><ScanFace size={16} /> Facial expression label</div><div className="choice-grid">{EXPRESSIONS.map((g, i) => <button key={g} className={`choice ${expressionLabel === i ? "selected" : ""}`} onClick={() => setExpressionLabel(i)}><span>{g}</span><small>{samplesByExpression[i]} samples</small></button>)}</div></div>
          </div>
          <div className="button-row sample-actions"><button className="btn btn-primary" onClick={captureSample} disabled={!cameraOn}><CheckCircle2 size={16} /> Capture labeled sample</button><button className="btn btn-ghost" onClick={clearSamples} disabled={!sampleCount || training}><Trash2 size={15} /> Clear dataset</button></div>
          <div className="tip-box"><Activity size={17} /><span><strong>Data tip:</strong> Aim for 10–20 samples per class at minimum. Make sure all four gestures and all four expressions appear in your dataset. The expression label is a human-provided training label, not an objective measurement of emotion.</span></div>
        </section>

        <section className="panel compare-panel">
          <div className="panel-heading"><div><span className="section-kicker">04 / EXPERIMENT</span><h2>Compare model accuracy</h2></div><span className="experiment-tag"><Activity size={14} /> VALIDATION</span></div>
          <p className="section-description">Both models train on the same captured examples. A random 80/20 split is used for a quick prototype comparison. With small datasets, scores can be noisy — collect more varied samples for a more meaningful result.</p>
          <div className="model-compare-grid">
            <div className={`model-summary ${activeModel === "baseline" ? "active-summary" : ""}`}><div className="model-summary-title"><span className="model-symbol"><Cpu size={18} /></span><div><strong>CNN baseline</strong><small>Shared CNN + two prediction heads</small></div></div><div className="metric-pair"><div><small>Gesture accuracy</small><strong>{metrics ? pct(metrics.baseline.gesture) : "—"}</strong></div><div><small>Expression accuracy</small><strong>{metrics ? pct(metrics.baseline.expression) : "—"}</strong></div></div><div className="metric-bottom"><span>Joint accuracy</span><strong>{metrics ? pct(metrics.baseline.joint) : "Not trained"}</strong></div><button className="btn btn-secondary btn-full" disabled={!trained} onClick={() => setActiveModel("baseline")}>{activeModel === "baseline" ? "Currently selected" : "Use CNN baseline"}</button></div>
            <div className={`model-summary attention-summary ${activeModel === "attention" ? "active-summary" : ""}`}><div className="model-summary-title"><span className="model-symbol green-symbol"><BrainCircuit size={18} /></span><div><strong>CNN + attention fusion</strong><small>Shared CNN + self-attention over face/hand tokens</small></div></div><div className="metric-pair"><div><small>Gesture accuracy</small><strong>{metrics ? pct(metrics.attention.gesture) : "—"}</strong></div><div><small>Expression accuracy</small><strong>{metrics ? pct(metrics.attention.expression) : "—"}</strong></div></div><div className="metric-bottom"><span>Joint accuracy</span><strong>{metrics ? pct(metrics.attention.joint) : "Not trained"}</strong></div><button className="btn btn-primary btn-full" disabled={!trained} onClick={() => setActiveModel("attention")}>{activeModel === "attention" ? "Currently selected" : "Use attention model"}</button></div>
          </div>
          <div className="train-footer"><div className="train-note"><BrainCircuit size={17} /><span>Training runs locally in your browser. Keep this tab open until it finishes.</span></div><button className="btn btn-primary train-button" onClick={trainModels} disabled={training || sampleCount < 24}><Play size={16} />{training ? "Training models…" : "Train & compare models"}</button></div>
          {training && <div className="training-progress"><span className="progress-glow" /></div>}
        </section>
        <div className="method-note"><Sparkles size={16} /><span><strong>About the comparison:</strong> the attention model is a small Transformer-style self-attention fusion block operating on two region tokens. It is an educational prototype, not a guarantee that attention will outperform the CNN baseline. For a credible experiment, use a held-out recording session and collect many more samples.</span></div>
      </main>
      <footer className="footer"><div className="footer-line" /><span>developed by <strong>Reyhaneh Ghalambor</strong> <i>— 2026</i></span><span className="footer-dot">•</span><span>Built with TensorFlow.js</span></footer>
    </div>
  );
}
