/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@latest/dist/tfjs-backend-wasm.wasm');

const stats = new Stats();
stats.showPanel(1);
document.body.prepend(stats.domElement);

let model;
let ctx;
let videoWidth;
let videoHeight;
let video;
let canvas;
let videoSource = ['./test_videos/aassnaulhq.mp4',
                   './test_videos/aayfryxljh.mp4'];
let pause = false;
let i = 0;
let renderCount = 0;
let avg = 0;

const state = {
  backend: 'wasm',
};

const gui = new dat.GUI();
gui.add(state, 'backend', ['wasm', 'webgl', 'cpu']).onChange(
    async (backend) => {
await tf.setBackend(backend);
});

function loadNext(videoNum) {
    video.src = videoSource[videoNum];
    video.load();
}

async function setupCamera() {
  video = document.getElementById('video');
  video.type = 'video/mp4';

  loadNext(i);

  return new Promise((resolve) => {
    video.oncanplaythrough = () => {
        videoWidth = video.videoWidth;
        videoHeight = video.videoHeight;
        video.width = videoWidth;
        video.height = videoHeight;

        canvas.width = videoWidth;
        canvas.height = videoHeight;

        video.play();
        renderPrediction();
        resolve(video);
    };
    video.onended = () => {
        i+=1;
        if (i == videoSource.length) {
          pause = true;
          console.log('played all');
          console.log(avg);
        } else {
          loadNext(i);
        }
    };
  });
}

const renderPrediction = async () => {
  if (pause) return;
  if (video.readyState < 2) {
      // When the video is not loaded enough, it doesn't make sesne to run
      // prediction. So simply return.
      return;
  }
  let startTime = performance.now();
  stats.begin();

  const returnTensors = false;
  const flipHorizontal = true;
  const annotateBoxes = true;
  const predictions = await model.estimateFaces(
    video, returnTensors, flipHorizontal, annotateBoxes);

  if (predictions.length > 0) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < predictions.length; i++) {
      if (returnTensors) {
        predictions[i].topLeft = predictions[i].topLeft.arraySync();
        predictions[i].bottomRight = predictions[i].bottomRight.arraySync();
        if (annotateBoxes) {
          predictions[i].landmarks = predictions[i].landmarks.arraySync();
        }
      }

      const start = predictions[i].topLeft;
      const end = predictions[i].bottomRight;
      const size = [end[0] - start[0], end[1] - start[1]];
      ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
      ctx.fillRect(start[0], start[1], size[0], size[1]);

      if (annotateBoxes) {
        const landmarks = predictions[i].landmarks;

        ctx.fillStyle = 'blue';
        for (let j = 0; j < landmarks.length; j++) {
          const x = landmarks[j][0];
          const y = landmarks[j][1];
          ctx.fillRect(x, y, 5, 5);
        }
      }
    }
  }

  stats.end();
  let endTime = performance.now();
  avg = ((avg * renderCount) + (endTime - startTime)) / (renderCount + 1);
  renderCount += 1;

  requestAnimationFrame(renderPrediction);
};

const setupPage = async () => {
  await tf.setBackend(state.backend);
  model = await blazeface.load();

  canvas = document.getElementById('output');
  ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';

  await setupCamera();
};

setupPage();
