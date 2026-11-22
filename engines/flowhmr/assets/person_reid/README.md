# Local person re-identification models

FlowHMR uses these models on CPU to compare anonymous observations from the same video at shot changes. The weights are installed with the application. Matching runs offline and keeps the feature vectors in memory for the duration of the comparison.

| File | Source | License | SHA-256 |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | [MIT](YuNet-LICENSE.txt) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec_int8.onnx` | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx) | [Apache-2.0](SFace-LICENSE.txt) | `2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a` |
| `person-reidentification-retail-0270-opencv.onnx` | [Intel Open Model Zoo, 2023.0](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-reidentification-retail-0286/model.yml) | [Apache-2.0](OpenVINO-LICENSE.txt) | `cc366e94f1ff03910686b0ec7e12e6e05d0c50f131f71021c6092ff6835e6a8b` |

The OpenVINO manifest lists `person-reidentification-retail-0270.onnx` as the original ONNX for its `person-reidentification-retail-0286` IR. Its SHA-384 is `d6f60731c58b9d9493ac8bc2e8e27ad26b6666133cd8cd2ed8af35e574e16a83def51938927c0e721ef053285c06db12`.

The bundled copy expresses the twelve `MeanVarianceNormalization` nodes as spatial mean subtraction, spatial RMS and division with the original `1e-9` epsilon. This allows OpenCV 4.11 to read the ONNX with its own DNN backend. Validation compared the converted graph in OpenCV against the unmodified graph in ONNX Runtime with cosine similarity above `0.99999` for every checked input.
