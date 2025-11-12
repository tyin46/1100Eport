import os, time, urllib.request, numpy as np
import openvino as ov

MODEL_URL  = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
MODEL_PATH = "mobilenetv2-12.onnx"

if not os.path.exists(MODEL_PATH):
    print("Downloading model…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

core = ov.Core()
print("Devices:", core.available_devices)   # should include 'NPU'

model = core.read_model(MODEL_PATH)

# Force static input shape: [1,3,224,224]
input_tensor_name = model.input(0).get_any_name()
static_shape = [1, 3, 224, 224]
model.reshape({input_tensor_name: ov.PartialShape(static_shape)})

# Compile on NPU
compiled = core.compile_model(model, device_name="NPU")

inp = np.random.rand(*static_shape).astype("float32")
infer = compiled.create_infer_request()

# Warmup
for _ in range(10):
    infer.infer({compiled.inputs[0]: inp})

print("Running loop on NPU… (watch Task Manager → Performance → NPU)")
t0 = time.time()
iters = 2000
for _ in range(iters):
    infer.infer({compiled.inputs[0]: inp})
t1 = time.time()
print(f"Done {iters} iters in {(t1 - t0):.2f}s  ->  {(iters/(t1-t0)):.1f} inf/s")
