"""
onnxruntime Backend for ONNX
============================

*onnxruntime* extends the 
`onnx backend API <https://github.com/onnx/onnx/blob/master/docs/ImplementingAnOnnxBackend.md>`_
to run predictions using this runtime.
Let's use the API to compute the prediction
of a simple logistic regression model.
"""
import numpy as np
from onnxruntime import datasets
import onnxruntime.backend as backend
from onnx import load

name = datasets.get_example("logreg_iris.onnx")
model = load(name)

rep = backend.prepare(model, 'CPU')
x = np.array([[-1.0, -2.0]], dtype=np.float32)
label, proba = rep.run(x)
print("label={}".format(label))
print("probabilities={}".format(proba))

########################################
# The device depends on how the package was compiled,
# GPU or CPU.
from onnxruntime import get_device
print(get_device())

########################################
# The backend can also directly load the model
# without using *onnx*.

rep = backend.prepare(name, 'CPU')
x = np.array([[-1.0, -2.0]], dtype=np.float32)
label, proba = rep.run(x)
print("label={}".format(label))
print("probabilities={}".format(proba))
