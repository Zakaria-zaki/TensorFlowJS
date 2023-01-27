import onnx_tf
import onnx

onnx_model = onnx.load("model.onnx")
tf_rep = onnx_tf.backend.prepare(onnx_model)

# The `tf_rep` object contains the TensorFlow model in the form of a TensorFlow GraphDef
tf_rep.save("model.pb")