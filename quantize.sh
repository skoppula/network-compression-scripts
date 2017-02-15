# git pull --recursive-submodules [tensorflow-url]
# bazel build tensorflow/tools/quantization:quantize_graph
export PATH=$HOME/bazelws/tensorflow/bazel-bin/tensorflow/tools/quantization:$PATH
quantize_graph --input=maxout.model.pb --print_nodes  --output_node_names="Softmax" --output=maxout.model.quantized.pb --mode=eightbit

# vi ../tensorflow/tools/quantization/quantize_graph.py
