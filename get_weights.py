from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework.graph_util import convert_variables_to_constants

export_version = 1
export_path = 'output_graph_ckpt'
work_dir = '/tmp'

if __name__ == '__main__':

    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)  # test mode

    model = load_model("maxout.model.hdf5")
    model.save_weights('maxout.weights.hdf5')
    model.summary()

    config = model.to_json()
    weights = model.get_weights()

    from keras.models import model_from_json
    new_model = model_from_json(config)
    new_model.set_weights(weights)

    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)

    print(model.output)
    signature = exporter.classification_signature(
        input_tensor=model.input, scores_tensor=model.output)
    # model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
    print("Signature:", signature)

    # model_exporter.export(export_path, tf.constant(export_version), sess)
    # print('Done exporting as checkpoint file!')

    print('Saving as PB file!')
    # http://stackoverflow.com/questions/34343259/is-there-an-example-on-how-to-generate-protobuf-files-holding-trained-tensorflow
    minimal_graph = convert_variables_to_constants(sess, sess.graph.as_graph_def(), ["Softmax"])
    tf.train.write_graph(minimal_graph, '.', 'maxout.model.pb', as_text=False)
    tf.train.write_graph(minimal_graph, '.', 'maxout.model.txt', as_text=True)
    print('Done!')

def print_layer_weights(model):
    for layer in model.layers:
        cfg=layer.get_config()
        weights=layer.get_weights()
        if len(weights) >= 2:
            print(cfg)
            for weight in weights:
                print("\t" + str(weight.shape))
                print("\t" + str(weight.dtype))
