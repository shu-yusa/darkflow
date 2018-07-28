import tensorflow as tf

graph = tf.get_default_graph()

with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        "saved_model")
    inp = sess.graph.get_tensor_by_name('input:0')
    out = sess.graph.get_tensor_by_name('output:0') 
    print(inp)
    print(out)
    writer = tf.summary.FileWriter("./graph", graph)
    writer.close()
