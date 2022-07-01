# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:55:00 2022

@author: kevin

"""

import tensorflow as tf

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        
        return graph
    

    
graph=load_pb('frozen_model/simple_frozen_graph_3.pb')
graph_protobuf = text_format.Parse(f.read(),  tf.compat.v1.GraphDef())

input = graph.get_tensor_by_name('Sequential')
output = graph.get_tensor_by_name('output:0')



