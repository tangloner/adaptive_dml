
"""Distributed MNIST training and validation, with model replicas.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#Construct the cluster and start the server
ps_spec = sys.argv[1].split(",")
print(ps_spec)
worker_spec = sys.argv[2].split(",")
print(worker_spec)

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/root/adaptive_dml/MNIST_data",
                    "Directory for storing mnist data")
flags.DEFINE_integer("task_index", 0,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("train_steps", 100,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", True,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_float("target_acc", 0.4, "The target accuracy of trained model")
FLAGS = flags.FLAGS


IMAGE_PIXELS = 28
target_acc = FLAGS.target_acc
batch_size = FLAGS.batch_size
tot_epoch = FLAGS.train_steps

def main(unused_argv):
    mnist = input_data.read_data_sets('/root/adaptive_dml/MNIST_data', one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index =="":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    # Get the number of workers.
    num_workers = len(worker_spec)

    cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()
    
    is_chief = (FLAGS.task_index == 0)
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device,cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Ops: located on the worker specified with FLAGS.task_index
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        # Variables of the hidden layer
        W1 = tf.Variable(tf.random_normal([784, 10]))
        #W2 = tf.Variable(tf.random_normal([50, 10]))
        b1 = tf.Variable(tf.zeros([10]))
        #b2 = tf.Variable(tf.zeros([10]))
        # Variables of the softmax layer
        y = tf.add(tf.matmul(x,W1),b1)
        #a2 = tf.nn.sigmoid(z2)
        #z3 = tf.add(tf.matmul(a2,W2),b2)
        #y = tf.nn.softmax(z3)

   
        cross_entropy =tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

        if FLAGS.replicas_to_aggregate is None:
            replicas_to_aggregate = num_workers
        else:
            replicas_to_aggregate = FLAGS.replicas_to_aggregate

        opt = tf.train.SyncReplicasOptimizer(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="mnist_sync_replicas")

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        if FLAGS.sync_replicas:
            local_init_op = opt.local_step_init_op
        if is_chief:
            local_init_op = opt.chief_init_op

        ready_for_local_init_op = opt.ready_for_local_init_op

        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = opt.get_chief_queue_runner()
        sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        if FLAGS.sync_replicas:
          sv = tf.train.Supervisor(
              is_chief=is_chief,
              logdir=train_dir,
              init_op=init_op,
              local_init_op=local_init_op,
              ready_for_local_init_op=ready_for_local_init_op,
              recovery_wait_secs=1,
              global_step=global_step)
        else:
          sv = tf.train.Supervisor(
              is_chief=is_chief,
              logdir=train_dir,
              init_op=init_op,
              recovery_wait_secs=1,
              global_step=global_step)


        if is_chief:
          print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
          print("Worker %d: Waiting for session to be initialized..." % FLAGS.task_index)

    sess = sv.prepare_or_wait_for_session(server.target)
    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    # Perform training
    time_begin = time.time()
    print("Training begins...")
    local_step = 0
    final_acc = 0
    epcho = 0
    #while( final_acc < target_acc):
    while True:
    #for epoch in range(tot_epoch):
      # Training feed
      print("%d" % epoch)
      print("%d" % tot_epoch)
      batch_count = int(mnist.train.num_examples/batch_size)
      local_step = 0
      i = 0
      for i in range(batch_count):
        print ("%d" % i)
        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}
        _, step = sess.run([train_step, global_step], feed_dict=train_feed)
        final_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        local_step += 1
      
        now = time.time()
        if local_step % 100 ==0 or local_step+1 == batch_count:
          print("Worker %d: Step: %d," % (FLAGS.task_index, step+1), "Epoch: %2d, " % int(step/batch_count), "Batch: %3d of %3d, " % (step%batch_count, batch_count), "AvgTime: %3.2fms "% float(now-time_begin))
      if final_acc >= target_acc:
        break

 
    time_end = time.time()
    print("Training ends...")
    training_time = time_end - time_begin
    print("Training elapsed time: %4.2f s" % training_time)
    print("Finall accuracy: %2.2f" % final_acc)

    # Validation feed
#    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
#    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
#    print("After %d training step(s), validation cross entropy = %g" % (FLAGS.train_steps, val_xent))

if __name__ == "__main__":
  tf.app.run()
