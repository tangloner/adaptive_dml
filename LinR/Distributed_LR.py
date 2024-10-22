import tensorflow as tf
import numpy as np
import argparse
import sys

#open 3 cmd line,each starts the tensorflow by "source activate tensorflow".
#Then, input of each as follows
#python distributed_example.py --job_name=ps --ps_hosts=localhost:1111 --worker_hosts=localhost:2222,localhost:2223 --task_index=0
#python distributed_example.py --job_name=worker --ps_hosts=localhost:1111 --worker_hosts=localhost:2222,localhost:2223 --task_index=0
#python distributed_example.py --job_name=worker --ps_hosts=localhost:1111 --worker_hosts=localhost:2222,localhost:2223 --task_index=1

FLAGS = None

def main(_):
	ps_hosts = FLAGS.ps_hosts.split(',')
	worker_hosts = FLAGS.worker_hosts.split(',')
	cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,'worker': worker_hosts})
	server = tf.train.Server(
                    {'ps': ps_hosts,'worker': worker_hosts},
                    job_name=FLAGS.job_name,
                    task_index=FLAGS.task_index)
	if FLAGS.job_name == 'ps':
		server.join()
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
        
        x_data = tf.placeholder(tf.float32, [100])
        y_data = tf.placeholder(tf.float32, [100])

        W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = W * x_data + b
        loss = tf.reduce_mean(tf.square(y - y_data))
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(loss, global_step=global_step)
        
        tf.summary.scalar('cost', loss)
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
    # The StopAtStepHook handles stopping after running given steps.
    hooks = [ tf.train.StopAtStepHook(last_step=1000000)]
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master="grpc://" + worker_hosts[FLAGS.task_index],
                                           is_chief=(FLAGS.task_index==0), # 我们制定task_index为0的任务为主任务，用于负责变量初始化、做checkpoint、保存summary和复原
                                           checkpoint_dir="/tmp/tf_train_logs",
                                           save_checkpoint_secs=None,
                                           hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step asynchronously.
            # See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training.
            # mon_sess.run handles AbortedError in case of preempted PS.
            train_x = np.random.rand(1000).astype(np.float32)
            train_y = train_x * 0.1 + np.random.rand(*train_x.shape)*0.8+ 10
            _, step, loss_v, weight, biase = mon_sess.run([train_op, global_step, loss, W, b], feed_dict={x_data: train_x, y_data: train_y})
            if step % 100 == 0:
                print "step: %d, weight: %f, biase: %f, loss: %f" %(step, weight, biase, loss_v)
        print "Optimization finished."

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")
	# Flags for defining the tf.train.ClusterSpec
	parser.add_argument(
		"--ps_hosts",
		type=str,
		default="",
		help="Comma-separated list of hostname:port pairs"
	)
	parser.add_argument(
		"--worker_hosts",
		type=str,
		default="",
		help="Comma-separated list of hostname:port pairs"
	)
	parser.add_argument(
		"--job_name",
		type=str,
		default="",
		help="One of 'ps', 'worker'"
	)
	# Flags for defining the tf.train.Server
	parser.add_argument(
		"--task_index",
		type=int,
		default=0,
		help="Index of task within the job"
	)
	FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

