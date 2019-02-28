# run black box attack on models

import numpy as np
import tensorflow as tf

import random

# construct graph for testing on original examples
def build_test(discriminator, test_image_placeholder, test_label_placeholder, config):
    x = test_image_placeholder
    x = 2.0*x - 1.0
    y = test_label_placeholder
    d_out = discriminator(x)
    predictions = tf.argmax(d_out, 1)

    # http://ronny.rest/blog/post_2017_09_11_tf_metrics/
    tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name='test_metric')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="test_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    return (tf_metric, tf_metric_update, running_vars_initializer)

# run actual test to collect performance statistics
def run_test(tf_metric, tf_metric_update, running_vars_initializer, sess, config, test_image, test_label, 
    test_image_placeholder, test_label_placeholder):

    sess.run(running_vars_initializer)

    batch_size = config['batch_size']

    num_test_examples = test_image.shape[0]
    num_steps = int(num_test_examples/batch_size)

    idx_list = list(range(num_test_examples))
    random.shuffle(idx_list)


    for i in range(num_steps):
        minibatch_idx = idx_list[:batch_size]
        idx_list = idx_list[batch_size:]

        image_minibatch = test_image[minibatch_idx,:,:,:]
        label_minibatch = test_label[minibatch_idx]

        sess.run(tf_metric_update, feed_dict={test_image_placeholder: image_minibatch, 
            test_label_placeholder: label_minibatch})

    accuracy = sess.run(tf_metric)

    return accuracy


# use discriminator2 to attack discriminator1 with FGS
def build_test_fgs(discriminator1, discriminator2, test_image_placeholder, test_label_placeholder, config):
    x = test_image_placeholder
    x = 2.0*x - 1.0
    y = test_label_placeholder
    epsilon = config['epsilon']
    class_num = config['class_num']

    d_out = discriminator2(x)
    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out, labels=tf.one_hot(y,class_num)))

    grad_x, = tf.gradients(d_loss, x)
    x_adv = tf.stop_gradient(x + epsilon*tf.sign(grad_x))
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

    adv_out = discriminator1(x_adv)
    predictions = tf.argmax(adv_out, 1)

    tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name='fgs_metric')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="fgs_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    return (tf_metric, tf_metric_update, running_vars_initializer)


# use discriminator2 to attack discriminator1 with PGD
def build_test_pgd(discriminator1, discriminator2, test_image_placeholder, test_label_placeholder, config):
    x0 = test_image_placeholder

    x0 = 2.0*x0 - 1.0
    y = test_label_placeholder
    epsilon = config['epsilon']
    class_num = config['class_num']
    pgd_iter = config['pgd_iter']

    step_size = epsilon*0.25

    # randomize
    x = x0 + tf.random_uniform(x0.shape, -epsilon, epsilon)
    for i in range(pgd_iter):
        d_out = discriminator2(x)
        d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=d_out, labels=tf.one_hot(y,class_num)))

        grad_x, = tf.gradients(d_loss, x)
        x = tf.stop_gradient(x + step_size*tf.sign(grad_x))
        x = tf.clip_by_value(x, x0 - epsilon, x0 + epsilon)
        x = tf.clip_by_value(x, -1.0, 1.0)

    adv_out = discriminator1(x)
    predictions = tf.argmax(adv_out, 1)

    tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name='pgd_metric')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="pgd_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    return (tf_metric, tf_metric_update, running_vars_initializer)


def black_box_attacks(discriminator1, discriminator2, ckpt_filename1, ckpt_filename2, test_image, test_label, config):

    batch_size = config['batch_size']
    my_dim = list(test_image.shape)
    num_test_examples = my_dim[0]
    my_dim[0] = batch_size
    test_image_placeholder = tf.placeholder(test_image.dtype, my_dim)
    my_dim = list(test_label.shape)
    my_dim[0] = batch_size
    test_label_placeholder = tf.placeholder(test_label.dtype, my_dim)

    # build test
    acc, acc_update, acc_init = build_test(discriminator1, test_image_placeholder, test_label_placeholder, config)
    acc_fgs, acc_update_fgs, acc_init_fgs = build_test_fgs(discriminator1, discriminator2, test_image_placeholder, 
        test_label_placeholder, config)
    acc_pgd, acc_update_pgd, acc_init_pgd = build_test_pgd(discriminator1, discriminator2, test_image_placeholder, 
        test_label_placeholder, config)

    var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator1')
    var_dict1 = {}
    for var in var_list1:
        var_dict1[var.op.name.replace('discriminator1', 'discriminator')] = var
    saver1 = tf.train.Saver(var_dict1)

    var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator2')
    var_dict2 = {}
    for var in var_list2:
        var_dict2[var.op.name.replace('discriminator2', 'discriminator')] = var
    saver2 = tf.train.Saver(var_dict2)    


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # retrieve trained model
        saver1.restore(sess, ckpt_filename1)
        saver2.restore(sess, ckpt_filename2)

        test_acc = run_test(acc, acc_update, acc_init, sess, config, test_image, test_label, 
            test_image_placeholder, test_label_placeholder)
        test_acc_fgs = run_test(acc_fgs, acc_update_fgs, acc_init_fgs, sess, config, test_image, test_label,
            test_image_placeholder, test_label_placeholder)
        test_acc_pgd = run_test(acc_pgd, acc_update_pgd, acc_init_pgd, sess, config, test_image, test_label, 
            test_image_placeholder, test_label_placeholder)

        print('test_acc: %.4f, fgs_acc: %.4f, pgd_acc: %.4f'
              % (test_acc, test_acc_fgs, test_acc_pgd), flush=True)

        coord.request_stop()
        coord.join(threads)



