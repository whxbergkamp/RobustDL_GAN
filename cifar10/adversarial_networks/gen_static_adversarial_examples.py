# Generate adversarial examples for EAT

import numpy as np
import tensorflow as tf



def gen_examples_fgs(discriminator, train_data, config):
    x = train_data[0]
    x = 2.0*x - 1.0

    epsilon = config['epsilon']
    d_out = discriminator(x)
    y = tf.stop_gradient(tf.argmax(d_out, 1))
    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out, labels=tf.one_hot(y,10)))

    grad_x, = tf.gradients(d_loss, x)
    x_adv = tf.stop_gradient(x + epsilon*tf.sign(grad_x))
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

    x_adv = (x_adv+1.0)/2.0

    return x_adv


# least likely class
def gen_examples_ll(discriminator, train_data, config):
    x = train_data[0]
    x = 2.0*x - 1.0
    epsilon = config['epsilon']

    d_out = discriminator(x)
    yhat = tf.stop_gradient(tf.argmin(d_out, 1))

    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out, labels=tf.one_hot(yhat,10)))

    grad_x, = tf.gradients(d_loss, x)
    x_adv = tf.stop_gradient(x - epsilon*tf.sign(grad_x))
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

    x_adv = (x_adv+1.0)/2.0

    return x_adv


def gen_examples_pgd(discriminator, train_data, config):
    x0 = train_data[0]

    x0 = 2.0*x0 - 1.0
    y = train_data[1]
    epsilon = config['epsilon']
    class_num = config['class_num']
    pgd_iter = config['pgd_iter']

    step_size = epsilon*0.25

    # randomize
    x = x0 + tf.random_uniform(x0.shape, -epsilon, epsilon)
    for i in range(pgd_iter):
        d_out = discriminator(x)
        d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=d_out, labels=tf.one_hot(y,class_num)))

        grad_x, = tf.gradients(d_loss, x)
        x = tf.stop_gradient(x + step_size*tf.sign(grad_x))
        x = tf.clip_by_value(x, x0 - epsilon, x0 + epsilon)
        x = tf.clip_by_value(x, -1.0, 1.0)

    x_adv = (x+1.0)/2.0

    return x_adv


# one model def for one model filename
def gen_adv_examples(discriminator, ckpt_filename, train_data, test_data, config):

    batch_size = config['batch_size']
    x_fgs = gen_examples_fgs(discriminator, train_data, config)
    x_pgd = gen_examples_pgd(discriminator, train_data, config)
    x_ll = gen_examples_ll(discriminator, train_data, config)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    saver = tf.train.Saver(var_list)

    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    time_diff = tf.placeholder(tf.float32)
    Wall_clock_time = tf.Variable(0., trainable=False)
    update_Wall_op = Wall_clock_time.assign_add(time_diff)


    #with sv.managed_session() as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, ckpt_filename)


        train_size = config['train_size']
        samples_fgs, samples_pgd, samples_ll, x, y = sess.run([x_fgs, x_pgd, x_ll, train_data[0], train_data[1]])

        num_steps = int(train_size/batch_size)
        my_dim = list(samples_fgs.shape)
        my_dim = [my_dim[0]*4*num_steps] + my_dim[1:]
        examples = np.zeros(my_dim)
        labels = np.zeros(4*num_steps*batch_size)
        i=0
        examples[(4*i*batch_size):((4*i+1)*batch_size),:,:,:] = samples_fgs
        examples[((4*i+1)*batch_size):((4*i+2)*batch_size),:,:,:] = samples_pgd
        examples[((4*i+2)*batch_size):((4*i+3)*batch_size),:,:,:] = samples_ll
        examples[((4*i+3)*batch_size):((4*i+4)*batch_size),:,:,:] = x

        labels[(4*i*batch_size):((4*i+1)*batch_size)] = y
        labels[((4*i+1)*batch_size):((4*i+2)*batch_size)] = y
        labels[((4*i+2)*batch_size):((4*i+3)*batch_size)] = y
        labels[((4*i+3)*batch_size):((4*i+4)*batch_size)] = y

        for i in range(1, num_steps):
            samples_fgs, samples_pgd, samples_ll, x, y = sess.run([x_fgs, x_pgd, x_ll, train_data[0], train_data[1]])

            examples[(4*i*batch_size):((4*i+1)*batch_size),:,:,:] = samples_fgs
            examples[((4*i+1)*batch_size):((4*i+2)*batch_size),:,:,:] = samples_pgd
            examples[((4*i+2)*batch_size):((4*i+3)*batch_size),:,:,:] = samples_ll
            examples[((4*i+3)*batch_size):((4*i+4)*batch_size),:,:,:] = x

            labels[(4*i*batch_size):((4*i+1)*batch_size)] = y
            labels[((4*i+1)*batch_size):((4*i+2)*batch_size)] = y
            labels[((4*i+2)*batch_size):((4*i+3)*batch_size)] = y
            labels[((4*i+3)*batch_size):((4*i+4)*batch_size)] = y



        # test data
        test_size = config['test_size']
        num_test_steps = int(test_size/batch_size)
        x_test, y_test = sess.run([test_data[0], test_data[1]])
        test_dim = list(x_test.shape)
        test_dim = [num_test_steps*batch_size]+test_dim[1:]
        test_examples = np.zeros(test_dim)
        test_labels = np.zeros(num_test_steps*batch_size)
        i = 0
        test_examples[(i*batch_size):((i+1)*batch_size),:,:,:] = x_test
        test_labels[(i*batch_size):((i+1)*batch_size)] = y_test

        for i in range(1, num_test_steps):
            x_test, y_test = sess.run([test_data[0], test_data[1]])
            test_examples[(i*batch_size):((i+1)*batch_size),:,:,:] = x_test
            test_labels[(i*batch_size):((i+1)*batch_size)] = y_test


        coord.request_stop()
        #sess.run(model.queue.close(cancel_pending_enqueues=True))
        coord.join(threads)

    # save numpy examples
    np.savez(config['sample_file'], examples=examples, labels=labels)
    np.savez(config['test_sample_file'], examples=test_examples, labels=test_labels)





