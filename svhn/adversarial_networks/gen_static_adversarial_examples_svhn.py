# Generate adversarial examples for EAT

import numpy as np
import tensorflow as tf

import random

def gen_examples_fgs(discriminator, train_image_placeholder, train_label_placeholder, config):
    x = train_image_placeholder
    x = 2.0*x - 1.0

    epsilon = config['epsilon']
    class_num = config['class_num']

    d_out = discriminator(x)
    y = tf.stop_gradient(tf.argmax(d_out, 1))    
    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out, labels=tf.one_hot(y,class_num)))

    grad_x, = tf.gradients(d_loss, x)
    x_adv = tf.stop_gradient(x + epsilon*tf.sign(grad_x))
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

    x_adv = (x_adv+1.0)/2.0

    return x_adv


# least likely class
def gen_examples_ll(discriminator, train_image_placeholder, train_label_placeholder, config):
    x = train_image_placeholder
    x = 2.0*x - 1.0
    epsilon = config['epsilon']
    class_num = config['class_num']

    d_out = discriminator(x)
    yhat = tf.stop_gradient(tf.argmin(d_out, 1))

    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out, labels=tf.one_hot(yhat,class_num)))

    grad_x, = tf.gradients(d_loss, x)
    x_adv = tf.stop_gradient(x - epsilon*tf.sign(grad_x))
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

    x_adv = (x_adv+1.0)/2.0

    return x_adv


def gen_examples_pgd(discriminator, train_image_placeholder, train_label_placeholder, config):
    x0 = train_image_placeholder

    x0 = 2.0*x0 - 1.0
    y = train_label_placeholder
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
def gen_adv_examples(discriminator, ckpt_filename, config, train_image, train_label, test_image, test_label):

    batch_size = config['batch_size']

    my_dim = list(train_image.shape)
    num_training_examples = my_dim[0]
    my_dim[0] = batch_size
    image_placeholder = tf.placeholder(train_image.dtype, my_dim)
    my_dim = list(train_label.shape)
    my_dim[0] = batch_size
    label_placeholder = tf.placeholder(train_label.dtype, my_dim)


    x_fgs = gen_examples_fgs(discriminator, image_placeholder, label_placeholder, config)
    x_pgd = gen_examples_pgd(discriminator, image_placeholder, label_placeholder, config)
    x_ll = gen_examples_ll(discriminator, image_placeholder, label_placeholder, config)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    saver = tf.train.Saver(var_list)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, ckpt_filename)

        idx_list = list(range(num_training_examples))
        random.shuffle(idx_list)

        train_size = num_training_examples

        minibatch_idx = idx_list[:batch_size]
        idx_list = idx_list[batch_size:]

        image_minibatch = train_image[minibatch_idx,:,:,:]
        label_minibatch = train_label[minibatch_idx]

        samples_fgs, samples_pgd, samples_ll= sess.run([x_fgs, x_pgd, x_ll], 
            feed_dict={image_placeholder: image_minibatch, label_placeholder: label_minibatch})

        num_steps = int(train_size/batch_size)
        my_dim = list(samples_fgs.shape)
        my_dim = [my_dim[0]*4*num_steps] + my_dim[1:]
        examples = np.zeros(my_dim)
        labels = np.zeros(4*num_steps*batch_size)
        i=0
        examples[(4*i*batch_size):((4*i+1)*batch_size),:,:,:] = samples_fgs
        examples[((4*i+1)*batch_size):((4*i+2)*batch_size),:,:,:] = samples_pgd
        examples[((4*i+2)*batch_size):((4*i+3)*batch_size),:,:,:] = samples_ll
        examples[((4*i+3)*batch_size):((4*i+4)*batch_size),:,:,:] = image_minibatch

        label_minibatch = np.squeeze(label_minibatch)
        labels[(4*i*batch_size):((4*i+1)*batch_size)] = label_minibatch
        labels[((4*i+1)*batch_size):((4*i+2)*batch_size)] = label_minibatch
        labels[((4*i+2)*batch_size):((4*i+3)*batch_size)] = label_minibatch
        labels[((4*i+3)*batch_size):((4*i+4)*batch_size)] = label_minibatch


        for i in range(1, num_steps):

            if len(idx_list)<batch_size:
                rand_list = list(range(num_training_examples))
                random.shuffle(rand_list)
                idx_list += rand_list

            minibatch_idx = idx_list[:batch_size]
            idx_list = idx_list[batch_size:]

            image_minibatch = train_image[minibatch_idx,:,:,:]
            label_minibatch = train_label[minibatch_idx]


            samples_fgs, samples_pgd, samples_ll = sess.run([x_fgs, x_pgd, x_ll], 
                feed_dict={image_placeholder: image_minibatch, label_placeholder: label_minibatch})

            examples[(4*i*batch_size):((4*i+1)*batch_size),:,:,:] = samples_fgs
            examples[((4*i+1)*batch_size):((4*i+2)*batch_size),:,:,:] = samples_pgd
            examples[((4*i+2)*batch_size):((4*i+3)*batch_size),:,:,:] = samples_ll
            examples[((4*i+3)*batch_size):((4*i+4)*batch_size),:,:,:] = image_minibatch

            label_minibatch = np.squeeze(label_minibatch)
            labels[(4*i*batch_size):((4*i+1)*batch_size)] = label_minibatch
            labels[((4*i+1)*batch_size):((4*i+2)*batch_size)] = label_minibatch
            labels[((4*i+2)*batch_size):((4*i+3)*batch_size)] = label_minibatch
            labels[((4*i+3)*batch_size):((4*i+4)*batch_size)] = label_minibatch



        # test data
        test_examples = test_image
        test_labels = test_label

        coord.request_stop()
        coord.join(threads)

    # save numpy examples
    np.savez(config['sample_file'], examples=examples, labels=labels)
    np.savez(config['test_sample_file'], examples=test_examples, labels=test_labels)





