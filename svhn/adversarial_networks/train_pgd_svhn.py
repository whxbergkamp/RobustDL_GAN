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


# construct graph for testing on examples perturbed with FGS
def build_test_fgs(discriminator, test_image_placeholder, test_label_placeholder, config):
    x = test_image_placeholder
    x = 2.0*x - 1.0
    y = test_label_placeholder
    epsilon = config['epsilon']
    class_num = config['class_num']

    d_out = discriminator(x)
    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out, labels=tf.one_hot(y,class_num)))

    grad_x, = tf.gradients(d_loss, x)
    x_adv = tf.stop_gradient(x + epsilon*tf.sign(grad_x))
    x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)

    adv_out = discriminator(x_adv)
    predictions = tf.argmax(adv_out, 1)

    tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name='fgs_metric')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="fgs_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    return (tf_metric, tf_metric_update, running_vars_initializer)


# construct graph for testing on examples perturbed with PGD
def build_test_pgd(discriminator, test_image_placeholder, test_label_placeholder, config):
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
        d_out = discriminator(x)
        d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=d_out, labels=tf.one_hot(y,class_num)))

        grad_x, = tf.gradients(d_loss, x)
        x = tf.stop_gradient(x + step_size*tf.sign(grad_x))
        x = tf.clip_by_value(x, x0 - epsilon, x0 + epsilon)
        x = tf.clip_by_value(x, -1.0, 1.0)

    adv_out = discriminator(x)
    predictions = tf.argmax(adv_out, 1)

    tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name='pgd_metric')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="pgd_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    return (tf_metric, tf_metric_update, running_vars_initializer)


def train(discriminator, config, train_image, train_label, test_image, test_label):
    tf.set_random_seed(int(config['random_seed']))

    batch_size = config['batch_size']    
    epsilon = config['epsilon']    # perturbation error
    class_num = config['class_num']    # number of output classes
    pgd_iter = config['pgd_iter']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']


    my_dim = list(train_image.shape)
    num_training_examples = my_dim[0]
    my_dim[0] = batch_size
    image_placeholder = tf.placeholder(train_image.dtype, my_dim)
    my_dim = list(train_label.shape)
    my_dim[0] = batch_size
    label_placeholder = tf.placeholder(train_label.dtype, my_dim)


    x_real = image_placeholder
    label = label_placeholder

    # Normalize to range [-1,1]
    x_real = 2.*x_real - 1.

    step_size = epsilon*0.25

    x = x_real + tf.random_uniform(x_real.shape, -epsilon, epsilon)
    for i in range(pgd_iter):
        d_out = discriminator(x)
        d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=d_out, labels=tf.one_hot(label,class_num)))

        grad_x, = tf.gradients(d_loss, x)
        x = tf.stop_gradient(x + step_size*tf.sign(grad_x))
        x = tf.clip_by_value(x, x_real - epsilon, x_real + epsilon)
        x = tf.clip_by_value(x, -1.0, 1.0)

    d_out_adv = discriminator(x)

    d_loss_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out_adv, labels=tf.one_hot(label,class_num)))

    d_loss = d_loss_adv

    d_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # Weight decay: assume weights are named 'kernel' or 'weights' 
    d_decay = weight_decay * 0.5 * sum(
        tf.reduce_sum(tf.square(v)) for v in d_vars if (v.name.find('kernel')>0 or v.name.find('weights')>0)
    )

    # SGD optimizer with different step sizes
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    optimizer2 = tf.train.MomentumOptimizer(learning_rate=learning_rate*0.1, momentum=0.9)

    d_grads = tf.gradients(d_loss + d_decay, d_vars)

    train_op = optimizer.apply_gradients(zip(d_grads,d_vars))
    train_op2 = optimizer2.apply_gradients(zip(d_grads,d_vars))


    # Gradient norm to evaluate convergence
    d_reg = 0.5 * sum(
        tf.reduce_sum(tf.square(g)) for g in d_grads
    )


    # build test
    my_dim = list(test_image.shape)
    num_test_examples = my_dim[0]
    my_dim[0] = batch_size
    test_image_placeholder = tf.placeholder(test_image.dtype, my_dim)
    my_dim = list(test_label.shape)
    my_dim[0] = batch_size
    test_label_placeholder = tf.placeholder(test_label.dtype, my_dim)

    acc, acc_update, acc_init = build_test(discriminator, test_image_placeholder, test_label_placeholder, config)
    acc_fgs, acc_update_fgs, acc_init_fgs = build_test_fgs(discriminator, test_image_placeholder, 
                                            test_label_placeholder, config)
    acc_pgd, acc_update_pgd, acc_init_pgd = build_test_pgd(discriminator, test_image_placeholder, 
                                            test_label_placeholder, config)


    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        num_steps_per_epoch = int(num_training_examples/batch_size)+1

        idx_list = list(range(num_training_examples))
        random.shuffle(idx_list)

        for batch_idx in range(config['nsteps']):
            if len(idx_list)<batch_size:
                rand_list = list(range(num_training_examples))
                random.shuffle(rand_list)
                idx_list += rand_list

            minibatch_idx = idx_list[:batch_size]
            idx_list = idx_list[batch_size:]

            image_minibatch = train_image[minibatch_idx,:,:,:]
            label_minibatch = train_label[minibatch_idx]

            if batch_idx<100000:
                d_loss_out, d_reg_out, _ = sess.run([d_loss, d_reg, train_op], 
                    feed_dict={image_placeholder: image_minibatch, label_placeholder: label_minibatch})
            else:
                d_loss_out, d_reg_out, _ = sess.run([d_loss, d_reg, train_op2], 
                    feed_dict={image_placeholder: image_minibatch, label_placeholder: label_minibatch})

            if batch_idx % num_steps_per_epoch==0:
                test_acc = run_test(acc, acc_update, acc_init, sess, config, test_image, test_label, 
                    test_image_placeholder, test_label_placeholder)
                test_acc_fgs = run_test(acc_fgs, acc_update_fgs, acc_init_fgs, sess, config, test_image, test_label, 
                    test_image_placeholder, test_label_placeholder)
                test_acc_pgd = run_test(acc_pgd, acc_update_pgd, acc_init_pgd, sess, config, test_image, test_label, 
                    test_image_placeholder, test_label_placeholder)

                print('i=%d, Loss_d: %4.4f, test_acc: %.4f, fgs_acc: %.4f pgd_acc: %.4f d_reg: %.4f'
                % (batch_idx, d_loss_out, test_acc, test_acc_fgs, test_acc_pgd, d_reg_out))

        model_filename = config['model_file']
        saver.save(sess, model_filename)


