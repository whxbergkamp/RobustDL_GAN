import tensorflow as tf
import time

# construct graph for testing on original examples
def build_test(discriminator, test_data, config):
    x = test_data[0]
    x = 2.0*x - 1.0
    y = test_data[1]
    d_out = discriminator(x)
    predictions = tf.argmax(d_out, 1)

    # http://ronny.rest/blog/post_2017_09_11_tf_metrics/
    tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name='test_metric')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="test_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    return (tf_metric, tf_metric_update, running_vars_initializer)

# run actual test to collect performance statistics
def run_test(tf_metric, tf_metric_update, running_vars_initializer, sess, config):
    sess.run(running_vars_initializer)

    batch_size = config['batch_size']
    test_size = config['test_size']
    num_steps = int(test_size/batch_size)
    for i in range(num_steps):
        sess.run(tf_metric_update)

    accuracy = sess.run(tf_metric)

    return accuracy

# construct graph for testing on examples perturbed with FGS
def build_test_fgs(discriminator, test_data, config):
    x = test_data[0]
    x = 2.0*x - 1.0
    y = test_data[1]
    epsilon = config['epsilon']
    class_num = config['class_num']

    d_out = discriminator(x)
    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out, labels=tf.one_hot(y,class_num)))       #use parameter replace 10

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
def build_test_pgd(discriminator, test_data, config):
    x0 = test_data[0]

    x0 = 2.0*x0 - 1.0
    y = test_data[1]
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



def train(discriminator, data, test_data, config):
    tf.set_random_seed(int(config['random_seed']))

    batch_size = config['batch_size']
    epsilon = config['epsilon']    # size of perturbation
    class_num = config['class_num']    # number of output classes
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    train_size = config['train_size']

    x_real = data[0]
    label = data[1]

    # Data augmentation for CIFAR10
    x_real = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 32+4, 32+4), x_real)
    x_real = tf.map_fn(lambda x: tf.random_crop(x, [32,32,3]), x_real)
    x_real = tf.map_fn(lambda x: tf.image.random_flip_left_right(x), x_real)
 
    # Normalize to range [-1,1]
    x_real = 2.*x_real - 1.

    d_out_real = discriminator(x_real)
    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out_real, labels=tf.one_hot(label,class_num)))

    d_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # Weight decay: assume weights are named 'kernel' or 'weights' 
    d_decay = weight_decay * 0.5 * sum(
        tf.reduce_sum(tf.square(v)) for v in d_vars if (v.name.find('kernel')>0 or v.name.find('weights')>0)
    )

    # SGD optimizer with different step sizes
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    optimizer2 = tf.train.MomentumOptimizer(learning_rate=learning_rate*0.1, momentum=0.9)
    optimizer3 = tf.train.MomentumOptimizer(learning_rate=learning_rate*0.01, momentum=0.9)

    d_grads = tf.gradients(d_loss + d_decay, d_vars)

    train_op = optimizer.apply_gradients(zip(d_grads,d_vars))
    train_op2 = optimizer2.apply_gradients(zip(d_grads,d_vars))
    train_op3 = optimizer3.apply_gradients(zip(d_grads,d_vars))


    # Gradient norm to evaluate convergence
    d_reg = 0.5 * sum(
        tf.reduce_sum(tf.square(g)) for g in d_grads
    )


    # build test
    acc, acc_update, acc_init = build_test(discriminator, test_data, config)
    acc_fgs, acc_update_fgs, acc_init_fgs = build_test_fgs(discriminator, test_data, config)
    acc_pgd, acc_update_pgd, acc_init_pgd = build_test_pgd(discriminator, test_data, config)


    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

    # Supervisor
    sv = tf.train.Supervisor(
        logdir=config['log_dir'], global_step=global_step,
        summary_op=None, 
    )


    with sv.managed_session() as sess:
        num_steps_per_epoch = int(train_size/batch_size)+1

        for batch_idx in range(config['nsteps']):
            if sv.should_stop():
               break

            if batch_idx<100000:
                d_loss_out, d_reg_out, _ = sess.run([d_loss, d_reg, train_op])
            elif batch_idx<150000:
                d_loss_out, d_reg_out, _ = sess.run([d_loss, d_reg, train_op2])
            else:
                d_loss_out, d_reg_out, _ = sess.run([d_loss, d_reg, train_op3])


            if batch_idx % num_steps_per_epoch==0:
                test_acc = run_test(acc, acc_update, acc_init, sess, config)
                test_acc_fgs = run_test(acc_fgs, acc_update_fgs, acc_init_fgs, sess, config)
                test_acc_pgd = run_test(acc_pgd, acc_update_pgd, acc_init_pgd, sess, config)

                print('i=%d, Loss_d: %4.4f, test_acc: %.4f, fgs_acc: %.4f pgd_acc: %.4f d_reg: %.4f'
                % (batch_idx, d_loss_out, test_acc, test_acc_fgs, test_acc_pgd, d_reg_out))

            sess.run(global_step_op)

        model_filename = config['model_file']
        saver.save(sess, model_filename)



