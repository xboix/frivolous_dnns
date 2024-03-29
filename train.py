import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import os.path
import shutil
import sys
import numpy as np
import tensorflow as tf
import experiments
from models import nets
from utils import summary

################################################################################################
# Read experiment to run
################################################################################################

ID = int(sys.argv[1:][0])

opt = experiments.opt[ID]

tf.compat.v1.reset_default_graph()
tf.compat.v1.set_random_seed(opt.seed)

# Skip execution if instructed in experiment
if opt.skip:
    print("SKIP")
    quit()

print(opt.name)
################################################################################################

################################################################################################
# Define training and validation datasets through Dataset API
################################################################################################

# Initialize dataset and creates TF records if they do not exist
if opt.dataset.dataset_name == 'cifar':
    from data import cifar_dataset
    dataset = cifar_dataset.Cifar10(opt)
elif opt.dataset.dataset_name == 'rand10':
    from data import rand10_dataset
    dataset = rand10_dataset.Rand10(opt)
elif opt.dataset.dataset_name == 'rand10000':
    from data import rand10000_dataset
    dataset = rand10000_dataset.Rand10000(opt)
elif opt.dataset.dataset_name == 'rand10_regression':
    from data import rand10_regression_dataset
    dataset = rand10_regression_dataset.Rand10_regression(opt)
elif opt.dataset.dataset_name == 'rand10000_regression':
    from data import rand10000_regression_dataset
    dataset = rand10000_regression_dataset.Rand10000_regression(opt)

# Repeatable datasets for training
train_dataset = dataset.create_dataset(augmentation=opt.hyper.augmentation, standarization=True, set_name='train',
                                       repeat=True)
val_dataset = dataset.create_dataset(augmentation=False, standarization=True, set_name='val', repeat=True)

# No repeatable dataset for testing
train_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='train', repeat=False)
val_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='val', repeat=False)
test_dataset_full = dataset.create_dataset(augmentation=False, standarization=True, set_name='test', repeat=False)

# Hadles to switch datasets
handle = tf.compat.v1.placeholder(tf.string, shape=[])
iterator = tf.compat.v1.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                         train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()
val_iterator = val_dataset.make_initializable_iterator()

train_iterator_full = train_dataset_full.make_initializable_iterator()
val_iterator_full = val_dataset_full.make_initializable_iterator()
test_iterator_full = test_dataset_full.make_initializable_iterator()

################################################################################################


################################################################################################
# Declare DNN
################################################################################################

# Get data from dataset dataset
image, y_ = iterator.get_next()

if opt.dataset.dataset_name == 'cifar':
    image = tf.image.resize_images(image, [opt.hyper.image_size, opt.hyper.image_size])
    if opt.extense_summary:
        tf.summary.image('input', image)
elif opt.dataset.dataset_name == 'rand10' or opt.dataset.dataset_name == 'rand10_regression':
    image = tf.compat.v1.reshape(image, [-1, 10])
elif opt.dataset.dataset_name == 'rand10000' or opt.dataset.dataset_name == 'rand10000_regression':
    image = tf.compat.v1.reshape(image, [-1, 10000])

# Call DNN
dropout_rate = tf.compat.v1.placeholder(tf.float32)
to_call = getattr(nets, opt.dnn.name)
y, parameters, _ = to_call(image, dropout_rate, opt, dataset.list_labels)

y = tf.cast(y, tf.float32)

# Loss function
with tf.name_scope('loss'):

    weights_norm = tf.reduce_sum(
        input_tensor=opt.hyper.weight_decay * tf.stack(
            [tf.nn.l2_loss(i) for i in parameters]
        ),
        name='weights_norm')
    tf.summary.scalar('weight_decay', weights_norm)

    if not opt.hyper.mse:
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
        total_loss = weights_norm + cross_entropy
        tf.summary.scalar('cross_entropy', cross_entropy)
    else:
        mse = tf.reduce_mean((y_ - y) ** 2)
        total_loss = weights_norm + mse
        tf.summary.scalar('mse', mse)

    tf.summary.scalar('total_loss', total_loss)

global_step = tf.Variable(0, name='global_step', trainable=False)

################################################################################################
# Set up Training
################################################################################################

# Learning rate
num_batches_per_epoch = dataset.num_images_epoch / opt.hyper.batch_size
decay_steps = int(opt.hyper.num_epochs_per_decay)
lr = tf.train.exponential_decay(opt.hyper.learning_rate, global_step, decay_steps,
                                opt.hyper.learning_rate_factor_per_decay, staircase=True)

tf.summary.scalar('learning_rate', lr)
tf.summary.scalar('weight_decay', opt.hyper.weight_decay)


if opt.hyper.mse:
    with tf.name_scope('mean_sq_err'):
        mean_sq_err = tf.reduce_mean((y_ - y) ** 2)
        tf.summary.scalar('mean_sq_err', mean_sq_err)
else:
    # Accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.summary.scalar('accuracy', accuracy)

################################################################################################

# allow for GPU memory to be allocated as needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    ################################################################################################
    # Set up Gradient Descent
    ################################################################################################
    all_var = tf.trainable_variables()

    # choose optimizer
    train_step = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr,
                                                      momentum=opt.hyper.momentum).minimize(total_loss,
                                                                                            var_list=all_var)
    if opt.optimizer == 1:
        train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss, var_list=all_var)
    elif opt.optimizer == 2:
        train_step = tf.train.AdamOptimizer().minimize(total_loss, var_list=all_var)
    elif opt.optimizer == 3:
        train_step = tf.train.AdagradOptimizer(learning_rate=opt.hyper.learning_rate/10).minimize(total_loss,
                                                                                                  var_list=all_var)
    elif opt.optimizer == 4:
        train_step = tf.train.ProximalAdagradOptimizer(learning_rate=opt.hyper.learning_rate/10). \
            minimize(total_loss, var_list=all_var)
    elif opt.optimizer == 5:
        train_step = tf.train.RMSPropOptimizer(learning_rate=opt.hyper.learning_rate/10).minimize(total_loss,
                                                                                                  var_list=all_var)
    elif opt.optimizer == 6:
        train_step = tf.train.FtrlOptimizer(learning_rate=opt.hyper.learning_rate/10).minimize(total_loss,
                                                                                               var_list=all_var)

    inc_global_step = tf.assign_add(global_step, 1, name='increment')

    raw_grads = tf.gradients(total_loss, all_var)
    grads = list(zip(raw_grads, tf.trainable_variables()))

    for g, v in grads:
        summary.gradient_summaries(g, v, opt)
    ################################################################################################

    ################################################################################################
    # Set up checkpoints and data
    ################################################################################################

    saver = tf.compat.v1.train.Saver(max_to_keep=opt.max_to_keep_checkpoints)

    # Automatic restore model, or force train from scratch
    flag_testable = False

    # Set up directories and checkpoints
    if not os.path.isfile(opt.log_dir_base + opt.name + '/models/checkpoint'):
        sess.run(tf.global_variables_initializer())

        best_acc_val = 0  # only for training with correctly labeled data
        best_acc_val_epoch = 0  # only for training with correctly labeled data
        best_acc_train = 0  # only for training with randomly labeled data
        best_acc_train_epoch = 0  # only for training with randomly labeled data

        best_mse_val = 0  # only for training with correctly labeled data
        best_mse_val_epoch = 0  # only for training with correctly labeled data
        best_mse_train = 0  # only for training with randomly labeled data
        best_mse_train_epoch = 0  # only for training with randomly labeled data

    elif opt.restart:
        print("RESTART")
        shutil.rmtree(opt.log_dir_base + opt.name + '/models/')
        shutil.rmtree(opt.log_dir_base + opt.name + '/train/')
        shutil.rmtree(opt.log_dir_base + opt.name + '/val/')
        sess.run(tf.global_variables_initializer())

        best_acc_val = 0  # only for training with correctly labeled data
        best_acc_val_epoch = 0  # only for training with correctly labeled data
        best_acc_train = 0  # only for training with randomly labeled data
        best_acc_train_epoch = 0  # only for training with randomly labeled data

        best_mse_val = 0  # only for training with correctly labeled data
        best_mse_val_epoch = 0  # only for training with correctly labeled data
        best_mse_train = 0  # only for training with randomly labeled data
        best_mse_train_epoch = 0  # only for training with randomly labeled data

    else:
        print("RESTORE")
        saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
        flag_testable = True

        # If we're restoring, we have to get the reference training and validation accuracies and epochs
        train_handle_full = sess.run(train_iterator_full.string_handle())
        valid_handle_full = sess.run(val_iterator_full.string_handle())

        if opt.hyper.mse:
            # Run one pass over a batch of the train dataset.
            sess.run(train_iterator_full.initializer)
            mse_tmp = 0.0
            train_iters = int(dataset.num_images_epoch / opt.hyper.batch_size)
            for num_iter in range(train_iters):
                mse_out = sess.run([mean_sq_err], feed_dict={handle: train_handle_full,
                                                             dropout_rate: opt.hyper.drop_test})
                mse_tmp += mse_out[0]
            best_mse_train = mse_tmp / float(train_iters)  # only for training with randomly labeled data

            # Run one pass over a batch of the validation dataset.
            sess.run(val_iterator_full.initializer)
            mse_tmp = 0.0
            val_iters = int(dataset.num_images_val / opt.hyper.batch_size)
            for num_iter in range(val_iters):
                mse_out = sess.run([accuracy], feed_dict={handle: valid_handle_full, dropout_rate: opt.hyper.drop_test})
                mse_tmp += mse_out[0]
            best_mse_val = mse_tmp / float(val_iters)  # only for training with correctly labeled data

            best_mse_val_epoch = sess.run(global_step)  # only for training with correctly labeled data
            best_mse_train_epoch = sess.run(global_step)  # only for training with randomly labeled data

        else:
            # Run one pass over a batch of the train dataset.
            sess.run(train_iterator_full.initializer)
            acc_tmp = 0.0
            train_iters = int(dataset.num_images_epoch / opt.hyper.batch_size)
            for num_iter in range(train_iters):
                acc_out = sess.run([accuracy], feed_dict={handle: train_handle_full, dropout_rate: opt.hyper.drop_test})
                acc_tmp += acc_out[0]
            best_acc_train = acc_tmp / float(train_iters)  # only for training with randomly labeled data

            # Run one pass over a batch of the validation dataset.
            sess.run(val_iterator_full.initializer)
            acc_tmp = 0.0
            val_iters = int(dataset.num_images_val / opt.hyper.batch_size)
            for num_iter in range(val_iters):
                acc_out = sess.run([accuracy], feed_dict={handle: valid_handle_full, dropout_rate: opt.hyper.drop_test})
                acc_tmp += acc_out[0]
            best_acc_val = acc_tmp / float(val_iters)  # only for training with correctly labeled data

            best_acc_val_epoch = sess.run(global_step)  # only for training with correctly labeled data
            best_acc_train_epoch = sess.run(global_step)  # only for training with randomly labeled data

        sys.stdout.flush()

    # Datasets
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(train_iterator.string_handle())
    validation_handle = sess.run(val_iterator.string_handle())
    ################################################################################################

    ################################################################################################
    # RUN TRAIN
    ################################################################################################
    if not opt.test:

        # Prepare summaries
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(opt.log_dir_base + opt.name + '/train', sess.graph)
        val_writer = tf.compat.v1.summary.FileWriter(opt.log_dir_base + opt.name + '/val')

        print("STARTING EPOCH = ", sess.run(global_step))
        ################################################################################################
        # Loop alternating between training and validation.
        ################################################################################################
        counter_stop = 0  # means different things for correctly and randomly labeled training data

        for iEpoch in range(int(sess.run(global_step)), opt.hyper.max_num_epochs):

            stop_train = False

            # Save metadata every epoch
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summ = sess.run([merged], feed_dict={handle: training_handle,
                                                 dropout_rate: opt.hyper.drop_train},
                            options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'epoch%03d' % iEpoch)
            val_writer.add_run_metadata(run_metadata, 'epoch%03d' % iEpoch)

            # Steps for doing one epoch
            for iStep in range(int(dataset.num_images_epoch / opt.hyper.batch_size)):

                # Epoch counter
                k = iStep * opt.hyper.batch_size + dataset.num_images_epoch * iEpoch

                # Print accuray and summaries + train steps
                if iStep == 0:  # !train_step
                    print("* epoch:", int(float(k) / float(dataset.num_images_epoch)))

                    # commented out stuff isn't very useful
                    '''
                    summ, acc_train_tmp = sess.run([merged, accuracy],
                                               feed_dict={handle: training_handle,
                                                          dropout_rate: opt.hyper.drop_train})
                    train_writer.add_summary(summ, k)
                    
                    summ, acc_val = sess.run([merged, accuracy], feed_dict={handle: validation_handle, 
                                                                            dropout_rate: opt.hyper.drop_test})
                    val_writer.add_summary(summ, k)
                    '''

                    train_handle_full = sess.run(train_iterator_full.string_handle())
                    sess.run(train_iterator_full.initializer)

                    if opt.hyper.mse:
                        mse_tmp = 0.0
                        train_iters = int(dataset.num_images_epoch / opt.hyper.batch_size)
                        for num_iter in range(train_iters):
                            summ, mse_out = sess.run([merged, mean_sq_err], feed_dict={handle: train_handle_full,
                                                                                    dropout_rate: opt.hyper.drop_test})
                            mse_tmp += mse_out
                            step_k = num_iter * opt.hyper.batch_size + dataset.num_images_epoch * iEpoch
                            train_writer.add_summary(summ, step_k)
                        mse_train = mse_tmp / float(train_iters)

                        # valid_handle_full = sess.run(val_iterator_full.string_handle())
                        # sess.run(val_iterator_full.initializer)
                        # mse_tmp = 0.0
                        # val_iters = int(dataset.num_images_val / opt.hyper.batch_size)
                        # for num_iter in range(val_iters):
                        #     summ, mse_out = sess.run([merged, mean_sq_err],
                        #     feed_dict={handle: valid_handle_full,
                        #                dropout_rate: opt.hyper.drop_test})
                        #     mse_tmp += mse_out
                        #     step_k = num_iter * opt.hyper.batch_size + dataset.num_images_epoch * iEpoch
                        #     val_writer.add_summary(summ, step_k)
                        # mse_val = mse_tmp / float(val_iters)

                        print("train mse:", mse_train)
                        # print("val mse: ", mse_val)
                        saver.save(sess, opt.log_dir_base + opt.name + '/models/model', global_step=iEpoch)

                    else:
                        acc_tmp = 0.0
                        train_iters = int(dataset.num_images_epoch / opt.hyper.batch_size)
                        for num_iter in range(train_iters):
                            summ, acc_out = sess.run([merged, accuracy], feed_dict={handle: train_handle_full,
                                                                                    dropout_rate: opt.hyper.drop_test})
                            acc_tmp += acc_out
                            step_k = num_iter * opt.hyper.batch_size + dataset.num_images_epoch * iEpoch
                            train_writer.add_summary(summ, step_k)
                        acc_train = acc_tmp / float(train_iters)

                        valid_handle_full = sess.run(val_iterator_full.string_handle())
                        sess.run(val_iterator_full.initializer)
                        acc_tmp = 0.0
                        val_iters = int(dataset.num_images_val / opt.hyper.batch_size)
                        for num_iter in range(val_iters):
                            summ, acc_out = sess.run([merged, accuracy], feed_dict={handle: valid_handle_full,
                                                                                    dropout_rate: opt.hyper.drop_test})
                            acc_tmp += acc_out
                            step_k = num_iter * opt.hyper.batch_size + dataset.num_images_epoch * iEpoch
                            val_writer.add_summary(summ, step_k)
                        acc_val = acc_tmp / float(val_iters)

                        print("train acc:", acc_train)
                        print("val acc: ", acc_val)

                        if opt.dataset.random_labels:  # stopping for randomly labeled data based on training accuracy
                            if acc_train > best_acc_train:
                                saver.save(sess, opt.log_dir_base + opt.name + '/models/model', global_step=iEpoch)
                                best_acc_train = acc_train
                                best_acc_train_epoch = iEpoch
                                counter_stop = 0
                            else:
                                counter_stop += 1
                                if counter_stop >= 25:
                                    stop_train = True
                            if acc_train == 1.0:  # if perfect train accuracy achieved, we can't do better
                                stop_train = True

                        else:  # stopping for correctly labeled data based on best validation acc
                            if acc_val > best_acc_val:
                                saver.save(sess, opt.log_dir_base + opt.name + '/models/model', global_step=iEpoch)
                                best_acc_val = acc_val
                                best_acc_val_epoch = iEpoch
                                counter_stop = 0  # reset
                            else:
                                # if trained another epoch without beating the best val acc
                                counter_stop += 1
                                if counter_stop >= 50:
                                    stop_train = True

                else:
                    sess.run([train_step], feed_dict={handle: training_handle, dropout_rate: opt.hyper.drop_train})

            if stop_train:
                # stop training because stipping criterion met
                if opt.dataset.random_labels:
                    print('model selected with best train acc of', best_acc_train, 'at epoch:', best_acc_train_epoch)
                else:
                    print('model selected with best val acc of', best_acc_val, 'at epoch:', best_acc_val_epoch)
                break

            sess.run([inc_global_step])
            print("----------------")
            sys.stdout.flush()
            ################################################################################################

        flag_testable = True

        train_writer.close()
        val_writer.close()

    ################################################################################################
    # RUN TEST
    ################################################################################################

    if flag_testable:

        if opt.hyper.mse:
            if best_mse_train + best_mse_val > 0:
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))
        else:
            if best_acc_train + best_acc_val > 0:
                saver.restore(sess, tf.train.latest_checkpoint(opt.log_dir_base + opt.name + '/models/'))

        train_handle_full = sess.run(train_iterator_full.string_handle())
        valid_handle_full = sess.run(val_iterator_full.string_handle())
        test_handle_full = sess.run(test_iterator_full.string_handle())

        if opt.hyper.mse:
            # Run one pass over a batch of the train dataset.
            sess.run(train_iterator_full.initializer)
            mse_tmp = 0.0
            train_iters = int(dataset.num_images_epoch / opt.hyper.batch_size)
            for num_iter in range(train_iters):
                mse_out = sess.run([mean_sq_err], feed_dict={handle: train_handle_full,
                                                             dropout_rate: opt.hyper.drop_test})
                mse_tmp += mse_out[0]

            train_mse = mse_tmp / float(train_iters)
            print("Full train mse = " + str(train_mse))

            # Run one pass over a batch of the validation dataset.
            # sess.run(val_iterator_full.initializer)
            #             # mse_tmp = 0.0
            #             # val_iters = int(dataset.num_images_val / opt.hyper.batch_size)
            #             # for num_iter in range(val_iters):
            #             #     mse_out = sess.run([mean_sq_err], feed_dict={handle: valid_handle_full,
            #             #                                                  dropout_rate: opt.hyper.drop_test})
            #             #     mse_tmp += mse_out[0]
            #             #
            #             # val_mse = mse_tmp / float(val_iters)
            #             # print("Full val mse = " + str(val_mse))

            # Run one pass over a batch of the test dataset.
            sess.run(test_iterator_full.initializer)
            mse_tmp = 0.0
            test_iters = int(dataset.num_images_test / opt.hyper.batch_size)
            for num_iter in range(test_iters):
                mse_out = sess.run([mean_sq_err], feed_dict={handle: test_handle_full,
                                                             dropout_rate: opt.hyper.drop_test})
                mse_tmp += mse_out[0]

            test_mse = mse_tmp / float(test_iters)
            print("Full test mse = " + str(test_mse))

        else:
            # Run one pass over a batch of the train dataset.
            sess.run(train_iterator_full.initializer)
            acc_tmp = 0.0
            train_iters = int(dataset.num_images_epoch / opt.hyper.batch_size)
            for num_iter in range(train_iters):
                acc_out = sess.run([accuracy], feed_dict={handle: train_handle_full, dropout_rate: opt.hyper.drop_test})
                acc_tmp += acc_out[0]

            train_acc = acc_tmp / float(train_iters)
            print("Full train acc = " + str(train_acc))

            # Run one pass over a batch of the validation dataset.
            sess.run(val_iterator_full.initializer)
            acc_tmp = 0.0
            val_iters = int(dataset.num_images_val / opt.hyper.batch_size)
            for num_iter in range(val_iters):
                acc_out = sess.run([accuracy], feed_dict={handle: valid_handle_full, dropout_rate: opt.hyper.drop_test})
                acc_tmp += acc_out[0]

            val_acc = acc_tmp / float(val_iters)
            print("Full val acc = " + str(val_acc))

            # Run one pass over a batch of the test dataset.
            sess.run(test_iterator_full.initializer)
            acc_tmp = 0.0
            test_iters = int(dataset.num_images_test / opt.hyper.batch_size)
            for num_iter in range(test_iters):
                acc_out = sess.run([accuracy], feed_dict={handle: test_handle_full, dropout_rate: opt.hyper.drop_test})
                acc_tmp += acc_out[0]

            test_acc = acc_tmp / float(test_iters)
            print("Full test acc = " + str(test_acc))

        sys.stdout.flush()
        print('train.py')
        print(":)")

    else:
        print("MODEL WAS NOT TRAINED")
