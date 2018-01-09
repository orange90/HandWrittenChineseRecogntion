import os
import numpy as np
import struct
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
from logging import Logger
logger = Logger()

data_dir = '../data1'



tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 12002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 10, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 2000, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', '../data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', '../data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "test"}')
FLAGS = tf.app.flags.FLAGS

def pre_process(images):
    # if FLAGS.random_flip_up_down:
    #     images = tf.image.random_flip_up_down(images)
    # if FLAGS.random_flip_left_right:
    #     images = tf.image.random_flip_left_right(images)
    # if FLAGS.random_brightness:
    #     images = tf.image.random_brightness(images, max_delta=0.3)
    # if FLAGS.random_contrast:
    #     images = tf.image.random_contrast(images, 0.8, 1.2)
    new_size = tf.constant([64,64], dtype=tf.int32)
    images = tf.image.resize_images(images, new_size)
    return images


def batch_data(file_labels,sess, batch_size=128):
    image_list = [file_label[0] for file_label in file_labels]
    label_list = [int(file_label[1]) for file_label in file_labels]
    print 'tag2 {0}'.format(len(image_list))
    images_tensor = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels_tensor = tf.convert_to_tensor(label_list, dtype=tf.int64)
    input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor])

    labels = input_queue[1]
    images_content = tf.read_file(input_queue[0])
    # images = tf.image.decode_png(images_content, channels=1)
    images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
    # images = images / 256
    images =  pre_process(images)
    # print images.get_shape()
    # one hot
    labels = tf.one_hot(labels, 3755)
    image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,min_after_dequeue=10000)
    # print 'image_batch', image_batch.get_shape()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return image_batch, label_batch, coord, threads


def network(images, labels=None):
    endpoints = {}
    conv_1 = slim.conv2d(images, 32, [3,3],1, padding='SAME')
    max_pool_1 = slim.max_pool2d(conv_1, [2,2],[2,2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 64, [3,3],padding='SAME')
    max_pool_2 = slim.max_pool2d(conv_2, [2,2],[2,2], padding='SAME')
    flatten = slim.flatten(max_pool_2)
    out = slim.fully_connected(flatten,3755, activation_fn=None)
    global_step = tf.Variable(initial_value=0)
    if labels is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, labels))
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
    output_score = tf.nn.softmax(out)
    predict_val_top3, predict_index_top3 = tf.nn.top_k(output_score, k=3)

    endpoints['global_step'] = global_step
    if labels is not None:
        endpoints['labels'] = labels
        endpoints['train_op'] = train_op
        endpoints['loss'] = loss
        endpoints['accuracy'] = accuracy
        endpoints['merged_summary_op'] = merged_summary_op
    endpoints['output_score'] = output_score
    endpoints['predict_val_top3'] = predict_val_top3
    endpoints['predict_index_top3'] = predict_index_top3
    return endpoints


def train():
    sess = tf.Session()
    file_labels = get_imagesfile(FLAGS.train_data_dir)
    images, labels, coord, threads = batch_data(file_labels, sess)
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.train.SummaryWriter('./log' + '/train',sess.graph)
    test_writer = tf.train.SummaryWriter('./log' + '/val')
    start_step = 0
    if FLAGS.restore:
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print "restore from the checkpoint {0}".format(ckpt)
            start_step += int(ckpt.split('-')[-1])
    logger.info(':::Training Start:::')
    try:
        while not coord.should_stop():
        # logger.info('step {0} start'.format(i))
            start_time = time.time()
            _, loss_val, train_summary, step = sess.run([endpoints['train_op'], endpoints['loss'], endpoints['merged_summary_op'], endpoints['global_step']])
            train_writer.add_summary(train_summary, step)
            end_time = time.time()
            logger.info("the step {0} takes {1} loss {2}".format(step, end_time-start_time, loss_val))
            if step > FLAGS.max_steps:
                break
            # logger.info("the step {0} takes {1} loss {2}".format(i, end_time-start_time, loss_val))
            if step % FLAGS.eval_steps == 1:
                accuracy_val,test_summary, step = sess.run([endpoints['accuracy'], endpoints['merged_summary_op'], endpoints['global_step']])
                test_writer.add_summary(test_summary, step)
                logger.info('===============Eval a batch in Train data=======================')
                logger.info( 'the step {0} accuracy {1}'.format(step, accuracy_val))
                logger.info('===============Eval a batch in Train data=======================')
            if step % FLAGS.save_steps == 1:
                logger.info('Save the ckpt of {0}'.format(step))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    except tf.errors.OutOfRangeError:
        # print "============train finished========="
        logger.info('==================Train Finished================')
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=endpoints['global_step'])
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


def validation():
    # it should be fixed by using placeholder with epoch num in train stage
    sess = tf.Session()

    file_labels = get_imagesfile(FLAGS.test_data_dir)
    test_size = len(file_labels)
    print test_size
    val_batch_size = FLAGS.val_batch_size
    test_steps = test_size / val_batch_size
    print test_steps
    # images, labels, coord, threads= batch_data(file_labels, sess)
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    labels = tf.placeholder(dtype=tf.int32, shape=[None, 3755])
    # read batch images from file_labels
    # images_batch = np.zeros([128,64,64,1])
    # labels_batch = np.zeros([128,3755])
    # labels_batch[0][20] = 1
    #
    endpoints = network(images, labels)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        # logger.info("restore from the checkpoint {0}".format(ckpt))
    # logger.info('Start validation')
    final_predict_val = []
    final_predict_index = []
    groundtruth = []
    for i in range(test_steps):
        start = i * val_batch_size
        end = (i + 1) * val_batch_size
        images_batch = []
        labels_batch = []
        labels_max_batch = []
        logger.info('=======start validation on {0}/{1} batch========='.format(i, test_steps))
        for j in range(start, end):
            image_path = file_labels[j][0]
            temp_image = Image.open(image_path).convert('L')
            temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
            temp_label = np.zeros([3755])
            label = int(file_labels[j][1])
            # print label
            temp_label[label] = 1
            # print "====",np.asarray(temp_image).shape
            labels_batch.append(temp_label)
            # print "====",np.asarray(temp_image).shape
            images_batch.append(np.asarray(temp_image) / 255.0)
            labels_max_batch.append(label)
        # print images_batch
        images_batch = np.array(images_batch).reshape([-1, 64, 64, 1])
        labels_batch = np.array(labels_batch)
        batch_predict_val, batch_predict_index = sess.run([endpoints['predict_val_top3'],
                                                           endpoints['predict_index_top3']],
                                                          feed_dict={images: images_batch, labels: labels_batch})
        logger.info('=======validation on {0}/{1} batch end========='.format(i, test_steps))
        final_predict_val += batch_predict_val.tolist()
        final_predict_index += batch_predict_index.tolist()
        groundtruth += labels_max_batch
    sess.close()
    return final_predict_val, final_predict_index, groundtruth


def inference(image):
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size),Image.ANTIALIAS)
    sess = tf.Session()
    logger.info('========start inference============')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    endpoints = network(images)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    predict_val, predict_index = sess.run([endpoints['predict_val_top3'],endpoints['predict_index_top3']], feed_dict={images:temp_image})
    sess.close()
    return final_predict_val, final_predict_index