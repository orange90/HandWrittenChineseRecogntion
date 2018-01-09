from PIL import Image
IMAGE_SIZE = 64
import tensorflow as tf
def inference(image):
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
    sess = tf.Session()
    print('========start inference============')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    endpoints = network(images)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    predict_val, predict_index = sess.run([endpoints['predict_val_top3'],endpoints['predict_index_top3']], feed_dict={images:temp_image})
    sess.close()
    return final_predict_val, final_predict_index


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.arg_max(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result