# Form Correction
import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse
import Pose
import Stance

from tensorflow.python.client import timeline

from common import estimate_pose, CocoPairsRender, read_imgfile, CocoColors, draw_humans
from networks import get_network
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
import numpy as np
from tensorpack import imgaug
from tensorpack.dataflow.common import MapDataComponent, MapData
from tensorpack.dataflow.image import AugmentImageComponent

from common import CocoPart
from pose_augment import *
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

#from pose_augment import set_network_input_wh
    #set_network_input_wh(368, 368)

show_sample = True
db = CocoPoseLMDB('/data/public/rw/coco-pose-estimation-lmdb/', is_train=True, decode_img=show_sample)
db.reset_state()

# list of body part coords for each human in frame (assume one)
RAnkle = []
RKnee = []
RHip = []
LAnkle = []
LKnee = []
LHip = []
Neck = []
for idx, metas in enumerate(db.get_data()):
    meta = metas[0]
    if len(meta.joint_list) <= 0:
        continue
    body = meta.joint_list[0]
    RAnkle.append(body.CocoPart.RAnkle.value)
    RKnee.append(body.CocoPart.RKnee.value)
    RHip.append(body.CocoPart.RHip.value)
    LKnee.append(body.Cocopart.LKnee.value)
    LAnkle.append(body.CocoPart.LAnkle.value)
    LHip.append(body.CocoPart.LHip.value)
    Neck.append(body.Cocoaprt.Neck.value)

# initialize Pose object for the runner in frame
pose = Pose([RAnkle[1], RKnee[1], RHip[1], LAnkle[1], LKnee[1], LHip[1], Neck[1]])

# templates for pose comparison
block_start = Stance(180, 180, 45, 45)
toe_off = Stance(195, 170, 90, 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/p2.jpg')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='mobilenet', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess, trainable=False)

        logging.debug('read image+')
        image = read_imgfile(args.imgpath, args.input_width, args.input_height)
        vec = sess.run(net.get_output(name='concat_stage7'), feed_dict={'image:0': [image]})

        a = time.time()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        pafMat, heatMat = sess.run(
            [
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
            ], feed_dict={'image:0': [image]}, options=run_options, run_metadata=run_metadata
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        heatMat, pafMat = heatMat[0], pafMat[0]

        logging.debug('inference+')

        avg = 0
        for _ in range(10):
            a = time.time()
            sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [image]}
            )
            logging.info('inference- elapsed_time={}'.format(time.time() - a))
            avg += time.time() - a
        logging.info('prediction avg= %f' % (avg / 10))

        logging.info('pose+')
        a = time.time()
        humans = estimate_pose(heatMat, pafMat)
        logging.info('pose- elapsed_time={}'.format(time.time() - a))

        logging.info('image={} heatMap={} pafMat={}'.format(image.shape, heatMat.shape, pafMat.shape))
        process_img = CocoPoseLMDB.display_image(image, heatMat, pafMat, as_numpy=True)

        # display estimated pose over the original image
        image = cv2.imread(args.imgpath)
        image_h, image_w = image.shape[:2]
        image = draw_humans(image, humans)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

        convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
        convas[:, :640] = process_img
        convas[:, 640:] = image

        cv2.imshow('result', convas)
        cv2.waitKey(0)

        tf.train.write_graph(sess.graph_def, '.', 'graph-tmp.pb', as_text=True)

        # Set what the current stance of the runner is
        #curr_stance = block_start
        curr_stance = toe_off

        # determine the error of each joint angle for this current stance
        errors = {}
        errors["Plant Side Hip"].append(curr_stance.get_ideal_PHip_angle() - pose.get.PHip_angle())
        errors["Plant Side Knee"].append(curr_stance.get_ideal_PKnee_angle() - pose.get.PKnee_angle())
        errors["Swing Side Hip"].append(curr_stance.get_ideal_SHip_angle() - pose.get.SHip_angle())
        errors["Swing Side Knee"].append(curr_stance.get_ideal_SKnee_angle() - pose.get.SKnee_angle())

        # sort them in descending order
        errors = sorted(errors.items(), key=lambda x:x[1], reverse=True)
        
        sig_errors = {}
        sig_lvl = 5 # how much deviation from the ideal form to tolerate

        # determine which joints need to be corrected
        for joint, error in errors:
            if abs(error) > sig_lvl:
                sig_errors[joint].append(error)

        # determine direction of correction
        for joint, error in sig_errors:
            # if error is negative
            if error < 0:
                # on the swing side
                if joint == "Swing Side Hip":
                    print("Drive your swing leg higher to be paralell with the ground. (Think high knees)")
                if joint == "Swing Side Knee":
                    print("Recover your swing ankle over your plant knee. (Think butt kicks)")
                if joint == "Plant Side Hip" or joint == "Plant Side Knee":
                    print("Don't extend too much through your", joint,". You're spending too much time in contact with the ground.")
            # if error is positive, joint needs to be extended more fully
            if error > 0:
                print("Try to extend more through your", joint,".")
            



