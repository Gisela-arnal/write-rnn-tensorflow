import argparse
import os
import pickle
import time

import tensorflow as tf

from model import Model
from utils import DataLoader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='data',
                        help='path of the data file')
    parser.add_argument('--data_files', type=str, default='data_files.list',
                        help='path of the data file')
    parser.add_argument('--save_h5', dest='save_h5', action='store_true',
                        help='save parameters to h5 file')
    parser.add_argument('--dataset_file', type=str, default='output_body.h5',
                        help='path of the data file')
    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=3 * 30,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--model_dir', type=str, default='save',
                        help='directory to save model to')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    parser.add_argument('--num_mixture', type=int, default=25,
                        help='number of gaussian mixtures')
    parser.add_argument('--data_scale', type=float, default=20,
                        help='factor to scale raw data down by')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.set_defaults(save_h5=False)
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = DataLoader(args.dataset_path, args.data_files, args.dataset_file, args.batch_size, args.seq_length,
                             args.data_scale, save_h5=args.save_h5)

    if args.model_dir != '' and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    model = Model(args)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'log'), sess.graph)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # data_loader.reset_batch_pointer()
            v_x, v_y = data_loader.validation_data()
            valid_feed = {model.input_data: v_x, model.target_data: v_y, model.state_in: model.state_in.eval()}
            state = model.state_in.eval()
            batch_generator = data_loader.next_batch()
            for b in range(data_loader.num_batches):
                i = e * data_loader.num_batches + b
                start = time.time()
                x, y = next(batch_generator)
                feed = {model.input_data: x, model.target_data: y, model.state_in: state}
                train_loss_summary, train_loss, state, _ = sess.run(
                    [model.train_loss_summary, model.cost, model.state_out, model.train_op], feed)
                summary_writer.add_summary(train_loss_summary, i)

                valid_loss_summary, valid_loss, = sess.run([model.valid_loss_summary, model.cost], valid_feed)
                summary_writer.add_summary(valid_loss_summary, i)

                end = time.time()
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, valid_loss = {:.3f}, time/batch = {:.3f}".format(
                        i,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        train_loss, valid_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    main()
