import random

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            # args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size, state_is_tuple=False)

        cell = tf.nn.rnn_cell.MultiRNNCell(
            [cell] * args.num_layers,
            state_is_tuple=False
        )

        if infer is False and args.keep_prob < 1:  # training mode
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=args.keep_prob)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, 3 * 25], name='data_in')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, 3 * 25], name='targets')
        zero_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        self.state_in = tf.identity(zero_state, name='state_in')

        self.num_mixture = args.num_mixture
        nout = 1 + self.num_mixture * 228  # end_of_stroke + prob + 2*(mu + sig) + corr
        print(nout)

        with tf.variable_scope('rnnlm'):
            output_w = tf.get_variable("output_w", [args.rnn_size, nout])
            output_b = tf.get_variable("output_b", [nout])

        inputs = tf.split(1, args.seq_length, value=self.input_data)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        # inputs = tf.unpack(self.input_data, axis=1)

        outputs, state_out = tf.nn.seq2seq.rnn_decoder(inputs, self.state_in, cell, loop_function=None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, values=outputs), [-1, args.rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        self.state_out = tf.identity(state_out, name='state_out')

        # reshape target data so that it is compatible with prediction shape
        x1_data = tf.reshape(self.target_data, [-1, 75])

        # x1_data = tf.split(1, 3 * 25, value=flat_target_data)

        # long method:
        # flat_target_data = tf.split(1, args.seq_length, self.target_data)
        # flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]
        # flat_target_data = tf.reshape(tf.concat(1, flat_target_data), [-1, 3])

        # def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
        #     # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
        #     norm1 = tf.subtract(x1, mu1)
        #     norm2 = tf.subtract(x2, mu2)
        #     s1s2 = tf.multiply(s1, s2)
        #     z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - 2 * tf.div(
        #         tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
        #     negRho = 1 - tf.square(rho)
        #     result = tf.exp(tf.div(-z, 2 * negRho))
        #     denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))
        #     result = tf.div(result, denom)
        #     return result

        def tf_nd_normal(x, mu, C):
            x = tf.reshape(x, [4500, 75, 1])
            mu = tf.reshape(mu, [4500, 75, 1])

            # eq wikipedia multivariate normal distribution
            norm = tf.subtract(x, mu)
            print(norm)
            cov = tf.reshape(C, (-1, 75, 75))
            print(cov)
            z = -0.5 * tf.transpose(norm, perm=[0, 2, 1])
            print('esto es la z ')
            print(tf.matrix_inverse(cov))
            z1 = tf.matmul(z, tf.matrix_inverse(cov))
            print(z1)
            z2 = tf.matmul(z1, norm)
            print('shape')
            print(cov.get_shape())

            result = tf.exp(z2)
            # denom = []
            # for i in range(cov.get_shape()[0]):
            #     print(cov[i])
            #     print(i)
            #     denom.append(tf.sqrt(tf.matrix_determinant(2 * np.pi * cov[i])))
            denom=tf.sqrt(tf.matrix_determinant(2 * np.pi * cov))

            print('det')
            print(2 * np.pi * cov)
            print(2 * np.pi * C)
            denom = tf.reshape(denom, [4500, 1, 1])
            result = result / denom
            print('result: ')
            print(result)
            return result

        # def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data):
        #     result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
        #     # implementing eq # 26 of http://arxiv.org/abs/1308.0850
        #     epsilon = 1e-20
        #     result1 = tf.multiply(result0, z_pi)
        #     result1 = tf.reduce_sum(result1, 1, keep_dims=True)
        #     result1 = -tf.log(tf.maximum(result1, 1e-20))  # at the beginning, some errors are exactly zero.
        #
        #     result2 = tf.multiply(z_eos, eos_data) + tf.multiply(1 - z_eos, 1 - eos_data)
        #     result2 = -tf.log(result2)
        #
        #     result = result1 + result2
        #     return tf.reduce_sum(result)

        def get_lossfunc(z_pi, m, C, x):
            result0 = tf_nd_normal(x, m, C)
            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            result1 = tf.multiply(result0, z_pi)
            result1 = tf.reduce_sum(result1, 1, keep_dims=True)
            result1 = -tf.log(tf.maximum(result1, 1e-20))  # at the beginning, some errors are exactly zero.

            result = result1
            return tf.reduce_sum(result)

        # below is where we need to do MDN splitting of distribution params
        def get_mixture_coef(output):
            lista = tf.split(1, 5701, value=output)
            z_pi = lista[0]
            # z_pi = tf.reshape(z_pi, (-1,4500))
            z_mu = lista[1:76]
            z_mu = tf.reshape(z_mu, (-1, 75))
            z_cov = lista[76:]
            z_cov = tf.reshape(z_cov, (-1, 75 * 75))

            # softmax all the pi's:
            max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
            z_pi = tf.subtract(z_pi, max_pi)
            z_pi = tf.exp(z_pi)
            normalize_pi = tf.reciprocal(tf.reduce_sum(z_pi, 1, keep_dims=True))
            z_pi = tf.multiply(normalize_pi, z_pi)

            # # exponentiate the covariance to make cov positive-definite matrix
            # z_sigma1 = tf.exp(z_sigma1)
            # z_sigma2 = tf.exp(z_sigma2)
            z_cov = tf.exp(z_cov)

            return [z_pi, z_mu, z_cov]  # , z_eos]

        # [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos] = get_mixture_coef(output)
        [o_pi, o_mu, o_cov] = get_mixture_coef(output)

        # I could put all of these in a single tensor for reading out, but this is more human readable
        data_out_pi = tf.identity(o_pi, "data_out_pi");
        data_out_mu1 = tf.identity(o_mu, "data_out_mu");
        # data_out_mu2 = tf.identity(o_mu2, "data_out_mu2");
        # data_out_sigma1 = tf.identity(o_sigma1, "data_out_sigma1");
        # data_out_sigma2 = tf.identity(o_sigma2, "data_out_sigma2");
        data_out_cov = tf.identity(o_cov, "data_out_cov");
        # data_out_eos = tf.identity(o_eos, "data_out_eos");

        # sticking them all (except eos) in one op anyway, makes it easier for freezing the graph later
        # IMPORTANT, this needs to stack the named ops above (data_out_XXX), not the prev ops (o_XXX)
        # otherwise when I freeze the graph up to this point, the named versions will be cut
        # eos is diff size to others, so excluding that
        # data_out_mdn = tf.identity(
        #     [data_out_pi, data_out_mu1, data_out_mu2, data_out_sigma1, data_out_sigma2, data_out_corr],
        #     name="data_out_mdn")
        print('foo')
        # data_out_mdn = tf.identity([data_out_pi, data_out_mu1, data_out_cov], name="data_out_mdn")
        print('bar')

        # self.pi = o_pi
        # self.mu1 = o_mu1
        # self.mu2 = o_mu2
        # self.sigma1 = o_sigma1
        # self.sigma2 = o_sigma2
        # self.corr = o_corr
        # self.eos = o_eos
        self.pi = o_pi
        self.mu = o_mu
        self.cov = o_cov

        # lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_eos, x1_data, x2_data, eos_data)
        lossfunc = get_lossfunc(o_pi, o_mu, o_cov, x1_data)
        self.cost = lossfunc / (args.batch_size * args.seq_length)

        self.train_loss_summary = tf.summary.scalar('train_loss', self.cost)
        self.valid_loss_summary = tf.summary.scalar('validation_loss', self.cost)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess):

        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print('error with sampling ensemble')
            return -1

        def sample_gaussian_nd(mu, cov):
            # mean = [mu1, mu2]
            # cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
            x = np.random.multivariate_normal(mu, cov, 75)
            return x

        # todo: mirar esto
        # prev_x = np.zeros((1, 1, 3*25), dtype=np.float32)
        prev_x = np.random.rand(1, 1, 3 * 25)
        # prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))

        dance = np.zeros((self.args.seq_length, 3 * 25), dtype=np.float32)
        mixture_params = []

        for i in range(self.args.seq_length):
            feed = {self.input_data: prev_x, self.state_in: prev_state}

            [o_pi, o_mu, o_cov, next_state] = sess.run(
                [self.pi, self.mu, self.cov, self.state_out], feed)

            idx = get_pi_idx(random.random(), o_pi[0])

            next_x1 = sample_gaussian_nd(o_mu[idx], o_cov[idx])

            dance[i, :] = next_x1

            params = [o_pi[0], o_mu[0], o_cov[0]]
            mixture_params.append(params)

            prev_x = np.zeros((1, 1, 3 * 25), dtype=np.float32)
            prev_x[0, 0:] = np.array([next_x1], dtype=np.float32)
            prev_state = next_state

        return dance, mixture_params
