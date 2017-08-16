from keras.optimizers import Optimizer
import keras.backend as K


class SGDWithAcc(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        accum_iters: int >= 1. Accumulate gradients over several
            iterations before updating weights.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, accum_iters=1, **kwargs):
        super(SGDWithAcc, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.momentum = K.variable(momentum, name='momentum')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.accum_iters = K.variable(accum_iters)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        # Accumulated gradients
        grads_acc = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + moments
        # 1 if we update the gradient, 0 otherwise
        update = K.cast(K.equal(self.iterations % self.accum_iters, 0), K.floatx())

        for p, g, m, gacc in zip(params, grads, moments, grads_acc):
            # if no update, g_t = 0, just keep accumulating
            g_t = update * ((gacc + g) / self.accum_iters)

            v = self.momentum * m - lr * g_t  # velocity
            self.updates.append(K.update(m, (1 - update) * m + update * v))
            self.updates.append((gacc, (1 - update) * (gacc + g)))

            if self.nesterov:
                new_p = p + update * (self.momentum * v - lr * g_t)
            else:
                new_p = p + update * v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'accum_iters': self.accum_iters}
        base_config = super(SGDWithAcc, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
