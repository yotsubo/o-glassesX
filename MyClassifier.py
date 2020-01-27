from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
import chainer.functions as F
import chainer

#arrnged from
#https://github.com/shunk031/chainer-FocalLoss/blob/master/focal_loss.py
def focal_loss(x, t, alpha=0.2, gamma=2, eps=1e-7):
    xp = chainer.cuda.get_array_module(t)
    logit = F.softmax(x)
    logit = F.clip(logit, x_min=eps, x_max=1-eps)
    class_num = x.shape[1]

    alpha = 1.0
    gamma = 5

    t_onehot = xp.eye(class_num)[t]

    loss_ce = -1 * F.mean(t_onehot * F.log(logit))*class_num
    loss_focal = loss_ce * alpha * F.mean(t_onehot *( (1 - logit) ** gamma))*class_num
    #print loss_focal,loss_ce
    #exit()
    return loss_focal#F.sqrt(loss_focal)

class MyClassifier(link.Chain):

    """A simple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """

    compute_accuracy = True

    def __init__(self, predictor,
                 #lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 lossfun=focal_loss,
                 accfun=accuracy.accuracy):
        super(MyClassifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.

        The all elements of ``args`` but last one are features and
        the last element corresponds to ground truth labels.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        
        #print self.y.shape
        #print self.loss
        #print self.loss.shape,self.loss.dtype
        #exit()

        summary = F.classification_summary(self.y, t, beta = 1.0)
        precision = summary[0]
        recall = summary[1]
        f_value = summary[2]

        reporter.report({'loss': self.loss}, self)
        reporter.report(dict(('precision_%d' % i, val) for i, val in enumerate(precision)), self)
        reporter.report(dict(('recall_%d' % i, val) for i, val in enumerate(recall)), self)
        reporter.report(dict(('f_value_%d' % i, val) for i, val in enumerate(f_value)), self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
        
