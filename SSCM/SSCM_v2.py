# SSCM model with simple feature extractor, modeled by a MLP.

import sys
import math
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from alps.common.model import BaseModel
from alps.common.validation import Validation
from alps.common.tools.trainer import TrainerHook
from alps.core.layer.embedding_layer import EmbeddingLayerV2


def new_act(x):
    return tf.log1p(x)


class DSF(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout=0.0, use_dropout=False, regularizer=None, **kwargs):
        super(DSF, self).__init__(**kwargs)
        self.layer_list = []
        for hidden_size in hidden_units:
            self.layer_list.append(
                Dense(
                    hidden_size,
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=0.1),
                    kernel_regularizer=regularizer,
                    kernel_constraint=tf.keras.constraints.NonNeg()
                )
            )
        self.dropout = Dropout(dropout)
        self.use_dropout = use_dropout

    def call(self, inputs):
        h = inputs
        for layer in self.layer_list[:-1]:
            h = layer(h)
            h = new_act(h)
            if self.use_dropout:
                h = self.dropout(h)
        h = self.layer_list[-1](h)
        return h


class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_units, dropout=0.0, use_dropout=False, regularizer=None, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.layer_list = []
        for hidden_size in hidden_units:
            self.layer_list.append(
                Dense(
                    hidden_size,
                    use_bias=True,
                    kernel_regularizer=regularizer
                )
            )
        self.dropout = Dropout(dropout)
        self.use_dropout = use_dropout

    def call(self, inputs):
        h = inputs
        for layer in self.layer_list[:-1]:
            h = layer(h)
            h = tf.nn.relu(h)
            if self.use_dropout:
                h = self.dropout(h)
        h = self.layer_list[-1](h)
        return h


class SubmodularModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.loss = 0.0
        self.reg_loss = 0.0
        self.prior_loss = 0.0
        self.total_loss = 0.0
        self.average_loss = 0.0
        self.train_loss_steps = self.config.get('train_loss_steps', 1)

    def build_model(self, inputs, labels):
        # model config
        model_def = self.config.model_def
        count = model_def.get_int('count')
        embedding_size = model_def.get_int('embedding_size', 32)
        mlp_hidden_units = model_def.get_list('mlp_hidden_units')
        mlp_hidden_units = [int(i) for i in mlp_hidden_units]
        pred_hidden_units = model_def.get_list('pred_hidden_units')
        pred_hidden_units = [int(i) for i in pred_hidden_units]
        dropout = model_def.get('dropout', 0)
        weight_decay = model_def.get('weight_decay', 0.0)
        with_prior = model_def.get('with_prior', False)
        alpha = model_def.get('alpha', 1)
        if weight_decay > 0:
            l2_regularizer = tf.keras.regularizers.l2(weight_decay)
        else:
            l2_regularizer = None
    
        # input and transformation
        # dense input
        logging.info('dense shape: {}'.format(self.config.x[0].shape))
        dense_dim = self.config.x[0].shape[0] // count
        w4dense = tf.get_variable(
            shape=[dense_dim, embedding_size], initializer=tf.glorot_uniform_initializer(), name='w4dense', regularizer=l2_regularizer
        )
        dense = tf.matmul(tf.reshape(inputs['dense'], shape=[-1, dense_dim]), w4dense)                  # [batch * count, embedding_size]
        dense = tf.reshape(dense, [-1, count, embedding_size])                                          # [batch, count, embedding_size]

        # deep input
        logging.info('deep shape: {}'.format(self.config.x[1].shape))
        deep_feat_cnt = self.config.x[1].shape[0][0]
        deep_embed = EmbeddingLayerV2(embedding_dim=embedding_size, id_size=deep_feat_cnt)
        deep = deep_embed(inputs['deep'], concat=False)                                                 # list of [batch, embedding_size]
        deep = tf.stack(deep, axis=1)                                                                   # [batch, count, embedding_size]

        # wide input
        logging.info('wide shape: {}'.format(self.config.x[2].shape))
        logging.info('wide group number: {}'.format(self.config.x[2].group))
        grp_num = int(self.config.x[2].group) // count
        wide_input_list = [list() for _ in range(grp_num)]
        for i, grp_in in enumerate(inputs['wide']):
            idx = i % grp_num
            wide_input_list[idx].append(grp_in)
        wide = []
        for i in range(grp_num):
            wide_embed = EmbeddingLayerV2(
                embedding_dim=embedding_size,
                id_size=self.config.x[2].shape[i][0],
                name='wide_embed_{}'.format(i)
            )
            grp_out = wide_embed(wide_input_list[i], concat=False)                                      # list of [batch, mbedding_size]
            grp_out = tf.stack(grp_out, axis=1)                                                         # [batch, count, embedding_size]
            wide.append(grp_out)                                                                        # list of [batch, count, embedding_size]
        
        feat = tf.concat([dense, deep] + wide, axis=2)       
        mlp = MLP(mlp_hidden_units, dropout, use_dropout=True, regularizer=l2_regularizer)
        feat = mlp(tf.nn.relu(feat))

        # modular function and concave transformation
        mask = inputs['mask']
        feat = tf.nn.relu(feat)                                                                         # non-negative feature
        feat = feat * mask[:,:,tf.newaxis]
        set_feat = tf.reduce_sum(feat, axis=1, keepdims=True)
        feat = tf.concat([feat, set_feat], axis=1)
        feat = new_act(feat)
        self.set_feat = set_feat[:, -1]

        # DSF
        pred_layer = DSF(pred_hidden_units, dropout, use_dropout=True, regularizer=l2_regularizer)
        logits = pred_layer(feat)                                                                       # [batch, count+1, 1]
        pred = 2 * tf.sigmoid(logits[:,:,0]) - 1
        self.set_pred = pred[:,-1:]
        self.individual_pred = pred[:,:-1]
        self.logits = logits[:,:,0]

        for loss in pred_layer.losses + mlp.losses:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)
        
        # loss
        # adjusted cross entropy for a = f(x) = 2*sigmoid(x) - 1
        # l = (1-y)*(x-log4) + 2*log(1+exp(-x))
        label = labels['label']
        set_logits = logits[:,-1]
        self.cross_entropy = tf.reduce_mean(
            (1-label)*(set_logits-tf.log(4.0)) + 2*tf.log1p(tf.exp(-set_logits))
        )

        if with_prior:
            org_label = labels['org_label']
            sample_weight = labels['weight']
            org_logits = logits[:,:-1,0]
            org_loss = (1-org_label)*(org_logits-tf.log(4.0)) + 2*tf.log1p(tf.exp(-org_logits))
            self.prior_loss = tf.reduce_sum(org_loss*mask*sample_weight) / tf.reduce_sum(mask*sample_weight)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss += tf.reduce_sum(reg_losses)

        self.loss_print = self.cross_entropy + alpha*self.prior_loss
        self.loss = self.loss_print + self.reg_loss

        return self.set_pred
    
    def on_step_end(self, sess, local_step, results):
        if local_step % self.train_loss_steps != 0:
            self.total_loss += results['loss_print']    
        else:
            self.total_loss += results['loss_print']
            self.average_loss = self.total_loss / self.train_loss_steps
            logging.info(
                "local step = %d, %d steps train_loss = %f" % (local_step, self.train_loss_steps, self.average_loss)
            )
            self.total_loss = 0.0
        
    def get_loss(self):
        return self.loss

    def get_metrics(self):
        metrics = {
            'loss_print': self.loss_print,
            'cross_entropy': self.cross_entropy,
            'prior_loss': self.prior_loss,
            'reg_loss': self.reg_loss          
        }
        return metrics
            
    def get_prediction_result(self):
        return self.set_pred
    
    def customer_outputs(self):
        outputs = {
            'set_score': self.set_pred[:,0],
            'individual_score': self.individual_pred,
            'set_feat': self.set_feat,
            'logits': self.logits
        }
        return outputs


class SubmodularModelHook(TrainerHook):
    def __init__(self, trainer):
        super(SubmodularModelHook, self).__init__(trainer)
        self.sess = None
        self.config = trainer.config
        self.local_step = 0

    def on_step_end(self, step, result, feed_dict=None):
        self.local_step += 1
        self._trainer.model.on_step_end(self.sess, self.local_step, result)
    
    def on_train_end(self):
        pass


class SubmodularModelValidation(Validation):
    def __init__(self, config):
        super(SubmodularModelValidation, self).__init__(config)
        self._best_loss_print = sys.float_info.max
        self._loss_print = []

    def prepare(self, tf_raw_session, validation_fetch_op, model_info):
        super(SubmodularModelValidation, self).prepare(tf_raw_session, validation_fetch_op, model_info)
        self._loss_print = []

    def process_result(self, step, label_data, feature_data, other_data, feed_dict, result):
        super(SubmodularModelValidation, self).process_result(step, label_data, feature_data, other_data, feed_dict, result)
        self._loss_print.append(result['loss_print'])
        return {'loss_print': result['loss_print']}

    def validation(self, y_true, y_pred, y_pred_result=None, other_data=None, **kwargs):
        if hasattr(self, 'validation_data') and self.validation_data is not None:
            loss_print = self.validation_data['loss_print']
        else:
            loss_print = self._loss_print
        loss_print = np.mean(loss_print)
        if loss_print < self._best_loss_print:
            self._best_loss_print = loss_print
            return True, {'current_loss_print': loss_print, 'best_loss_print': self._best_loss_print}
        else:
            return False, {'current_loss_print': loss_print, 'best_loss_print': self._best_loss_print}

    @property
    def metric_names(self):
        return ["current_loss_print", "best_loss_print"]

    def get_metric_comparator(self):
        from alps.framework.exporter.base import MetricComparator, Goal
        return MetricComparator("best_loss_print", Goal.MINIMIZE)