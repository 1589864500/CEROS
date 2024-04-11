# SSCM model with complex feature extractor, modeled by SMPLayer which is much suitable for tabular data analysis, 
# The feature extractor refers to the following literature in IEEE TKDE 2023:
# Zheng Q, Peng Z, Dang Z, et al. Deep tabular data modeling with dual-route structure-adaptive graph networks[J]. IEEE Transactions on Knowledge and Data Engineering, 2023.

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


def top_k_mask(X, k):
    n = X.shape[1]
    top_k_indices = tf.math.top_k(X, k).indices
    mask = tf.reduce_sum(tf.one_hot(top_k_indices, n), axis=1)
    return mask


class SMPLayer(object):
    def __init__(self, embedding_size, n_layer):
        self.embedding_size = embedding_size
        self.w = tf.get_variable(shape=[1, 2],
                                 initializer=tf.initializers.glorot_uniform,
                                 dtype=tf.float32,
                                 name="w" + str(n_layer))
        self.w_mlp = tf.get_variable(
            shape=[embedding_size, embedding_size],
            initializer=tf.initializers.glorot_uniform,
            dtype=tf.float32,
            name="w_mlp" + str(n_layer))
        self.w_fc = tf.get_variable(shape=[embedding_size, embedding_size],
                                    initializer=tf.initializers.glorot_uniform,
                                    dtype=tf.float32,
                                    name="w_fc" + str(n_layer))
        self.b_mlp = tf.get_variable(initializer=tf.zeros([embedding_size]),
                                     dtype=tf.float32,
                                     name="b_mlp" + str(n_layer))
        self.b_fc = tf.get_variable(initializer=tf.zeros([embedding_size]),
                                    dtype=tf.float32,
                                    name="b_fc" + str(n_layer))
        self.params = [self.w, self.w_mlp, self.w_fc]

    def __call__(self, input, query, key, adj_topk_mask, adj, n_node):
        query_vec = tf.tensordot(input, query, 1)
        key_vec = tf.tensordot(input, key, 1)
        attention = tf.matmul(query_vec, tf.transpose(
            key_vec, perm=[0, 2, 1])) / math.sqrt(self.embedding_size)
        attention = attention * tf.cast(adj_topk_mask, attention.dtype)
        adj_attention = self.w[0][0] * attention + self.w[0][1] * adj
        sf_attention = tf.nn.softmax(adj_attention)
        final_attention = sf_attention + tf.eye(n_node)
        out = tf.keras.layers.PReLU()(
            tf.tensordot(tf.matmul(final_attention, input), self.w_mlp, 1) +
            self.b_mlp)
        prediction = tf.matmul(tf.reduce_sum(out, 1), self.w_fc) + self.b_fc

        reg_loss = 0
        for param in self.params:
            reg_loss += tf.nn.l2_loss(param)

        return out, prediction, reg_loss


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
            grp_out = wide_embed(wide_input_list[i])                                                    # [batch, count * embedding_size]
            wide.append(tf.reshape(grp_out, shape=[-1, embedding_size]))                                # list of [batch * count, embedding_size]

        # parameters initialization for SMP
        n_node = grp_num
        low_rank = model_def.get_int('low_rank', 8)
        top_k = model_def.get_int('top_k', 33)
        n_layers = model_def.get('n_layers', 2)
        u = tf.get_variable(
            shape=[n_node, low_rank], initializer=tf.initializers.glorot_uniform, dtype=tf.float32, name="u", regularizer=l2_regularizer
        )
        v = tf.get_variable(
            shape=[low_rank, n_node], initializer=tf.initializers.glorot_uniform, dtype=tf.float32, name="v", regularizer=l2_regularizer)
        query = tf.get_variable(
            shape=[embedding_size, embedding_size], initializer=tf.initializers.glorot_uniform, dtype=tf.float32, name="query", regularizer=l2_regularizer
        )
        key = tf.get_variable(
            shape=[embedding_size, embedding_size], initializer=tf.initializers.glorot_uniform, dtype=tf.float32, name="key", regularizer=l2_regularizer
        )
        adj = tf.matmul(u, v)
        adj_topk_mask = top_k_mask(adj, top_k)
        adj = adj*tf.cast(adj_topk_mask, adj.dtype)
        # SMP_layer forward
        smp_out = []
        smp_h = tf.stack(wide, axis=1)                                                                  # [batch * count, grp_num, embedding_size]
        smp_out.append(tf.reshape(smp_h, [-1, count, n_node*embedding_size]))                           # [batch, count, grp_num * embedding_size]
        for i in range(n_layers):
            smplayer = SMPLayer(embedding_size, i)
            smp_h, _, reg_loss = smplayer(smp_h, query, key, adj_topk_mask, adj, n_node)
            self.reg_loss += weight_decay * reg_loss
            smp_out.append(tf.reshape(smp_h, [-1, count, n_node*embedding_size]))

        feat = tf.concat([dense, deep] + smp_out, axis=2)        
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