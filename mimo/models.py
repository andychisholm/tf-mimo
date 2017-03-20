from __future__ import absolute_import
from __future__ import division
from six.moves import xrange
import random
import numpy as np
import math
from itertools import tee
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.rnn import _rnn_step
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_impl import _compute_sampled_logits
from tensorflow.python.util import nest


#from tensorflow.contrib.learn.python.learn.models import bidirectional_rnn

from tensorflow.contrib.rnn.python.ops.core_rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn.python.ops.core_rnn import static_rnn

from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import OutputProjectionWrapper, GRUCell, MultiRNNCell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
_linear = core_rnn_cell_impl._linear

from tensorflow.contrib.rnn import LayerNormBasicLSTMCell

VERSION=1

def sampled_sigmoid_loss(weights, biases, inputs, labels, num_sampled,
                         num_classes, num_true=2,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         name="sampled_softmax_loss"):
    logits, labels = _compute_sampled_logits(
        weights, biases, inputs, labels, num_sampled, num_classes,
        num_true=num_true,
        sampled_values=sampled_values,
        subtract_log_q=True,
        remove_accidental_hits=remove_accidental_hits,
        partition_strategy=partition_strategy,
        name=name)
    sampled_losses = nn_ops.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    return sampled_losses

def attention_decoder(decoder_inputs, sequence_length, initial_state, attention_matrix, cell,
                      output_size=None, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False):
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if not attention_matrix.get_shape()[1:].is_fully_defined():
        raise ValueError("Shape of attention matrix must be known: %s" % attention_matrix.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder"):
        #batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if decoder_inputs[0].get_shape().ndims != 1:
            (fixed_batch_size, input_size) = decoder_inputs[0].get_shape().with_rank(2)
            if input_size.value is None:
                raise ValueError(
                    "Input size (second dimension of inputs[0]) must be accessible via "
                    "shape inference, but saw value None.")
        else:
            fixed_batch_size = decoder_inputs[0].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(decoder_inputs[0])[0]

        if sequence_length is not None:
            sequence_length = math_ops.to_int32(sequence_length)
            zero_output = array_ops.zeros(tf.stack([batch_size, cell.output_size]), decoder_inputs[0].dtype)
            zero_output.set_shape(tensor_shape.TensorShape([fixed_batch_size.value, cell.output_size]))
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        # ATTENTION COMPUTATION
        
        attn_size = attention_matrix.get_shape()[-1].value
        batch_attn_size = tf.stack([batch_size, attn_size])

        def _attention_dot(query, states):
            """Put attention masks on hidden using hidden_features and query."""
            attn_length = states.get_shape()[1].value

            hidden = array_ops.reshape(states, [-1, attn_length, 1, attn_size])
            y = _linear(query, attn_size, True)

            # dot product to produce the attention over incoming states
            s = tf.reduce_sum(tf.multiply(states, tf.expand_dims(y, 1)), -1)
            a = nn_ops.softmax(s)

            # Now calculate the attention-weighted vector d.
            d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
            d = array_ops.reshape(d, [-1, attn_size])
            return d

        def _attention_concat(query, states):
            """Put attention masks on hidden using hidden_features and query."""
            v = variable_scope.get_variable("AttnV", [attn_size])
            k = variable_scope.get_variable("AttnW", [1, 1, attn_size, attn_size])

            # attn is v^T * tanh(W1*h_t + U*q)
            
            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            attn_length = states.get_shape()[1].value
            hidden = array_ops.reshape(states, [-1, attn_length, 1, attn_size])
            hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")

            y = _linear(query, attn_size, True)
            y = array_ops.reshape(y, [-1, 1, 1, attn_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + y), [2, 3])
            a = nn_ops.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = math_ops.reduce_sum(array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
            d = array_ops.reshape(d, [-1, attn_size])
            return d

        _attention = _attention_dot

        def attention(query):
            if nest.is_sequence(query):
                query_list = nest.flatten(query)
                for q in query_list:
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)

            outer_states = tf.unstack(attention_matrix, axis=1)

            inner_states = []
            for i, states in enumerate(outer_states):
                with variable_scope.variable_scope("Attention_outer", reuse=i>0):
                    inner_states.append(_attention(query, states))

            with variable_scope.variable_scope("Attention_inner"):
                return _attention(query, tf.stack(inner_states, 1))

        state = cell.zero_state(batch_size, dtype) if initial_state == None else initial_state
        outputs = []
        prev = None

        attns = array_ops.zeros(batch_attn_size, dtype=dtype)
        attns.set_shape([None, attn_size])
        
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = _linear([inp] + [attns], input_size, True)

            if sequence_length is not None:
                call_cell = lambda: cell(x, state)
                if sequence_length is not None:
                    cell_output, state = _rnn_step(
                      i, sequence_length, min_sequence_length, max_sequence_length, zero_output, state, call_cell, cell.state_size)
            else:
                cell_output, state = cell(x, state)


            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = _linear([cell_output] + [attns], output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

class Encoder(object):
    def __init__(self, embedding, size, num_layers, max_length, dtype, **kwargs):
        self.embedding = embedding
        self.size = size
        self.num_layers = num_layers
        self.cell = GRUCell(self.size)
        if self.num_layers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers)

        max_length += 2 # account for _GO and _EOS

        self.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, shape=[None], name="encoder_lengths"))
        self.inputs = kwargs.get('inputs', [tf.placeholder(tf.int32, shape=[None], name="encoder_input{0}".format(i)) for i in xrange(max_length)])
        self.weights = kwargs.get('weights', [tf.placeholder(tf.float32, shape=[None], name="encoder_weight{0}".format(i)) for i in xrange(max_length)])

        inputs = [embedding_ops.embedding_lookup(embedding, i) for i in self.inputs]

        self.outputs, self.state = static_rnn(self.cell, inputs, sequence_length=self.lengths, dtype=dtype)
        top_states = [array_ops.reshape(e, [-1, 1, self.cell.output_size]) for e in self.outputs]

        # BiRNN
        #self.outputs, self.state_fw, self.state_bw = static_bidirectional_rnn(self.cell, self.cell, inputs, sequence_length=self.lengths, dtype=dtype)
        #self.state = self.state_fw + self.state_bw # aggregate fw+bw state (use this)
        #top_states = [array_ops.reshape(e, [-1, 1, self.cell.output_size*2]) for e in self.outputs]

        #self.outputs = [tf.add(*tf.split(1, 2, o)) for o in self.outputs] # concat fw + bw states
        #self.state = tf.concat([self.state_fw, self.state_bw], 1) # concatenate fw+bw states

        self.attention_states = array_ops.concat(top_states, 1)

    def transform_batch(self, feed, data):
        data = [([GO_ID]+seq+[EOS_ID]) if seq else [] for seq in data]
        feed[self.lengths.name] = [len(seq) for seq in data]
        for seq in data:
            if len(seq) > len(self.inputs):
                raise ValueError('Input sequence length (%i) larger than max encoder length (%i)' % (len(seq), len(self.inputs)))
            for i in xrange(len(self.inputs)):
                feed.setdefault(self.inputs[i].name, []).append(seq[i] if i < len(seq) else PAD_ID)
                feed.setdefault(self.weights[i].name, []).append(1.0 if i < len(seq) else 0.0)

class Decoder(object):
    def __init__(self,
                 embedding,
                 initial_state,
                 attention_states,
                 size,
                 num_layers,
                 max_length,
                 num_samples=512,
                 feed_previous=False,
                 update_embedding_for_previous=True,
                 dtype=dtypes.float32,
                 scope=None,
                 initial_state_attention=False,
                 **kwargs):
        # account for _GO and _EOS
        self.max_length = max_length + 2

        self.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, shape=[None], name="decoder_lengths"))
        self.inputs = kwargs.get('inputs', [tf.placeholder(tf.int32, shape=[None], name="decoder_input{0}".format(i)) for i in xrange(self.max_length)])
        self.weights = kwargs.get('weights', [tf.placeholder(tf.float32, shape=[None], name="decoder_weight{0}".format(i)) for i in xrange(self.max_length)])
        
        self.targets = [self.inputs[i + 1] for i in xrange(len(self.inputs) - 1)]
        self.targets.append(tf.zeros_like(self.targets[0]))

        num_symbols = embedding.get_shape()[0].value
        output_projection = None
        loss_function = None

        self.num_layers = num_layers
        self.cell = GRUCell(size) #tf.contrib.rnn.LayerNormBasicLSTMCell(size)
        if self.num_layers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * self.num_layers)

        self.feed_previous = feed_previous
        
        if num_samples > 0 and num_samples < num_symbols:
            #with tf.device('/cpu:0'):
            w = tf.get_variable('proj_w', [self.cell.output_size, num_symbols])
            w_t = tf.transpose(w)
            b = tf.get_variable('proj_b', [num_symbols])
            output_projection = (w, b)
            def sampled_loss(labels, inputs):
                #with tf.device('/cpu:0'):
                labels = tf.reshape(labels, [-1, 1])
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.nn.sampled_softmax_loss(
                    weights=local_w_t,
                    biases=local_b,
                    labels=labels,
                    inputs=local_inputs,
                    num_sampled=num_samples,
                    num_classes=num_symbols)
            loss_function = sampled_loss
        
        output_size = None
        if output_projection is None:
            self.cell = OutputProjectionWrapper(self.cell, num_symbols)
            output_size = num_symbols
        
        if output_size is None:
            output_size = self.cell.output_size
        if output_projection is not None:
            proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
            proj_weights.get_shape().assert_is_compatible_with([self.cell.output_size, num_symbols])
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
            loop_fn_factory = self._extract_argmax_and_embed #self._extract_grumble_softmax_embed
            loop_function = loop_fn_factory(embedding, output_projection, update_embedding_for_previous) if feed_previous else None

            emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in self.inputs]
            self.outputs, self.state = attention_decoder(
                emb_inp,
                self.lengths,
                initial_state,
                attention_states,
                self.cell,
                output_size=output_size,
                loop_function=loop_function,
                initial_state_attention=initial_state_attention)

        targets = [self.inputs[i + 1] for i in xrange(len(self.inputs) - 1)]
        targets.append(tf.zeros_like(self.inputs[-1]))
        
        # loss for each instance in batch
        self.instance_loss = sequence_loss_by_example(self.outputs, targets, self.weights, softmax_loss_function=loss_function)

        # aggregated average loss per instance for batch
        self.loss = tf.reduce_sum(self.instance_loss) / math_ops.cast(array_ops.shape(targets[0])[0], self.instance_loss.dtype)

        if output_projection is not None:
            self.projected_output = [tf.matmul(o, output_projection[0]) + output_projection[1] for o in self.outputs]
            self.decoded_outputs = tf.unstack(tf.argmax(tf.stack(self.projected_output), 2))
            self.decoded_output_prob = tf.reduce_max(tf.nn.softmax(tf.stack(self.projected_output)), 2)
        else:
            self.decoded_outputs = tf.unstack(tf.argmax(tf.stack(self.outputs), 2))
            self.decoded_output_prob = tf.reduce_max(tf.nn.softmax(tf.stack(self.outputs)), 2)

        self.decoded_lenghts = tf.reduce_sum(tf.sign(tf.transpose(tf.stack(self.decoded_outputs))), 1)
        self.decoded_batch = tf.transpose(tf.stack(self.decoded_outputs))
        self.decoded_batch_probs = tf.transpose(tf.stack(self.decoded_output_prob))

    def decode_batch(self, session, feed, index_vocab):
        outputs, probs = session.run((self.decoded_batch, self.decoded_batch_probs), feed)
        return self.process_decode(outputs, probs, index_vocab)

    def process_decode(self, batch_outputs, batch_probs, index_vocab):
        results = []
        for outputs, probs in zip(batch_outputs, batch_probs):
            result = []
            found_eos = False
            for t in outputs:
                if t != EOS_ID:
                    result.append(index_vocab[t])
                else:
                    found_eos = True
                    break
            result = result if found_eos else []
            results.append({
                'sequence': result,
                'confidences': probs[:len(result)]
            })
        return results
    
    def transform_batch(self, feed, data):
        if self.feed_previous:
            # TODO: assumes feed_previous == forward_only
            feed[self.inputs[0].name] = [GO_ID]*len(data)
            feed[self.lengths.name] = [self.max_length for _ in xrange(len(data))]
        else:
            data = [[GO_ID]+seq+[EOS_ID] for seq in data]
            feed[self.lengths.name] = [len(seq) for seq in data]
            for seq in data:
                if len(seq) > len(self.inputs):
                    raise ValueError('Input sequence length (%i) larger than max decoder length (%i)' % (len(seq), len(self.inputs)))
                for i in xrange(len(self.inputs)):
                    feed.setdefault(self.inputs[i].name, []).append(seq[i] if i < len(seq) else PAD_ID)
                    feed.setdefault(self.weights[i].name, []).append(1.0 if i < len(seq) else 0.0)

    @staticmethod
    def _extract_argmax_and_embed(embedding, output_projection=None, update_embedding=True):
        def loop_function(prev, _):
            if output_projection is not None:
                prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
            prev_symbol = math_ops.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of embedding_lookup.
            emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
            if not update_embedding:
                emb_prev = array_ops.stop_gradient(emb_prev)
            return emb_prev
        return loop_function

    @staticmethod
    def _extract_grumble_softmax_embed(embedding, output_projection=None, update_embedding=True):
        def sample_gumbel(shape, eps=1e-20):
            """Sample from Gumbel(0, 1)"""
            U = tf.random_uniform(shape, minval=0, maxval=1)
            return -tf.log(-tf.log(U + eps) + eps)

        def gumbel_softmax_sample(logits, temperature):
            """ Draw a sample from the Gumbel-Softmax distribution"""
            y = logits + sample_gumbel(tf.shape(logits))
            return tf.nn.softmax(y / temperature)

        def loop_function(prev, _):
            if output_projection is not None:
                prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
                prev_probs = gumbel_softmax_sample(prev, 0.75)

            emb_prev = tf.reduce_sum(embedding * tf.expand_dims(prev_probs, -1), 1)
            if not update_embedding:
                emb_prev = array_ops.stop_gradient(emb_prev)
            return emb_prev
        return loop_function

class MultiEncoder(object):
    def __init__(self, vocab_size, batch_size, embedding, share_encoder, encoder_args=None):
        encoder_args = encoder_args or {}
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.encoder_names = encoder_args.keys()

        with tf.variable_scope("multi_encoder"):
            self.encoders = {}
            for i, (k, args) in enumerate(sorted(encoder_args.iteritems())):
                reuse = share_encoder and i != 0
                vs_key = k.replace(' ', '_')
                with tf.variable_scope("encoder"+('' if share_encoder else '_'+vs_key), reuse=reuse):
                    print 'Building Encoder ['+k+']...' + (' (shared params)' if reuse else '')
                    encoder = Encoder(embedding=embedding, dtype=tf.float32, **args)
                with tf.variable_scope("encoder_"+vs_key):
                    weight = args.get('weight', tf.placeholder(tf.float32, shape=[None, 1], name="encoder_{0}_weight".format(k)))
                    self.encoders[k] = {
                        'encoder': encoder,
                        'weight': weight,
                    }
            # TODO: may need to apply weights explicitly, i.e. (tf.reduce_sum(..)*weight)
            # sum individual encoder outputs to get aggregate attention
            states = [tf.reduce_sum(self.encoders[k]['encoder'].attention_states, 1, True) for k in self.encoder_names]
            self.attention_states = array_ops.concat(states, 1)
            self.attention_matrix = tf.stack([self.encoders[k]['encoder'].attention_states for k in self.encoder_names], 1)

            # sum individual encoder states to get aggregate state
            self.state = tf.reduce_sum(tf.stack([self.encoders[k]['encoder'].state for k in self.encoder_names]), 0)

    def transform_batch(self, feed, data):
        for k, e in self.encoders.iteritems():
            instances = [instance.get(k, []) for instance in data]
            feed[e['weight'].name] = [[1.0 if i else 0.0] for i in instances]
            e['encoder'].transform_batch(feed, instances)

class MultiDecoder(object):
    def __init__(self,
        encoder, vocab_size, batch_size, embedding, share_decoder,
        num_samples=512, forward_only=False, propagate_state=False,
        decoder_args=None):

        decoder_args = decoder_args or {}
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.decoder_names = decoder_args.keys()
        self.loss = None
        self.forward_only = forward_only

        with tf.variable_scope("multi_decoder"):
            self.decoders = {}
            for i, (k, args) in enumerate(sorted(decoder_args.iteritems())):
                reuse = share_decoder and i != 0
                vs_key = k.replace(' ', '_').replace(',', '')
                with tf.variable_scope("decoder"+('' if share_decoder else '_'+vs_key), reuse=reuse):
                    print 'Building Decoder: ['+k+']...' + (' (shared params)' if reuse else '')
                    state = self.encoder.state if propagate_state else None
                    decoder = Decoder(embedding, state, self.encoder.attention_matrix, feed_previous=forward_only, **args)

                with tf.variable_scope("decoder_"+vs_key):
                    weight = args.get('weight', tf.placeholder(tf.float32, shape=[None], name="decoder_{0}_weight".format(vs_key)))

                    # instance weighted
                    instance_loss = tf.multiply(weight, decoder.instance_loss)

                    # aggregate loss over batch
                    weighted_loss = tf.reduce_sum(instance_loss) / tf.reduce_sum(weight)

                    if self.loss == None:
                        self.loss = weighted_loss
                    else:
                        self.loss += weighted_loss

                    self.decoders[k] = {
                        'decoder': decoder,
                        'weight': weight,
                        'instance_loss': instance_loss,
                        'weighted_loss': weighted_loss
                    }

    def decode_batch(self, session, feed, index_vocab):
        batch_by_decoder = {k:d['decoder'].decode_batch(session, feed, index_vocab) for k, d in self.decoders.iteritems()}
        batch_size = len(batch_by_decoder.values()[0])
        return [{k:batch_by_decoder[k][i] for k in self.decoders.iterkeys()} for i in xrange(batch_size)]

    def transform_batch(self, feed, data):
        for k, d in self.decoders.iteritems():
            instances = [instance.get(k, []) for instance in data]
            feed[d['weight'].name] = [(1.0 if i else 0.0) for i in instances]
            d['decoder'].transform_batch(feed, instances)
            
class MIMO(object):
    def __init__(self,
        vocab_size, embedding_size, batch_size,
        share_encoder, share_decoder,
        num_samples=512, forward_only=False,
        encoder_args=None,
        decoder_args=None):
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.forward_only = forward_only

        with tf.variable_scope("mimo") as vs:
            with tf.device("/cpu:0"):
                sqrt3 = math.sqrt(3)
                init = tf.random_uniform_initializer(-sqrt3, sqrt3)
                self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], initializer=init)

            self.encoder = MultiEncoder(
                vocab_size=vocab_size,
                batch_size=batch_size,
                embedding=self.embedding,
                share_encoder=share_encoder,
                encoder_args=encoder_args)

            self.decoder = MultiDecoder(
                self.encoder,
                vocab_size=vocab_size,
                batch_size=batch_size,
                embedding=self.embedding,
                share_decoder=share_decoder,
                num_samples=num_samples,
                forward_only=forward_only,
                decoder_args=decoder_args)

    def decode_batch(self, session, feed, index_vocab):
        return self.decoder.decode_batch(session, feed, index_vocab)

    @property
    def loss(self):
        return self.decoder.loss

    def transform_batch(self, feed, data):
        source_seqs, target_seqs = data
        self.encoder.transform_batch(feed, source_seqs)
        self.decoder.transform_batch(feed, target_seqs)
